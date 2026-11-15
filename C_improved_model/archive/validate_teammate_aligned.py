"""Evaluate the aligned model with patient-disjoint folds and causal features."""
from pathlib import Path
import numpy as np
import pandas as pd

from teammate_aligned_model import AlignedModel, History, HORIZONS, KEYS, TARGETS, load_training
from visit_history_model import read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/teammate_aligned_results'


def main():
    train_rows, train_features = load_training(DATA)
    clinical, omics = read_data(DATA)
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS)
    requests = training_rows(clinical, omics, 'union')
    assignments = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    requests = requests.merge(assignments, on='patient_id', validate='many_to_one')
    assert requests.fold.notna().all()
    history = History()
    history.update(pd.concat([clinical[KEYS], omics], ignore_index=True), proteins)
    features = np.array([history.features(p, t, h) for p, t, h in
                         requests[KEYS + ['horizon']].itertuples(index=False, name=None)], dtype=np.float32)
    outputs = []
    for fold in range(5):
        valid = requests.fold.eq(fold).to_numpy()
        held = set(requests.loc[valid, 'patient_id'])
        train = ~train_rows.patient_id.isin(held).to_numpy()
        assert set(train_rows.loc[train, 'patient_id']).isdisjoint(held)
        model = AlignedModel().fit(train_rows.loc[train].reset_index(drop=True), train_features[train])
        part = requests.loc[valid].copy()
        targets = part.target.map({name: i for i, name in enumerate(TARGETS)}).to_numpy()
        heads = part.horizon.map({h: i * 4 for i, h in enumerate(HORIZONS)}).to_numpy() + targets
        predictions = model.predict_features(features[valid], targets, heads)
        assert np.isfinite(predictions).all() and (predictions >= 0).all()
        part['prediction'] = predictions
        part['loss'] = 200 * abs(part.rating - predictions) / (part.rating + predictions + 2)
        outputs.append(part)
        print('fold', fold, 'patients', len(held), 'requests', len(part), 'SMAPE+1', round(part.loss.mean(), 4), flush=True)
    result = pd.concat(outputs, ignore_index=True)
    assert len(result) == len(requests)
    assert not result.duplicated(['patient_id', 'visit_month', 'target', 'horizon']).any()
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'oof_predictions.csv', index=False)
    result.groupby(['target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    scores = result.groupby('fold').loss.mean()
    print('fold mean', round(scores.mean(), 4), 'pooled OOF', round(result.loss.mean(), 4), flush=True)


if __name__ == '__main__':
    main()

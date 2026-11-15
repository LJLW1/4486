"""Patient-fold comparison of directly supplied protein features with V12."""
from pathlib import Path
import numpy as np
import pandas as pd
from teammate_aligned_model import History, HORIZONS, KEYS, TARGETS, load_training
from teammate_neural_proteins import ProteinModel, protein_matrix
from visit_history_model import read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/neural_direct_protein_results'


def main():
    train_rows, train_features = load_training(DATA)
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    train_proteins = protein_matrix(train_rows, proteins)
    train_proteins[train_rows.source.eq(1).to_numpy()] = np.nan
    clinical, omics = read_data(DATA)
    rows = training_rows(clinical, omics, 'union')
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    rows = rows.merge(folds, on='patient_id', validate='many_to_one')
    history = History()
    history.update(pd.concat([clinical[KEYS], omics], ignore_index=True), proteins)
    features = np.array([history.features(p, t, h) for p, t, h in
                         rows[KEYS + ['horizon']].itertuples(index=False, name=None)], dtype=np.float32)
    sample_proteins = protein_matrix(rows, proteins)
    output = []
    for fold in range(5):
        valid = rows.fold.eq(fold).to_numpy()
        held = set(rows.loc[valid, 'patient_id'])
        train = ~train_rows.patient_id.isin(held).to_numpy()
        assert set(train_rows.loc[train, 'patient_id']).isdisjoint(held)
        model = ProteinModel().fit(train_rows.loc[train].reset_index(drop=True), train_features[train], train_proteins[train])
        part = rows.loc[valid].copy()
        targets = part.target.map({name: i for i, name in enumerate(TARGETS)}).to_numpy()
        heads = part.horizon.map({h: 4 * i for i, h in enumerate(HORIZONS)}).to_numpy() + targets
        part['prediction'] = model.predict_features(features[valid], targets, heads, sample_proteins[valid])
        assert np.isfinite(part.prediction).all() and part.prediction.ge(0).all()
        part['loss'] = 200 * abs(part.rating - part.prediction) / (part.rating + part.prediction + 2)
        output.append(part)
        print('fold', fold, 'score', part.loss.mean(), flush=True)
    result = pd.concat(output, ignore_index=True)
    assert len(result) == len(rows) and not result.duplicated(['patient_id', 'visit_month', 'target', 'horizon']).any()
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'oof_predictions.csv', index=False)
    result.groupby(['target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    print('fold mean', result.groupby('fold').loss.mean().mean(), 'pooled', result.loss.mean(), flush=True)


if __name__ == '__main__':
    main()

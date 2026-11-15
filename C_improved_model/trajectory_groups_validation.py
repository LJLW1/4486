"""Patient-fold validation of the fifth-place-inspired grouped trajectories."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from trajectory_groups_model import assign_groups, fit_curves, predict_rows, training_rows
from visit_history_model import read_data, training_rows as evaluation_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/trajectory_groups_results'


def main():
    clinical, omics = read_data(DATA)
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=['patient_id', 'visit_month'])
    raw_folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv').set_index('patient_id').fold
    patients = np.sort(raw_folds.index.unique())
    splits = {'original': raw_folds}
    for seed in (42, 2024):
        assignment = pd.Series(index=patients, dtype=int)
        for fold, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=seed).split(patients)):
            assignment.loc[patients[valid]] = fold
        splits[str(seed)] = assignment
    results = []
    for stream in ('union', 'omics'):
        visits = pd.concat([clinical[['patient_id', 'visit_month']], omics]).drop_duplicates() if stream == 'union' else omics
        rows = evaluation_rows(clinical, omics, stream)
        rows['group'] = assign_groups(rows, visits)
        training = training_rows(clinical, proteins, clinical[['patient_id', 'visit_month']])
        for name, assignment in splits.items():
            rows['fold'] = rows.patient_id.map(assignment)
            training['fold'] = training.patient_id.map(assignment)
            assert rows.fold.notna().all()
            for fold in range(5):
                train = training[training.fold.ne(fold)]
                valid = rows[rows.fold.eq(fold)]
                assert set(train.patient_id).isdisjoint(valid.patient_id)
                pred = predict_rows(valid, fit_curves(train))
                assert np.isfinite(pred).all() and (pred >= 0).all()
                part = valid[['patient_id', 'visit_month', 'target', 'horizon', 'rating', 'fold', 'group']].copy()
                part['stream'], part['split'], part['prediction'] = stream, name, pred
                part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
                results.append(part)
                print(stream, name, fold, round(part.loss.mean(), 4), flush=True)
    result = pd.concat(results, ignore_index=True)
    assert not result.duplicated(['stream', 'split', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'oof.csv', index=False)
    result.groupby(['stream', 'split']).loss.agg(['mean', 'size']).to_csv(OUT / 'summary.csv')
    result.groupby(['stream', 'split', 'group']).agg(score=('loss', 'mean'), patients=('patient_id', 'nunique'), requests=('loss', 'size')).to_csv(OUT / 'groups.csv')


if __name__ == '__main__':
    main()

"""Test clinical-origin training requests using the unchanged V6 predictor."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from visit_history_model import KEYS, read_data, training_rows, history_codes
from protein_history_experiment import protein_values
from horizon_kernel_experiment import predict

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/top5_results'
MONTHS = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]


def main():
    clinical, omics = read_data(DATA)
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv').set_index('patient_id').fold
    patients = np.sort(folds.index.unique())
    splits = {'original': folds}
    for seed in (42, 2024):
        assignment = pd.Series(index=patients, dtype=int)
        for fold, (_, val) in enumerate(KFold(5, shuffle=True, random_state=seed).split(patients)):
            assignment.loc[patients[val]] = fold
        splits[str(seed)] = assignment
    results = []
    for stream in ('union', 'omics'):
        validation = training_rows(clinical, omics, stream)
        train_rows = training_rows(clinical, clinical[KEYS], stream)
        train_rows = train_rows[train_rows.visit_month.isin(MONTHS)].copy()
        visible = pd.concat([clinical[KEYS], omics]).drop_duplicates() if stream == 'union' else omics
        train_rows['history'] = history_codes(train_rows, visible)
        for split, assignment in splits.items():
            validation['fold'] = validation.patient_id.map(assignment)
            train_rows['fold'] = train_rows.patient_id.map(assignment)
            for fold in range(5):
                train, valid = train_rows[train_rows.fold.ne(fold)], validation[validation.fold.eq(fold)]
                train_patients = set(train.patient_id)
                assert train_patients.isdisjoint(valid.patient_id)
                z = protein_values(wide, train_patients, ['P05060', 'O15240'])
                pred = predict(train, valid, z, 'horizon6')
                assert np.isfinite(pred).all() and (pred >= 0).all()
                part = valid[['patient_id', 'visit_month', 'target', 'horizon', 'rating', 'fold']].copy()
                part['split'], part['stream'], part['prediction'] = split, stream, pred
                part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
                results.append(part)
                print(stream, split, fold, len(train), round(part.loss.mean(), 4), flush=True)
    result = pd.concat(results, ignore_index=True)
    assert not result.duplicated(['stream', 'split', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    result.to_csv(OUT / 'winner_v6_rows_oof.csv', index=False)
    result.groupby(['stream', 'split']).loss.agg(['mean', 'size']).to_csv(OUT / 'winner_v6_rows_summary.csv')


if __name__ == '__main__':
    main()

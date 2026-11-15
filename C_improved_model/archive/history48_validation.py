"""Compare V6 with a causal 48-month visit flag under repeated patient splits."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from horizon_kernel_experiment import predict
from protein_history_experiment import protein_values
from visit_history_model import KEYS, read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/history48_results'


def history48(rows, visits):
    seen = set(visits.loc[visits.visit_month.eq(48), 'patient_id'])
    state = np.where(rows.visit_month.lt(48), 0,
                     np.where(rows.patient_id.isin(seen), 2, 1))
    return rows.history.to_numpy() + 9 * state


def main():
    clinical, omics = read_data(DATA)
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    original = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    patients = np.sort(original.patient_id.unique())
    splits = {'original': original.set_index('patient_id').fold}
    for seed in (42, 2024):
        assignments = pd.Series(index=patients, dtype=int)
        for fold, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=seed).split(patients)):
            assignments.loc[patients[valid]] = fold
        splits[f'seed_{seed}'] = assignments.astype(int)
    output = []
    for stream in ('union', 'omics'):
        rows = training_rows(clinical, omics, stream)
        visits = pd.concat([clinical[KEYS], omics]).drop_duplicates() if stream == 'union' else omics
        rows['history48'] = history48(rows, visits)
        rows['seen48'] = rows.patient_id.isin(set(visits.loc[visits.visit_month.eq(48), 'patient_id'])) & rows.visit_month.ge(48)
        current_protein = wide.reindex(pd.MultiIndex.from_frame(rows[KEYS]))[['P05060', 'O15240']]
        rows['protein_count'] = current_protein.notna().sum(axis=1).to_numpy()
        assert set(rows.patient_id).issubset(patients)
        for split, assignments in splits.items():
            rows['fold'] = rows.patient_id.map(assignments)
            assert rows.fold.notna().all()
            for fold in range(5):
                train, valid = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
                train_patients = set(train.patient_id)
                assert train_patients.isdisjoint(valid.patient_id)
                z = protein_values(wide, train_patients, ['P05060', 'O15240'])
                baseline = predict(train, valid, z, 'horizon6')
                tr48, va48 = train.copy(), valid.copy()
                tr48['history'], va48['history'] = tr48.history48, va48.history48
                candidate = predict(tr48, va48, z, 'horizon6')
                # Before month 48, retain V6's full reference pool exactly.
                candidate[valid.visit_month.lt(48).to_numpy()] = baseline[valid.visit_month.lt(48).to_numpy()]
                assert np.array_equal(baseline[valid.visit_month.lt(48)], candidate[valid.visit_month.lt(48)])
                conservative = np.where(valid.seen48.to_numpy(), candidate, baseline)
                for name, pred in [('v6', baseline), ('v6_48', candidate), ('v6_48_seen_only', conservative)]:
                    assert np.isfinite(pred).all() and (pred >= 0).all()
                    part = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon',
                                  'rating', 'fold', 'history', 'seen48', 'protein_count']].copy()
                    part['split'], part['stream'], part['model'] = split, stream, name
                    part['prediction'] = pred
                    part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
                    output.append(part)
                print(f'{stream} {split} fold {fold}: V6={np.mean(output[-3].loss):.4f} 48={np.mean(output[-2].loss):.4f} seen={np.mean(output[-1].loss):.4f}', flush=True)
    result = pd.concat(output, ignore_index=True)
    keys = ['stream', 'split', 'model', 'patient_id', 'visit_month', 'target', 'horizon']
    assert not result.duplicated(keys).any()
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'oof_predictions.csv', index=False)
    result.groupby(['stream', 'split', 'model']).loss.agg(score='mean', requests='size').to_csv(OUT / 'comparison.csv')
    for dimension in ('target', 'horizon', 'history', 'protein_count', 'seen48'):
        result.groupby(['stream', 'split', 'model', dimension]).agg(
            score=('loss', 'mean'), requests=('loss', 'size'), patients=('patient_id', 'nunique')
        ).to_csv(OUT / f'by_{dimension}.csv')
    result['visit_period'] = np.where(result.visit_month.lt(48), 'before_48', 'at_or_after_48')
    result.groupby(['stream', 'split', 'model', 'visit_period', 'seen48']).agg(
        score=('loss', 'mean'), requests=('loss', 'size'), patients=('patient_id', 'nunique')
    ).to_csv(OUT / 'by_visit_period.csv')


if __name__ == '__main__':
    main()

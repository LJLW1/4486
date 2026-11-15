"""Patient-level comparison of visit-history alternatives to V6."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from horizon_kernel_experiment import predict
from protein_history_experiment import protein_values
from visit_history_model import KEYS, read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/top5_results'


def weighted_predict(train, valid, z, feature, scale):
    tr = z.reindex(pd.MultiIndex.from_frame(train[KEYS])).to_numpy()
    va = z.reindex(pd.MultiIndex.from_frame(valid[KEYS])).to_numpy()
    target = train.target.to_numpy()
    history = train.history.to_numpy()
    ratings = train.rating.to_numpy()
    months = train.target_month.to_numpy()
    horizons = train.horizon.to_numpy()
    ref_feature = train[feature].to_numpy()
    query_feature = valid[feature].to_numpy()
    output = np.empty(len(valid))
    groups = valid.reset_index(drop=True).groupby(['target', 'target_month', 'history', 'horizon'], sort=False).indices
    for (name, month, hist, horizon), positions in groups.items():
        target_mask = target == name
        mask = target_mask & (history == hist)
        if not mask.any():
            mask = target_mask
        truth = ratings[mask]
        candidates = np.arange(int(ratings[target_mask].max()) + 1)
        loss = 200 * abs(truth[:, None] - candidates) / (truth[:, None] + candidates + 2)
        weights = np.exp(-0.5 * ((months[mask] - month) / 12) ** 2)
        weights *= np.where((horizons[mask] == 6) == (horizon == 6), 1, 0.25)
        for start in range(0, len(positions), 64):
            ix = positions[start:start + 64]
            diff = va[ix, None, :] - tr[mask][None, :, :]
            sim = np.where(np.isfinite(diff), np.exp(-0.5 * diff ** 2), 1).mean(axis=2)
            gap = query_feature[ix, None] - ref_feature[mask][None, :]
            modifier = np.where(np.isfinite(gap), 0.5 + 0.5 * np.exp(-0.5 * (gap / scale) ** 2), 1)
            risk = ((0.5 + 0.5 * sim) * modifier * weights) @ loss
            output[ix] = candidates[risk.argmin(axis=1)]
    return output


def run():
    clinical, omics = read_data(DATA)
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv').set_index('patient_id').fold
    patients = np.sort(folds.index.unique())
    splits = {'original': folds}
    for seed in (42, 2024):
        assignment = pd.Series(index=patients, dtype=int)
        for fold, (_, valid) in enumerate(KFold(5, shuffle=True, random_state=seed).split(patients)):
            assignment.loc[patients[valid]] = fold
        splits[str(seed)] = assignment
    results = []
    for stream in ('union', 'omics'):
        rows = training_rows(clinical, omics, stream)
        visits = pd.concat([clinical[KEYS], omics]).drop_duplicates() if stream == 'union' else omics.drop_duplicates()
        visits = visits.sort_values(KEYS).copy()
        visits['gap'] = visits.groupby('patient_id').visit_month.diff()
        visits['count'] = visits.groupby('patient_id').cumcount() + 1
        rows = rows.merge(visits, on=KEYS, how='left', validate='many_to_one')
        # Ternary history digits are 0=not reached, 1=absent, 2=seen.
        first, second = rows.history % 3, rows.history // 3
        rows['merged_history'] = np.where((first == 2) | (second == 2), 9, rows.history)
        for split, assignment in splits.items():
            rows['fold'] = rows.patient_id.map(assignment)
            assert rows.fold.notna().all()
            for fold in range(5):
                train, valid = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
                train_patients = set(train.patient_id)
                assert train_patients.isdisjoint(valid.patient_id)
                z = protein_values(wide, train_patients, ['P05060', 'O15240'])
                baseline = predict(train, valid, z, 'horizon6')
                tr, va = train.copy(), valid.copy()
                tr['history'], va['history'] = tr.merged_history, va.merged_history
                candidate = predict(tr, va, z, 'horizon6')
                gap = weighted_predict(train, valid, z, 'gap', 12)
                count = weighted_predict(train, valid, z, 'count', 2)
                for model, values in [('v6', baseline), ('merged', candidate), ('gap', gap), ('count', count)]:
                    part = valid[['patient_id', 'visit_month', 'target', 'horizon', 'rating', 'history']].copy()
                    part['stream'], part['split'], part['fold'], part['model'] = stream, split, fold, model
                    part['prediction'] = values
                    part['loss'] = 200 * abs(part.rating - values) / (part.rating + values + 2)
                    assert np.isfinite(values).all() and (values >= 0).all()
                    results.append(part)
                print(stream, split, fold, *(round(x.loss.mean(), 4) for x in results[-4:]), flush=True)
    result = pd.concat(results, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'merged_oof.csv', index=False)
    result.groupby(['stream', 'split', 'model']).loss.agg(['mean', 'size']).to_csv(OUT / 'merged_summary.csv')
    result.groupby(['stream', 'split', 'model', 'history']).agg(score=('loss', 'mean'), patients=('patient_id', 'nunique'), requests=('loss', 'size')).to_csv(OUT / 'merged_history.csv')


if __name__ == '__main__':
    run()

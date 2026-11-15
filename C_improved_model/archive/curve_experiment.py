"""Grouped monotonic score curves evaluated on the V6 patient splits."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold
from visit_history_model import read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/top5_results'


def predict_curves(train, valid):
    result = np.empty(len(valid))
    grouped = valid.reset_index(drop=True).groupby(['target', 'history'], sort=False).indices
    for (target, history), indices in grouped.items():
        source = train[(train.target == target) & (train.history == history)]
        if source.empty:
            source = train[train.target == target]
        points, scores, weights = [], [], []
        max_rating = int(train.loc[train.target == target, 'rating'].max())
        candidates = np.arange(max_rating + 1)
        for month, part in source.groupby('target_month'):
            truth = part.rating.to_numpy()[:, None]
            losses = 200 * abs(truth - candidates) / (truth + candidates + 2)
            points.append(month)
            scores.append(candidates[losses.mean(axis=0).argmin()])
            weights.append(len(part))
        if len(points) == 1:
            result[indices] = scores[0]
        else:
            curve = IsotonicRegression(increasing=True, out_of_bounds='clip')
            curve.fit(points, scores, sample_weight=weights)
            result[indices] = curve.predict(valid.iloc[indices].target_month)
    return result


def main():
    clinical, omics = read_data(DATA)
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
        rows = training_rows(clinical, omics, stream)
        for split, assignment in splits.items():
            rows['fold'] = rows.patient_id.map(assignment)
            for fold in range(5):
                train, valid = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
                assert set(train.patient_id).isdisjoint(valid.patient_id)
                pred = predict_curves(train, valid)
                assert np.isfinite(pred).all() and (pred >= 0).all()
                part = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon', 'rating', 'history', 'fold']].copy()
                part['stream'], part['split'], part['prediction'] = stream, split, pred
                part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
                results.append(part)
                print(stream, split, fold, round(part.loss.mean(), 4), flush=True)
    result = pd.concat(results, ignore_index=True)
    assert not result.duplicated(['stream', 'split', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    result.to_csv(OUT / 'curve_oof.csv', index=False)
    result.groupby(['stream', 'split']).loss.agg(['mean', 'size']).to_csv(OUT / 'curve_summary.csv')
    result.groupby(['stream', 'split', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'curve_breakdown.csv')


if __name__ == '__main__':
    main()

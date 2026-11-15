"""Patient-fold comparison of protein-weighted visit-history predictions.

Run: OPENBLAS_NUM_THREADS=1 python3 C_improved_model/protein_history_experiment.py
Selection, scaling, and all labels are fitted using only training patients.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from visit_history_model import KEYS, TARGETS, VisitHistoryModel, read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/protein_history_results'


def ranking(clinical, wide):
    """Prior experiment's mean absolute month-adjusted Spearman, current scores."""
    current = clinical.merge(wide.reset_index(), on=KEYS, how='inner', validate='one_to_one')
    records = []
    for target in TARGETS:
        for protein in wide.columns:
            v = current[[protein, target, 'visit_month', 'patient_id']].dropna()
            if len(v) < 100 or v.patient_id.nunique() < 50 or v[protein].nunique() < 2 or v[target].nunique() < 2:
                continue
            x, y, month = (v[c].rank().to_numpy() for c in (protein, target, 'visit_month'))
            control = np.column_stack([np.ones(len(v)), month])
            x = x - control @ np.linalg.lstsq(control, x, rcond=None)[0]
            y = y - control @ np.linalg.lstsq(control, y, rcond=None)[0]
            corr = np.corrcoef(x, y)[0, 1]
            if np.isfinite(corr):
                records.append((protein, target, abs(corr), len(v), v.patient_id.nunique()))
    scores = pd.DataFrame(records, columns=['protein', 'target', 'abs_correlation', 'visits', 'patients'])
    order = scores.groupby('protein').agg(score=('abs_correlation', 'mean'), n_targets=('target', 'nunique'),
                                            min_visits=('visits', 'min'), min_patients=('patients', 'min'))
    return order.loc[order.n_targets.eq(4)].sort_values(['score', 'protein'], ascending=[False, True]).reset_index()


def protein_values(wide, train_patients, proteins):
    """Log NPX and scale from unique training visits, preserving missing values."""
    values = np.log1p(wide[proteins])
    fit = values.loc[values.index.get_level_values('patient_id').isin(train_patients)]
    mean, std = fit.mean(), fit.std(ddof=0).replace(0, 1)
    return ((values - mean) / std).clip(-5, 5)


def predict_with_proteins(train, valid, normalized):
    """Fixed conservative similarity: half baseline + half protein similarity.

    Each missing comparison is neutral (similarity 1); no future measurement used.
    """
    train_values = normalized.reindex(pd.MultiIndex.from_frame(train[KEYS])).to_numpy()
    valid_values = normalized.reindex(pd.MultiIndex.from_frame(valid[KEYS])).to_numpy()
    predictions = np.empty(len(valid), dtype=float)
    tr_target = train.target.to_numpy()
    tr_history = train.history.to_numpy()
    tr_month = train.target_month.to_numpy()
    tr_rating = train.rating.to_numpy()
    groups = valid.reset_index(drop=True).groupby(['target', 'target_month', 'history'], sort=False).indices
    for (target, month, history), positions in groups.items():
        target_mask = tr_target == target
        match = target_mask & (tr_history == history)
        mask = match if match.any() else target_mask
        truth = tr_rating[mask]
        candidates = np.arange(int(tr_rating[target_mask].max()) + 1)
        losses = 200 * abs(truth[:, None] - candidates) / (truth[:, None] + candidates + 2)
        weights = np.exp(-0.5 * ((tr_month[mask] - month) / 12) ** 2)
        ref = train_values[mask]
        # Bound memory for large groups while retaining the exact same calculation.
        for start in range(0, len(positions), 64):
            idx = positions[start:start + 64]
            query = valid_values[idx]
            d = query[:, None, :] - ref[None, :, :]
            similarity = np.exp(-0.5 * d * d)
            similarity = np.where(np.isfinite(d), similarity, 1.0).mean(axis=2)
            risk = ((0.5 + 0.5 * similarity) * weights) @ losses
            predictions[idx] = candidates[np.argmin(risk, axis=1)]
    return predictions


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    rows = training_rows(clinical, omics, 'union').merge(
        pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv'), on='patient_id', validate='many_to_one')
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = proteins.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    results, selected = [], []
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        patients = set(train.patient_id)
        assert patients.isdisjoint(set(valid.patient_id))
        train_clinical = clinical[clinical.patient_id.isin(patients)]
        train_wide = wide.loc[wide.index.get_level_values('patient_id').isin(patients)]
        order = ranking(train_clinical, train_wide)
        if len(order) < 20:
            raise ValueError('Fewer than 20 eligible proteins in a training fold')
        order.insert(0, 'fold', int(fold))
        selected.append(order)
        baseline = VisitHistoryModel(train).predict_rows(valid).to_numpy()
        for k in (0, 5, 10, 20):
            if k == 0:
                pred = baseline
            else:
                columns = order.head(k).protein.tolist()
                normalized = protein_values(wide, patients, columns)
                pred = predict_with_proteins(train, valid, normalized)
            part = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon', 'rating', 'fold']].copy()
            part['model'] = f'top{k}'
            part['prediction'] = pred
            part['loss'] = 200 * abs(part.rating - part.prediction) / (part.rating + part.prediction + 2)
            results.append(part)
            print(f'fold={fold} top{k}: {part.loss.mean():.4f}', flush=True)
    all_rows = pd.concat(results, ignore_index=True)
    all_rows.to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.concat(selected).to_csv(OUT / 'fold_protein_rankings.csv', index=False)
    folds = all_rows.groupby(['model', 'fold']).loss.mean().unstack()
    scores = all_rows.groupby('model').loss.agg(oof_score='mean', n='size')
    scores['fold_mean'] = folds.mean(axis=1)
    scores['fold_std'] = folds.std(axis=1, ddof=0)
    scores.join(folds.add_prefix('fold_')).to_csv(OUT / 'comparison.csv')
    all_rows.groupby(['model', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    print(scores.join(folds.add_prefix('fold_')).to_string(), flush=True)


if __name__ == '__main__':
    main()

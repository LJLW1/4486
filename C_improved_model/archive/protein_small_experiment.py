"""Small protein combinations on the visit-history model; exploratory CV only."""
from pathlib import Path
import numpy as np
import pandas as pd
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows
from protein_history_experiment import ranking, protein_values, predict_with_proteins

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/protein_small_results'
FIXED = ['Q06481', 'P17174', 'P04180']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    rows = training_rows(clinical, omics, 'union').merge(folds, on='patient_id', validate='many_to_one')
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = proteins.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    results, choices = [], []
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        patients = set(train.patient_id)
        assert patients.isdisjoint(set(valid.patient_id))
        train_wide = wide.loc[wide.index.get_level_values('patient_id').isin(patients)]
        order = ranking(clinical[clinical.patient_id.isin(patients)], train_wide)
        groups = {'baseline': [], 'top1': order.protein.head(1).tolist(),
                  'top2': order.protein.head(2).tolist(), 'top3': order.protein.head(3).tolist(),
                  **{f'fixed_{p}': [p] for p in FIXED}, 'fixed_3': FIXED}
        for name, columns in groups.items():
            choices.append({'fold': int(fold), 'model': name, 'proteins': ','.join(columns)})
            if columns:
                normalized = protein_values(wide, patients, columns)
                pred = predict_with_proteins(train, valid, normalized)
            else:
                pred = VisitHistoryModel(train).predict_rows(valid).to_numpy()
            frame = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon', 'rating', 'fold']].copy()
            frame['model'] = name
            frame['prediction'] = pred
            frame['loss'] = 200 * abs(frame.rating - pred) / (frame.rating + pred + 2)
            results.append(frame)
            print(f'fold={fold} {name}: {frame.loss.mean():.4f}', flush=True)
    full = pd.concat(results, ignore_index=True)
    assert not full.duplicated(['model', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    assert full.groupby('model').size().eq(len(rows)).all()
    assert np.isfinite(full.prediction).all() and full.prediction.ge(0).all()
    full.to_csv(OUT / 'oof_predictions.csv', index=False)
    pd.DataFrame(choices).to_csv(OUT / 'fold_proteins.csv', index=False)
    fold_scores = full.groupby(['model', 'fold']).loss.mean().unstack()
    scores = full.groupby('model').loss.agg(oof_score='mean', n='size')
    scores['fold_mean'] = fold_scores.mean(axis=1)
    scores['fold_std'] = fold_scores.std(axis=1, ddof=0)
    scores.join(fold_scores.add_prefix('fold_')).to_csv(OUT / 'comparison.csv')
    full.groupby(['model', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    print(scores.join(fold_scores.add_prefix('fold_')).to_string())


if __name__ == '__main__':
    main()

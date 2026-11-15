"""Exploratory all-subsets search among five previously identified proteins."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows
from protein_history_experiment import protein_values, predict_with_proteins

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/protein_combinations_results'
PROTEINS = ['Q06481', 'P17174', 'P04180', 'P05060', 'O15240']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    rows = training_rows(clinical, omics, 'union').merge(folds, on='patient_id', validate='many_to_one')
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = proteins.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    subsets = [list(c) for n in range(1, len(PROTEINS) + 1) for c in combinations(PROTEINS, n)]
    all_results = []
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        patients = set(train.patient_id)
        assert patients.isdisjoint(set(valid.patient_id))
        normalized = protein_values(wide, patients, PROTEINS)
        for columns in [[]] + subsets:
            name = '+'.join(columns) if columns else 'baseline'
            prediction = (predict_with_proteins(train, valid, normalized[columns]) if columns
                          else VisitHistoryModel(train).predict_rows(valid).to_numpy())
            frame = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon', 'rating', 'fold']].copy()
            frame['model'], frame['prediction'] = name, prediction
            frame['loss'] = 200 * abs(frame.rating - prediction) / (frame.rating + prediction + 2)
            all_results.append(frame)
        print(f'fold {fold} completed: {len(subsets)} combinations', flush=True)
    full = pd.concat(all_results, ignore_index=True)
    assert not full.duplicated(['model', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    assert full.groupby('model').size().eq(len(rows)).all()
    assert np.isfinite(full.prediction).all() and full.prediction.ge(0).all()
    full.to_csv(OUT / 'oof_predictions.csv', index=False)
    fold = full.groupby(['model', 'fold']).loss.mean().unstack()
    scores = full.groupby('model').loss.agg(oof_score='mean', n='size')
    scores['fold_mean'] = fold.mean(axis=1)
    scores['fold_std'] = fold.std(axis=1, ddof=0)
    scores['n_proteins'] = [0 if name == 'baseline' else name.count('+') + 1 for name in scores.index]
    scores['folds_improved'] = fold.lt(fold.loc['baseline'], axis=1).sum(axis=1)
    scores['oof_delta'] = scores.oof_score - scores.loc['baseline', 'oof_score']
    scores.join(fold.add_prefix('fold_')).sort_values('oof_score').to_csv(OUT / 'comparison.csv')
    full.groupby(['model', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    print(scores.sort_values('oof_score').head(12).to_string(), flush=True)


if __name__ == '__main__':
    main()

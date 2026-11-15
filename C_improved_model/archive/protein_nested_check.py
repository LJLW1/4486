"""Three inner patient folds select among fixed candidate subsets per outer fold."""
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows
from protein_history_experiment import protein_values, predict_with_proteins
from protein_combinations_experiment import PROTEINS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/protein_combinations_results'


def main():
    clinical, omics = read_data(DATA)
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    rows = training_rows(clinical, omics, 'union').merge(folds, on='patient_id', validate='many_to_one')
    proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = proteins.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    subsets = [[]] + [list(c) for n in range(1, len(PROTEINS)+1) for c in combinations(PROTEINS, n)]
    results, details = [], []
    for outer in sorted(rows.fold.unique()):
        outer_train = rows[rows.fold.ne(outer)]
        outer_valid = rows[rows.fold.eq(outer)]
        patients = np.sort(outer_train.patient_id.unique())
        losses = {tuple(c): [0.0, 0] for c in subsets}
        for inner, (train_ix, valid_ix) in enumerate(KFold(n_splits=3, shuffle=True, random_state=4486).split(patients)):
            train_ids, valid_ids = patients[train_ix], patients[valid_ix]
            train = outer_train[outer_train.patient_id.isin(train_ids)]
            valid = outer_train[outer_train.patient_id.isin(valid_ids)]
            assert set(train.patient_id).isdisjoint(valid.patient_id) and set(train.patient_id).isdisjoint(outer_valid.patient_id)
            norm = protein_values(wide, set(train_ids), PROTEINS)
            for cols in subsets:
                pred = predict_with_proteins(train, valid, norm[cols]) if cols else VisitHistoryModel(train).predict_rows(valid).to_numpy()
                loss = 200 * abs(valid.rating.to_numpy() - pred) / (valid.rating.to_numpy() + pred + 2)
                losses[tuple(cols)][0] += loss.sum()
                losses[tuple(cols)][1] += len(loss)
            print(f'outer {outer} inner {inner} completed', flush=True)
        best = min(subsets, key=lambda cols: (losses[tuple(cols)][0] / losses[tuple(cols)][1], len(cols), cols))
        train_ids = set(outer_train.patient_id)
        norm = protein_values(wide, train_ids, PROTEINS)
        pred = predict_with_proteins(outer_train, outer_valid, norm[best]) if best else VisitHistoryModel(outer_train).predict_rows(outer_valid).to_numpy()
        base = VisitHistoryModel(outer_train).predict_rows(outer_valid).to_numpy()
        frame = outer_valid[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
        frame['prediction'], frame['baseline'] = pred, base
        frame['loss'] = 200 * abs(frame.rating - pred) / (frame.rating + pred + 2)
        frame['baseline_loss'] = 200 * abs(frame.rating - base) / (frame.rating + base + 2)
        results.append(frame)
        details.append({'fold': int(outer), 'selected': '+'.join(best) or 'baseline',
                        'inner_score': losses[tuple(best)][0] / losses[tuple(best)][1],
                        'outer_score': frame.loss.mean(), 'baseline_score': frame.baseline_loss.mean()})
        print('selected',details[-1],flush=True)
    pd.DataFrame(details).to_csv(OUT / 'nested_selection.csv', index=False)
    full = pd.concat(results, ignore_index=True)
    assert len(full) == len(rows) and not full.duplicated(['patient_id','visit_month','target','horizon']).any()
    full.to_csv(OUT / 'nested_oof_predictions.csv', index=False)
    print('nested OOF:',full.loss.mean(), 'baseline OOF:', full.baseline_loss.mean(),flush=True)


if __name__ == '__main__':
    main()

"""Patient-fold ablation of V12 and the V13 protein similarity branch."""
from pathlib import Path
import numpy as np
import pandas as pd
from horizon_kernel_experiment import predict
from protein_history_experiment import protein_values
from visit_history_model import KEYS, read_data, training_rows

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/protein_ablation_results'
PROTEINS = ['P05060', 'O15240']
REQUEST = ['patient_id', 'visit_month', 'target', 'horizon']


def main():
    clinical, omics = read_data(DATA)
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    rows = training_rows(clinical, omics, 'union').merge(folds, on='patient_id', validate='many_to_one')
    saved = pd.read_csv(ROOT / 'C_improved_model/teammate_aligned_results/oof_predictions.csv')
    saved = saved[REQUEST + ['prediction', 'fold', 'rating']].rename(columns={
        'prediction': 'base_prediction', 'fold': 'saved_fold', 'rating': 'saved_rating'})
    rows = rows.merge(saved, on=REQUEST, validate='one_to_one')
    assert rows.fold.eq(rows.saved_fold).all() and rows.rating.eq(rows.saved_rating).all()
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    outputs = []
    for fold in range(5):
        train, valid = rows[rows.fold.ne(fold)], rows[rows.fold.eq(fold)]
        patients = set(train.patient_id)
        assert patients.isdisjoint(valid.patient_id)
        z = protein_values(wide, patients, PROTEINS)
        null_z = z * np.nan
        with_protein = predict(train, valid, z, 'horizon6')
        without_protein = predict(train, valid, null_z, 'horizon6')
        available = np.isfinite(z.reindex(pd.MultiIndex.from_frame(valid[KEYS])).to_numpy()).any(axis=1)
        base = valid.base_prediction.to_numpy()
        for branch, aux in [('history', without_protein), ('protein', with_protein)]:
            for weight in (0.10, 0.25, 0.50):
                # Match V13: only apply the auxiliary model where selected proteins exist.
                prediction = np.where(available, (1 - weight) * base + weight * aux, base)
                part = valid[REQUEST + ['rating', 'fold']].copy()
                part['available'], part['model'], part['prediction'] = available, f'{branch}_{weight:.2f}', prediction
                part['loss'] = 200 * abs(part.rating - prediction) / (part.rating + prediction + 2)
                outputs.append(part)
        part = valid[REQUEST + ['rating', 'fold']].copy()
        part['available'], part['model'], part['prediction'] = available, 'v12', base
        part['loss'] = 200 * abs(part.rating - base) / (part.rating + base + 2)
        outputs.append(part)
        print('fold', fold, 'patients', len(set(valid.patient_id)), 'requests', len(valid), flush=True)
    result = pd.concat(outputs, ignore_index=True)
    assert not result.duplicated(REQUEST + ['model']).any()
    assert result.groupby('model').size().eq(len(rows)).all()
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / 'oof_predictions.csv', index=False)
    fold_scores = result.groupby(['model', 'fold']).loss.mean().unstack()
    summary = result.groupby('model').loss.agg(oof='mean', requests='size')
    summary['fold_mean'] = fold_scores.mean(axis=1)
    summary.join(fold_scores.add_prefix('fold_')).to_csv(OUT / 'summary.csv')
    result.groupby(['model', 'target', 'horizon', 'available']).agg(
        score=('loss', 'mean'), requests=('loss', 'size'), patients=('patient_id', 'nunique')
    ).to_csv(OUT / 'breakdown.csv')
    print(summary.sort_values('oof').to_string(), flush=True)


if __name__ == '__main__':
    main()

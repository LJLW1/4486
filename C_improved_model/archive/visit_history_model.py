"""Time baseline and causal visit-history model for the AMP competition.

Notebook: model = VisitHistoryModel.from_data_dir('/kaggle/input/amp-pd')
          predictions = model.predict_batch(test, test_peptides, test_proteins, sample)
Call predict_batch once per API batch, in the order supplied by iter_test().
Local validation: python C_improved_model/visit_history_model.py --validate
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

TARGETS = [f'updrs_{i}' for i in range(1, 5)]
KEYS = ['patient_id', 'visit_month']
MILESTONES = (6, 18)


def read_data(data_dir):
    data_dir = Path(data_dir)
    clinical = pd.read_csv(data_dir / 'train_clinical_data.csv')
    omics = pd.concat([pd.read_csv(data_dir / name, usecols=KEYS) for name in
                       ('train_proteins.csv', 'train_peptides.csv')]).drop_duplicates()
    return clinical, omics


def history_codes(requests, visits):
    """Milestone visits are visible only at or after their month."""
    result = np.zeros(len(requests), dtype=int)
    for i, milestone in enumerate(MILESTONES):
        seen = set(visits.loc[visits.visit_month.eq(milestone), 'patient_id'])
        digit = np.where(requests.visit_month.to_numpy() < milestone, -1,
                         requests.patient_id.isin(seen).to_numpy().astype(int))
        result += (digit + 1) * 3 ** i
    return result


def training_rows(clinical, omics, stream='union'):
    origins = omics.drop_duplicates().copy()
    requests = origins.merge(pd.DataFrame({'horizon': [0, 6, 12, 24]}), how='cross')
    requests['target_month'] = requests.visit_month + requests.horizon
    labels = clinical.melt(id_vars=KEYS, value_vars=TARGETS, var_name='target',
                           value_name='rating').dropna(subset=['rating'])
    labels = labels.rename(columns={'visit_month': 'target_month'})
    rows = requests.merge(labels, on=['patient_id', 'target_month'], validate='many_to_many')
    if rows.duplicated(['patient_id', 'visit_month', 'target', 'horizon']).any():
        raise ValueError('Duplicate training requests; check clinical visit keys')
    visits = pd.concat([clinical[KEYS], omics]).drop_duplicates() if stream == 'union' else omics
    rows['history'] = history_codes(rows, visits)
    return rows


class VisitHistoryModel:
    def __init__(self, rows, use_history=True):
        self.rows = rows.copy()
        self.use_history = use_history
        self.visits = set()
        self.last_month = None

    @classmethod
    def from_data_dir(cls, data_dir, use_history=True, stream='union'):
        clinical, omics = read_data(data_dir)
        return cls(training_rows(clinical, omics, stream), use_history)

    def predict_rows(self, rows):
        result = pd.Series(index=rows.index, dtype=float)
        groups = ['target', 'target_month'] + (['history'] if self.use_history else [])
        for key, part in rows.groupby(groups, sort=False):
            target, month = key[:2]
            selected = self.rows.loc[self.rows.target.eq(target)]
            if self.use_history:
                same = selected.loc[selected.history.eq(key[2])]
                if not same.empty:
                    selected = same
            truth = selected.rating.to_numpy()[:, None]
            candidates = np.arange(int(truth.max()) + 1)
            losses = 200 * np.abs(truth - candidates) / (truth + candidates + 2)
            weights = np.exp(-0.5 * ((selected.target_month.to_numpy() - month) / 12) ** 2)
            result.loc[part.index] = candidates[(weights @ losses).argmin()]
        return result

    def predict_batch(self, test, peptides, proteins, sample):
        match = sample.prediction_id.str.extract(
            r'^(?P<patient_id>\d+)_(?P<visit_month>\d+)_(?P<target>updrs_[1-4])_plus_(?P<horizon>0|6|12|24)_months$')
        if match.isna().any().any():
            raise ValueError('Unexpected prediction_id format')
        for col in ('patient_id', 'visit_month', 'horizon'):
            match[col] = match[col].astype(int)
        match['target_month'] = match.visit_month + match.horizon
        current = pd.concat([frame[KEYS] for frame in (test, peptides, proteins)], ignore_index=True).drop_duplicates()
        months = pd.concat([current.visit_month, match.visit_month]).dropna().unique()
        if len(months) != 1:
            raise ValueError('Expected one visit month per API batch')
        month = int(months[0])
        if self.last_month is not None and month < self.last_month:
            raise ValueError('API batches must be in chronological order')
        self.last_month = month
        self.visits.update(map(tuple, current[KEYS].itertuples(index=False, name=None)))
        visits = pd.DataFrame(list(self.visits), columns=KEYS)
        match['history'] = history_codes(match, visits)
        output = sample.copy()
        output['rating'] = self.predict_rows(match).to_numpy()
        return output


def validate(data_dir, folds_path, output_dir):
    clinical, omics = read_data(data_dir)
    folds = pd.read_csv(folds_path)
    if folds.patient_id.duplicated().any():
        raise ValueError('Patient fold assignments must be unique')
    union = training_rows(clinical, omics, 'union').merge(folds, on='patient_id', validate='many_to_one')
    conservative = training_rows(clinical, omics, 'omics')
    assert len(union) == len(conservative)
    conservative['fold'] = union.fold.to_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    all_predictions = []
    for name, rows, history in [('baseline', union, False), ('history_union', union, True),
                                ('history_omics', conservative, True)]:
        pred = pd.Series(index=rows.index, dtype=float)
        for fold in sorted(rows.fold.unique()):
            train = rows.loc[rows.fold.ne(fold)]
            valid = rows.loc[rows.fold.eq(fold)]
            assert set(train.patient_id).isdisjoint(valid.patient_id)
            pred.loc[valid.index] = VisitHistoryModel(train, history).predict_rows(valid)
        details = rows[['patient_id', 'visit_month', 'horizon', 'target', 'target_month', 'rating', 'fold']].copy()
        details['prediction'] = pred
        details['loss'] = 200 * abs(details.rating - details.prediction) / (details.rating + details.prediction + 2)
        details['model'] = name
        scores = details.groupby('fold').loss.mean()
        results.append({'model': name, 'oof_score': details.loss.mean(), 'fold_mean': scores.mean(),
                        'fold_std': scores.std(ddof=0), 'fold_scores': scores.to_dict(), 'n': len(details)})
        all_predictions.append(details)
        print(json.dumps(results[-1]), flush=True)
    full = pd.concat(all_predictions, ignore_index=True)
    full.to_csv(output_dir / 'oof_predictions.csv', index=False)
    full.groupby(['model', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(output_dir / 'breakdown.csv')
    (output_dir / 'results.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--data-dir', default=root / 'amp-parkinsons-disease-progression-prediction')
    parser.add_argument('--folds', default=root / 'artifacts/shared/patient_folds.csv')
    parser.add_argument('--output-dir', default=root / 'C_improved_model/validation_results')
    args = parser.parse_args()
    if args.validate:
        validate(args.data_dir, args.folds, args.output_dir)
    else:
        parser.print_help()

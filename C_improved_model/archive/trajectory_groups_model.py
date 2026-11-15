"""Visit-group model inspired by AmbrosM's fifth-place approach."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TARGETS = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
KEYS = ['patient_id', 'visit_month']


def assign_groups(rows, visits):
    visits = visits.drop_duplicates(KEYS).sort_values(KEYS)
    visits = visits.copy()
    visits['nonannual'] = visits.visit_month.mod(12).ne(0)
    visits['ever_nonannual'] = visits.groupby('patient_id').nonannual.cummax()
    all_rows = rows[KEYS].reset_index(drop=True).copy()
    all_rows['_order'] = np.arange(len(all_rows))
    merged = pd.merge_asof(all_rows.sort_values('visit_month'),
                           visits.sort_values('visit_month'), on='visit_month',
                           by='patient_id', direction='backward')
    seen = merged.ever_nonannual.fillna(False).to_numpy(dtype=bool)
    group = np.where(seen, 'ill', np.where(merged.visit_month.ge(12), 'healthy', 'unknown'))
    ordered = pd.Series(group, index=merged['_order']).sort_index()
    return ordered.to_numpy()


def training_rows(clinical, proteins, visits):
    origins = clinical[KEYS].drop_duplicates()
    first = proteins.groupby('patient_id').visit_month.min().rename('first')
    origins = origins.merge(first, on='patient_id')
    origins = origins[(origins.visit_month >= origins['first']) & (origins.visit_month <= 84)]
    origins = origins.merge(pd.DataFrame({'horizon': [0, 6, 12, 24]}), how='cross')
    origins['target_month'] = origins.visit_month + origins.horizon
    origins['group'] = assign_groups(origins, visits)
    labels = clinical.melt(id_vars=KEYS, value_vars=TARGETS, var_name='target', value_name='rating').dropna(subset=['rating'])
    labels = labels.rename(columns={'visit_month': 'target_month'})
    return origins.merge(labels, on=['patient_id', 'target_month'], validate='many_to_many')


def loss(y, p):
    return np.mean(200 * abs(y - p) / (y + p + 2))


def smape_constant(y):
    candidates = np.arange(int(np.max(y)) + 1)
    return candidates[(200 * abs(y[:, None] - candidates) / (y[:, None] + candidates + 2)).mean(axis=0).argmin()]


def fit_curves(rows):
    result = {}
    bounds = {'updrs_1': [(3, 7), (0, .1)], 'updrs_2': [(3, 7), (0, .1)],
              'updrs_3': [(16, 22), (0, .3)], 'updrs_4': [(-7, -5), (0, .2)]}
    for target in TARGETS:
        target_rows = rows[rows.target.eq(target)]
        for group in ('unknown', 'healthy', 'ill'):
            subset = target_rows[target_rows.group.eq(group)]
            if subset.empty:
                subset = target_rows
            y = subset.rating.to_numpy()
            months = subset.target_month.to_numpy()
            if group == 'unknown':
                result[(target, group)] = ('constant', smape_constant(y))
            elif group == 'ill':
                clipped = np.maximum(months, 60) if target == 'updrs_4' else months
                def objective(x):
                    return loss(y, np.maximum(0, np.round(x[0] + clipped * x[1])))
                b = bounds[target]
                opt = minimize(objective, [b[0][0] + 1, b[1][0] + .05], bounds=b, method='Powell')
                result[(target, group)] = ('linear', opt.x)
            else:
                unique, inverse = np.unique(months, return_inverse=True)
                prediction = np.zeros(len(unique))
                while True:
                    current, best = loss(y, prediction[inverse]), None
                    for i in range(len(unique) - 1, -1, -1):
                        candidate = prediction.copy()
                        candidate[i:] += 1
                        score = loss(y, candidate[inverse])
                        if score < current:
                            current, best = score, i
                    if best is None:
                        break
                    prediction[best:] += 1
                result[(target, group)] = ('isotonic', (unique, prediction))
    return result


def predict_rows(rows, curves):
    output = np.zeros(len(rows))
    for (target, group), positions in rows.reset_index(drop=True).groupby(['target', 'group']).indices.items():
        mode, params = curves[(target, group)]
        month = rows.iloc[positions].target_month.to_numpy()
        if mode == 'constant':
            values = np.full(len(month), params)
        elif mode == 'linear':
            if target == 'updrs_4':
                month = np.maximum(month, 60)
            values = np.maximum(0, np.round(params[0] + month * params[1]))
        else:
            indices = np.searchsorted(params[0], month, side='right') - 1
            values = params[1][np.clip(indices, 0, len(params[1]) - 1)]
        output[positions] = values
    return output


class TrajectoryModel:
    def __init__(self, clinical, proteins):
        self.curves = fit_curves(training_rows(clinical, proteins, clinical[KEYS]))
        self.visits = pd.DataFrame(columns=KEYS)

    def predict_batch(self, test, peptides, proteins, sample):
        self.visits = pd.concat([self.visits, test[KEYS], peptides[KEYS], proteins[KEYS]]).drop_duplicates()
        fields = sample.prediction_id.str.extract(r'^(\d+)_(\d+)_(updrs_[1-4])_plus_(0|6|12|24)_months$')
        if fields.isna().any().any():
            raise ValueError('Unexpected prediction ID')
        fields.columns = ['patient_id', 'visit_month', 'target', 'horizon']
        for column in ('patient_id', 'visit_month', 'horizon'):
            fields[column] = fields[column].astype(int)
        fields['target_month'] = fields.visit_month + fields.horizon
        fields['group'] = assign_groups(fields, self.visits)
        output = sample.copy()
        output['rating'] = predict_rows(fields, self.curves)
        return output

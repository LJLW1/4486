"""Patient-fold check of the winner network and a three-seed ensemble."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/winner_bagged_results'


def scheduler(optimizer, num_warmup_steps, num_training_steps):
    def factor(step):
        if step < num_warmup_steps:
            return float(step) / max(1, num_warmup_steps)
        progress = float(step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, factor)


def main():
    torch.set_num_threads(1)
    notebook = json.loads((ROOT / 'C_improved_model/winner_reproduction_submission.ipynb').read_text())
    cells = [''.join(cell['source']) for cell in notebook['cells']]
    namespace = {}
    cells[0] = cells[0].replace('/kaggle/input/amp-pd', str(DATA))
    exec(compile(cells[0], 'winner_load', 'exec'), namespace)
    exec(compile(cells[1], 'winner_preprocess', 'exec'), namespace)
    exec(compile(cells[2], 'winner_helpers', 'exec'), namespace)
    cells[3] = cells[3].replace('from transformers import get_cosine_schedule_with_warmup',
                                'get_cosine_schedule_with_warmup = local_scheduler')
    namespace['local_scheduler'] = scheduler
    exec(compile(cells[3], 'winner_network', 'exec'), namespace)
    exec(compile(cells[5].split('\nmodel_nn = NNRegModel1(cfg)')[0], 'winner_config', 'exec'), namespace)
    original = namespace['sample3']
    supplemental = namespace['sup_sample3']
    cfg = namespace['cfg']
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    requests = pd.read_csv(ROOT / 'C_improved_model/horizon_kernel_results/oof_predictions.csv')
    requests = requests[requests.model.eq('horizon6')].copy()
    winners = original.rename(columns={'target_i': 'target_number', 'target': 'actual'})
    winners = winners.copy()
    winners['target'] = 'updrs_' + winners.target_number.astype(str)
    keys = ['patient_id', 'visit_month', 'horizon', 'target']
    valid = requests.merge(winners, on=keys, how='left', validate='one_to_one', indicator=True,
                           suffixes=('_request', ''))
    print('Matched request features:', (valid._merge == 'both').sum(), '/', len(valid), flush=True)
    valid = valid[valid._merge.eq('both')].copy()
    assert np.allclose(valid.rating, valid.actual)
    results = []
    for fold in range(5):
        held = set(folds.loc[folds.fold.eq(fold), 'patient_id'])
        train = pd.concat([original[~original.patient_id.isin(held)],
                           supplemental[~supplemental.patient_id.isin(held)]], ignore_index=True)
        query = valid[valid.fold.eq(fold)].copy()
        assert set(train.patient_id).isdisjoint(query.patient_id)
        original_features = list(cfg.features)
        cfg.seed = 101
        network = namespace['run_train'](cfg, train, None, verbose=False)
        single = namespace['run_test'](network, cfg, query).reshape(-1)
        cfg.features = original_features + ['blood_taken', 'count_non12_visits']
        forecasts = []
        for seed in (101, 211, 307):
            cfg.seed = seed
            network = namespace['run_train'](cfg, train, None, verbose=False)
            forecasts.append(namespace['run_test'](network, cfg, query).reshape(-1))
            print('fold', fold, 'seed', seed, 'complete', flush=True)
        cfg.features = original_features
        for name, forecast in [('single', single), ('enriched_bag3', np.mean(forecasts, axis=0))]:
            pred = np.round(np.clip(forecast, 0, None))
            part = query[keys + ['rating', 'fold']].copy()
            part['model'], part['prediction'] = name, pred
            part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
            results.append(part)
            print(fold, name, len(part), part.loss.mean(), flush=True)
    output = pd.concat(results, ignore_index=True)
    OUT.mkdir(exist_ok=True)
    output.to_csv(OUT / 'oof.csv', index=False)
    output.groupby('model').loss.agg(['mean', 'size']).to_csv(OUT / 'summary.csv')


if __name__ == '__main__':
    main()

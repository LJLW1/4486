"""Replay public batches with random network weights and a stub LightGBM."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction' / 'example_test_files'
notebook = json.loads((ROOT / 'C_improved_model/winner_neural_ensemble_submission.ipynb').read_text())
code = [''.join(cell.get('source', [])) for cell in notebook['cells']]
namespace = {'np': np, 'pd': pd}
exec(compile(code[1].split('\ndp3 = DataPrep()')[0], 'preprocessing', 'exec'), namespace)
namespace['dp3'] = namespace['DataPrep']()
exec(compile(code[2], 'helpers', 'exec'), namespace)
network_code = code[3].replace('from transformers import get_cosine_schedule_with_warmup',
                               'get_cosine_schedule_with_warmup = None')
exec(compile(network_code, 'network', 'exec'), namespace)
frames = {name: pd.read_csv(DATA / (name + '.csv')) for name in
          ('test', 'test_peptides', 'test_proteins', 'sample_submission')}
namespace['sample3'] = namespace['dp3'].transform_test(
    frames['test_proteins'], frames['test_peptides'], frames['test'], frames['sample_submission'])
exec(compile(code[5].split('\nmodel_nn = NNRegModel1(cfg)')[0], 'config', 'exec'), namespace)
cfg = namespace['cfg']
model = namespace['NNRegModel1'](cfg)
model.models = [namespace['Net'](cfg) for _ in range(3)]
namespace['model_nn'] = model
namespace['model_lgb'] = types.SimpleNamespace(predict=lambda frame: np.zeros(len(frame)))


class SampleEnvironment:
    def __init__(self):
        self.expected = []
        self.count = 0

    def iter_test(self):
        sample = frames['sample_submission']
        months = sample.prediction_id.str.split('_').str[1].astype(int)
        for month in sorted(months.unique()):
            batch = sample.loc[months.eq(month)].copy()
            self.expected = batch.prediction_id.tolist()
            yield tuple(frames[name].loc[frames[name].visit_month.eq(month)].copy()
                        for name in ('test', 'test_peptides', 'test_proteins')) + (batch,)

    def predict(self, batch):
        assert batch.prediction_id.tolist() == self.expected
        assert np.isfinite(batch.rating).all() and batch.rating.ge(0).all()
        self.count += len(batch)


environment = SampleEnvironment()
sys.modules['amp_pd_peptide'] = types.SimpleNamespace(make_env=lambda: environment)
torch.set_num_threads(1)
exec(compile(code[6], 'api_loop', 'exec'), namespace)
assert environment.count == len(frames['sample_submission'])
print('Public sample API loop passed with random NN weights and stub LightGBM:', environment.count)

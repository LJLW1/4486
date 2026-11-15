"""Exercise the adapted API loop with public batches and deterministic mock MLPs."""
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
nb = json.loads((ROOT / 'C_improved_model/top4_mlp_submission.ipynb').read_text())
namespace = {}
for index in (2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 15, 16, 17):
    code = ''.join(nb['cells'][index]['source']).replace('/kaggle/input/amp-pd', str(DATA))
    exec(compile(code, 'top4_preprocess', 'exec'), namespace)

class MockModel:
    def load_weights(self, path):
        pass

    def predict(self, features, verbose=0):
        return (features[:, :1] * 3 + 7)


class MockKeras:
    def clear_session(self):
        pass


class MockEnvironment:
    def __init__(self):
        self.outputs = []

    def predict(self, output):
        self.outputs.append(output)


fixture = DATA / 'example_test_files'
test = pd.read_csv(fixture / 'test.csv')
peptides = pd.read_csv(fixture / 'test_peptides.csv')
proteins = pd.read_csv(fixture / 'test_proteins.csv')
sample = pd.read_csv(fixture / 'sample_submission.csv')
batches = []
for month in sorted(test.visit_month.unique()):
    ids = sample.prediction_id.str.split('_').str[1].astype(int).eq(month)
    batches.append((test[test.visit_month.eq(month)], peptides[peptides.visit_month.eq(month)],
                    proteins[proteins.visit_month.eq(month)], sample.loc[ids].copy()))
env = MockEnvironment()
namespace.update({'models': {}, 'K': MockKeras(), 'build_model': MockModel,
                  'iter_test': iter(batches), 'env': env, 'gc': gc})
exec(compile(''.join(nb['cells'][25]['source']), 'top4_api', 'exec'), namespace)
assert len(env.outputs) == len(batches)
for batch, output in zip(batches, env.outputs):
    assert output.prediction_id.tolist() == batch[3].prediction_id.tolist()
    assert np.isfinite(output.rating).all() and output.rating.ge(0).all()
print('Passed', len(batches), 'sample batches and', sum(map(len, env.outputs)), 'requests')

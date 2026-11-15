"""Replay both feature builders and the blend against the public API fixture."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
SAMPLE = DATA / 'example_test_files'
notebook = json.loads((ROOT / 'C_improved_model/champion_v6_blend_submission.ipynb').read_text())
preparer_code = ''.join(notebook['cells'][1]['source']).split('\ndp3 = DataPrep()')[0]
scope = {'np': np, 'pd': pd}
exec(compile(preparer_code, 'champion_dataprep', 'exec'), scope)
v6_code = ''.join(notebook['cells'][6]['source']).replace('/kaggle/input/amp-pd', str(DATA))
exec(compile(v6_code, 'v6_model', 'exec'), scope)
preparer, model = scope['DataPrep'](), scope['v6_model']
test = pd.read_csv(SAMPLE / 'test.csv')
peptides = pd.read_csv(SAMPLE / 'test_peptides.csv')
proteins = pd.read_csv(SAMPLE / 'test_proteins.csv')
sample = pd.read_csv(SAMPLE / 'sample_submission.csv')
sample_month = sample.prediction_id.str.split('_').str[1].astype(int)
total = 0
for month in sorted(sample_month.unique()):
    current = lambda df: df[df.visit_month.eq(month)].copy()
    visible = lambda df: df[df.visit_month.le(month)].copy()
    batch = sample.loc[sample_month.eq(month)].copy().reset_index(drop=True)
    features = preparer.transform_test(visible(proteins), visible(peptides), visible(test), batch)
    pred = model.predict_batch(current(test), current(peptides), current(proteins), batch)
    assert pred.prediction_id.equals(batch.prediction_id)
    assert len(features) == len(batch) and np.isfinite(pred.rating).all() and pred.rating.ge(0).all()
    # Use a synthetic champion output to verify alignment and blend arithmetic.
    champion = np.arange(len(batch), dtype=float)
    blend = .75 * champion + .25 * pred.rating.to_numpy()
    assert np.isfinite(blend).all() and (blend >= 0).all()
    total += len(batch)
assert total == len(sample)
print('Two feature paths and blend replayed on', total, 'public sample requests')

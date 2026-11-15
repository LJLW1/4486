"""Check first-place feature preprocessing on the supplied public API sample."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction' / 'example_test_files'
notebook = json.loads((ROOT / 'C_improved_model/winner_reproduction_submission.ipynb').read_text())
code = ''.join(notebook['cells'][1]['source']).split('\ndp3 = DataPrep()')[0]
namespace = {'np': np, 'pd': pd}
exec(compile(code, 'winner_dataprep', 'exec'), namespace)
preparer = namespace['DataPrep']()
test = pd.read_csv(DATA / 'test.csv')
peptides = pd.read_csv(DATA / 'test_peptides.csv')
proteins = pd.read_csv(DATA / 'test_proteins.csv')
sample = pd.read_csv(DATA / 'sample_submission.csv')
features = preparer.transform_test(proteins, peptides, test, sample)
assert len(features) == len(sample)
assert features.patient_id.notna().all()
assert features.filter(regex='^(visit_|btest_)').notna().all().all()
assert np.isfinite(features.select_dtypes(include='number').to_numpy()).all()
print('Feature preprocessing passed for', len(features), 'public sample requests.')

"""Prepare Deotte's fourth-place MLP with the established official API."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
original = Path('/private/tmp/amp_top4/4th-place-solution-public-54-9-private-60-1.ipynb')
notebook = json.loads(original.read_text())
api = '''import sys
sys.path.append('/kaggle/input/amp-pd')
import amp_pd_peptide
env = amp_pd_peptide.make_env()
iter_test = env.iter_test()
'''
loop = '''p_ct = {}
for test, test_peptides, test_proteins, sample_submission in iter_test:
    del models
    gc.collect()
    K.clear_session()
    models = {}
    for target in TARS:
        models[target] = []
        for fold in range(5):
            clf = build_model()
            clf.load_weights('MLP_{}_f{}.h5'.format(target, fold))
            models[target].append(clf)

    month = int(test.visit_month.iloc[0])
    for patient in test.patient_id.unique():
        if patient not in p_ct:
            p_ct[patient] = {u: (0 if u <= month else -1) for u in USE}
        p_ct[patient][month] = 1
    for patient in p_ct:
        if p_ct[patient].get(month) == -1:
            p_ct[patient][month] = 0

    parsed = sample_submission.prediction_id.str.extract(
        r'^(?P<patient_id>\\d+)_(?P<visit_month>\\d+)_(?P<target>updrs_[1-4])_plus_(?P<horizon>0|6|12|24)_months$')
    if parsed.isna().any().any():
        raise ValueError('Unexpected API prediction identifier')
    for col in ['patient_id', 'visit_month', 'horizon']:
        parsed[col] = parsed[col].astype(int)
    if not parsed.visit_month.eq(month).all():
        raise ValueError('Prediction month differs from API batch')
    x = np.zeros((len(parsed), len(USE) + 1))
    x[:, 0] = (parsed.visit_month.to_numpy() + parsed.horizon.to_numpy() - mms[0]) / sss[0]
    for i, milestone in enumerate(USE):
        values = np.array([p_ct[p][milestone] for p in parsed.patient_id])
        x[:, i + 1] = (values - mms[i + 1]) / sss[i + 1] if sss[i + 1] else values - mms[i + 1]
    predictions = np.zeros(len(parsed))
    for target in TARS:
        mask = parsed.target.eq(target).to_numpy()
        if mask.any():
            scores = np.mean([clf.predict(x[mask], verbose=0).reshape(-1) for clf in models[target]], axis=0)
            predictions[mask] = np.maximum(np.floor(scores + 0.5), 0)
    output = sample_submission.copy()
    output['rating'] = predictions
    env.predict(output)
'''
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source']).replace('/kaggle/input/amp-parkinsons-disease-progression-prediction', '/kaggle/input/amp-pd')
    if i == 23:
        code = api
    elif i == 25:
        code = loop
    ast.parse(code)
    cell['source'] = code.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None
destination = ROOT / 'C_improved_model/top4_mlp_submission.ipynb'
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Saved and syntax-checked', destination)

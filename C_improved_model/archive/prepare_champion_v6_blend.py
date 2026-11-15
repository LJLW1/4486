"""Blend the verified champion notebook and V6 in the official API loop."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
winner = json.loads((ROOT / 'C_improved_model/winner_reproduction_submission.ipynb').read_text())
v6 = json.loads((ROOT / 'C_improved_model/horizon_history_submission.ipynb').read_text())
v6_cell = v6['cells'][2].copy()
v6_code = ''.join(v6_cell['source'])
assert "model = ProteinHistoryModel.from_data_dir('/kaggle/input/amp-pd')" in v6_code
v6_code = v6_code.replace("model = ProteinHistoryModel.from_data_dir('/kaggle/input/amp-pd')", "v6_model = ProteinHistoryModel.from_data_dir('/kaggle/input/amp-pd')")
v6_cell['source'] = v6_code.splitlines(keepends=True)
v6_cell['outputs'] = []
v6_cell['execution_count'] = None
loop = ''.join(winner['cells'][6]['source'])
old = '    sample_submission["rating"] = np.round( (sample_test["preds_lgb"] + sample_test["preds_nn"]) / 2)'
new = '''    champion = np.round((sample_test["preds_lgb"] + sample_test["preds_nn"]) / 2)
    v6_prediction = v6_model.predict_batch(test_df, test_peptides, test_proteins, sample_submission)
    assert v6_prediction.prediction_id.equals(sample_submission.prediction_id)
    sample_submission["rating"] = 0.75 * champion + 0.25 * v6_prediction.rating.to_numpy()'''
assert loop.count(old) == 1
winner['cells'][6]['source'] = loop.replace(old, new).splitlines(keepends=True)
winner['cells'].insert(6, v6_cell)
for cell in winner['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
path = ROOT / 'C_improved_model/champion_v6_blend_submission.ipynb'
path.write_text(json.dumps(winner, ensure_ascii=False, indent=1) + '\n')
print('Saved and syntax checked', path)

"""Create a three-seed, feature-enriched variant of the winning notebook."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / 'C_improved_model/winner_reproduction_submission.ipynb').read_text())
cell = notebook['cells'][5]
code = ''.join(cell['source'])
before = 'self.models = [run_train(self.cfg, df_train, None, None, verbose=False) for _ in range(self.cfg.bag_size)]'
after = '''self.models = []
        for seed in (101, 211, 307):
            self.cfg.seed = seed
            self.models.append(run_train(self.cfg, df_train, None, None, verbose=False))'''
assert before in code
code = code.replace(before, after)
assert 'cfg.bag_size = 1' in code
code = code.replace('cfg.bag_size = 1', 'cfg.bag_size = 3')
assert 'cfg.features += ["horizon_scaled"]' in code
code = code.replace('cfg.features += ["horizon_scaled"]',
                    'cfg.features += ["horizon_scaled", "blood_taken", "count_non12_visits"]')
cell['source'] = code.splitlines(keepends=True)
cell = notebook['cells'][6]
code = ''.join(cell['source'])
before = 'np.round( (sample_test["preds_lgb"] + sample_test["preds_nn"]) / 2)'
after = 'np.round(0.25 * sample_test["preds_lgb"] + 0.75 * sample_test["preds_nn"])'
assert before in code
cell['source'] = code.replace(before, after).splitlines(keepends=True)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        cell['outputs'] = []
        cell['execution_count'] = None
path = ROOT / 'C_improved_model/winner_neural_ensemble_submission.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Syntax checked:', path)

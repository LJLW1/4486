"""Adapt the downloaded first-place notebook to the established Kaggle kernel."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = Path('/private/tmp/amp_top1/1st-place-solution.ipynb')
destination = ROOT / 'C_improved_model/winner_reproduction_submission.ipynb'
notebook = json.loads(source.read_text())
for cell in notebook['cells']:
    if cell['cell_type'] != 'code':
        continue
    code = ''.join(cell['source']).replace('../input/amp-parkinsons-disease-progression-prediction', '/kaggle/input/amp-pd')
    ast.parse(code)
    cell['source'] = code.splitlines(keepends=True)
    cell['outputs'] = []
    cell['execution_count'] = None
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Syntax checked; saved', destination)

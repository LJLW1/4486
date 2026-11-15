"""Package the freshly trained, teammate-aligned model with the official API."""
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
source = (directory / 'teammate_aligned_model.py').read_text()
train = '''DATA_DIR = '/kaggle/input/amp-pd'
import sys
sys.path.append(DATA_DIR)
rows, features = load_training(DATA_DIR)
print('Training requests', len(rows), 'patients', rows.patient_id.nunique())
model = AlignedModel().fit(rows, features)
'''
api = '''import amp_pd_peptide
env = amp_pd_peptide.make_env()
for test, peptides, proteins, sample in env.iter_test():
    env.predict(model.predict_batch(test, peptides, proteins, sample))
'''
cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# 组员结构对齐模型\n',
    '共享 LightGBM 分类器和三个 128→64→16 PyTorch 网络，从数据训练所有参数。Codex 协助实现。\n',
    '特征索引、输出头和分量表权重对齐组员推理文件。原训练脚本未提供，训练行、优化器和超参数为本次明确选择，成绩须独立验证。\n']}]
for code in [source, train, api]:
    compile(code, '<notebook>', 'exec')
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': code.splitlines(True),
                  'execution_count': None, 'outputs': []})
notebook = {'cells': cells, 'metadata': {'kernelspec': {
    'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
    'nbformat': 4, 'nbformat_minor': 4}
path = directory / 'teammate_aligned_submission.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)

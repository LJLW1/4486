"""Package direct protein neural features for the official Kaggle API."""
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
base = (directory / 'teammate_aligned_model.py').read_text()
extension = (directory / 'teammate_neural_proteins.py').read_text().replace('from teammate_aligned_model import *', '')
training = '''DATA_DIR = '/kaggle/input/amp-pd'
import sys
sys.path.append(DATA_DIR)
rows, features = load_training(DATA_DIR)
protein_table = pd.read_csv(Path(DATA_DIR) / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
protein_values = protein_matrix(rows, protein_table)
protein_values[rows.source.eq(1).to_numpy()] = np.nan
print('Training requests', len(rows), 'patients', rows.patient_id.nunique())
model = ProteinModel().fit(rows, features, protein_values)
'''
api = '''import amp_pd_peptide
env = amp_pd_peptide.make_env()
for test, peptides, proteins, sample in env.iter_test():
    env.predict(model.predict_batch(test, peptides, proteins, sample))
'''
cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
    '# 访问历史和蛋白输入的神经网络\n',
    '将 P05060 和 O15240 的当前访问蛋白值及缺失标记输入网络，保留 V12 的共享 LightGBM 和融合权重。Codex 协助实现。\n']}]
for source in (base, extension, training, api):
    compile(source, '<notebook>', 'exec')
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': source.splitlines(True),
                  'execution_count': None, 'outputs': []})
notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 4}
path = directory / 'neural_direct_protein_submission.ipynb'
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)

"""Package architecture experiments into native Kaggle API notebooks."""
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
base = (directory / 'teammate_aligned_model.py').read_text()
protein = (directory / 'teammate_neural_proteins.py').read_text().replace('from teammate_aligned_model import *', '')
architecture = (directory / 'architectures.py').read_text()
architecture = architecture.replace('from teammate_neural_proteins import *', '')
architecture = architecture.replace('import teammate_neural_proteins as direct', '')
architecture = architecture.replace('original = direct.ProteinNetwork', "original = globals()['ProteinNetwork']")
architecture = architecture.replace("direct.ProteinNetwork = {'residual': ResidualNetwork, 'split': SplitNetwork}[self.variant]", "globals()['ProteinNetwork'] = {'residual': ResidualNetwork, 'split': SplitNetwork}[self.variant]")
architecture = architecture.replace('direct.ProteinNetwork = original', "globals()['ProteinNetwork'] = original")
training = '''DATA_DIR = '/kaggle/input/amp-pd'
import sys
sys.path.append(DATA_DIR)
rows, features = load_training(DATA_DIR)
proteins = pd.read_csv(Path(DATA_DIR) / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
protein_values = protein_matrix(rows, proteins)
protein_values[rows.source.eq(1).to_numpy()] = np.nan
print('Training requests', len(rows), 'patients', rows.patient_id.nunique())
model = ArchitectureModel(VARIANT).fit(rows, features, protein_values)
'''
api = '''import amp_pd_peptide
env = amp_pd_peptide.make_env()
for test, peptides, proteins, sample in env.iter_test():
    env.predict(model.predict_batch(test, peptides, proteins, sample))
'''
for variant in ['residual', 'split', 'distribution']:
    cells = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# 访问历史和蛋白架构实验\n', '实验方案：' + variant + '。从头训练，Codex 协助实现。\n']}]
    for text in [base, protein, architecture, 'VARIANT = ' + repr(variant) + '\n' + training, api]:
        compile(text, '<notebook>', 'exec')
        cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True), 'execution_count': None, 'outputs': []})
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, 'nbformat': 4, 'nbformat_minor': 4}
    path = directory / ('architecture_' + variant + '_submission.ipynb')
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    print(path)

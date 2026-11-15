"""Generate a readable notebook inspired by the teammate's prediction model."""
import json
from pathlib import Path

cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})


def code(text):
    compile(text, '<notebook>', 'exec')
    cells.append({'cell_type': 'code', 'metadata': {}, 'source': text.splitlines(True),
                  'execution_count': None, 'outputs': []})


markdown('''# 访问历史、LightGBM 与 PyTorch 融合

参考组员的访问历史和分量表融合方案。此实现由 Codex 协助生成，神经网络和 LightGBM 在本 notebook 中重新训练，成绩需要单独验证。相关公开方案及 AI 使用情况应在课程材料中如实披露。
''')
code('''import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMClassifier

DATA = Path('/kaggle/input/amp-pd')
sys.path.append(str(DATA))
torch.set_num_threads(1)
torch.manual_seed(4486)
TARGETS = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
KEYS = ['patient_id', 'visit_month']
ALPHA = np.array([0.625, 0.75, 0.75, 0.375])
FEATURES = ['visit_month', 'horizon', 'target_month', 'visible_visit_count',
            'previous_gap', 'protein_gap', 'protein_known', 'protein_now',
            'seen_6', 'seen_18', 'seen_48', 'first_nonzero',
            'first_nonzero_known', 'annual_observed']

class History:
    def __init__(self):
        self.visits = {}
        self.proteins = {}

    def update(self, visits, proteins):
        for table, cache in [(visits, self.visits), (proteins, self.proteins)]:
            for patient, month in table[KEYS].drop_duplicates().itertuples(index=False, name=None):
                cache.setdefault(int(patient), set()).add(int(month))

    def features(self, patient, month, horizon):
        known = sorted(v for v in self.visits.get(int(patient), set()) if v <= month)
        previous = [v for v in known if v < month]
        blood = [v for v in self.proteins.get(int(patient), set()) if v <= month]
        nonzero = [v for v in known if v > 0]
        return [month, horizon, month + horizon, len(known),
                month - previous[-1] if previous else 0,
                month - max(blood) if blood else 0, int(bool(blood)), int(month in blood),
                int(6 in known), int(18 in known), int(48 in known),
                nonzero[0] if nonzero else 0, int(bool(nonzero)),
                int(month >= 12 and 6 not in known and 18 not in known)]
''')
code('''# 仅使用当前及此前的访问记录构造特征；标签来自目标月。
clinical = pd.read_csv(DATA / 'train_clinical_data.csv')
proteins = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS)
peptides = pd.read_csv(DATA / 'train_peptides.csv', usecols=KEYS)
origins = pd.concat([proteins, peptides], ignore_index=True).drop_duplicates()
visits = pd.concat([clinical[KEYS], origins], ignore_index=True).drop_duplicates()
history = History()
history.update(visits, proteins)
requests = pd.concat([origins.assign(horizon=h) for h in [0, 6, 12, 24]], ignore_index=True)
requests['target_month'] = requests.visit_month + requests.horizon
labels = clinical.melt(id_vars=KEYS, value_vars=TARGETS, var_name='target', value_name='rating')
labels = labels.dropna(subset=['rating']).rename(columns={'visit_month': 'target_month'})
rows = requests.merge(labels, on=['patient_id', 'target_month'], how='inner')
x = np.array([history.features(p, t, h) for p, t, h in
              rows[['patient_id', 'visit_month', 'horizon']].itertuples(index=False, name=None)], dtype=np.float32)
print('Training requests:', len(rows), 'patients:', rows.patient_id.nunique())
''')
code('''# 每个量表训练一个分类器和一个 PyTorch 回归网络。
class ScoreNetwork(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(inputs, 32), nn.ReLU(),
                                    nn.Linear(32, 16), nn.ReLU(),
                                    nn.Linear(16, 1), nn.Softplus())

    def forward(self, features):
        return self.layers(features).squeeze(1)

models = {}
for target in TARGETS:
    mask = rows.target.eq(target).to_numpy()
    train_x, train_y = x[mask], rows.loc[mask, 'rating'].to_numpy(dtype=np.float32)
    classifier = LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=7,
                                max_depth=3, min_child_samples=50, reg_lambda=10,
                                random_state=4486, n_jobs=1, verbosity=-1)
    classifier.fit(train_x, train_y.astype(int))
    mean, scale = train_x.mean(axis=0), train_x.std(axis=0)
    scale[scale == 0] = 1
    tensor_x = torch.tensor((train_x - mean) / scale, dtype=torch.float32)
    tensor_y = torch.tensor(train_y, dtype=torch.float32)
    network = ScoreNetwork(len(FEATURES))
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01, weight_decay=0.01)
    for epoch in range(200):
        prediction = network(tensor_x)
        loss = (200 * torch.abs(prediction - tensor_y) / (prediction + tensor_y + 2)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    network.eval()
    models[target] = (classifier, network, mean, scale)
    print(target, 'training loss', round(float(loss.detach()), 4))

def decode_smape(classifier, features):
    probabilities = classifier.predict_proba(features)
    classes = classifier.classes_.astype(int)
    candidates = np.arange(int(classes.max()) + 1)
    loss = 200 * np.abs(classes[:, None] - candidates) / (classes[:, None] + candidates + 2)
    return candidates[(probabilities @ loss).argmin(axis=1)]
''')
code('''# 官方 API 按时间批次返回数据；状态持续保存在同一个 history 对象中。
import amp_pd_peptide
env = amp_pd_peptide.make_env()
history = History()
for test, test_peptides, test_proteins, sample in env.iter_test():
    history.update(pd.concat([test[KEYS], test_peptides[KEYS], test_proteins[KEYS]],
                             ignore_index=True), test_proteins)
    parsed = sample.prediction_id.str.extract(
        r'^(?P<patient_id>\\d+)_(?P<visit_month>\\d+)_(?P<target>updrs_[1-4])_plus_(?P<horizon>0|6|12|24)_months$')
    if parsed.isna().any().any():
        raise ValueError('Unexpected prediction identifier')
    for column in ['patient_id', 'visit_month', 'horizon']:
        parsed[column] = parsed[column].astype(int)
    features = np.array([history.features(p, t, h) for p, t, h in
                         parsed[['patient_id', 'visit_month', 'horizon']].itertuples(index=False, name=None)], dtype=np.float32)
    prediction = np.empty(len(sample), dtype=float)
    for i, target in enumerate(TARGETS):
        mask = parsed.target.eq(target).to_numpy()
        if not mask.any():
            continue
        classifier, network, mean, scale = models[target]
        tree_prediction = decode_smape(classifier, features[mask])
        with torch.no_grad():
            neural_prediction = network(torch.tensor((features[mask] - mean) / scale, dtype=torch.float32)).numpy()
        prediction[mask] = (1 - ALPHA[i]) * tree_prediction + ALPHA[i] * neural_prediction
    output = sample.copy()
    output['rating'] = prediction
    env.predict(output)
''')

notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4, 'nbformat_minor': 4}
path = Path(__file__).with_name('teammate_history_pytorch_submission.ipynb')
path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print(path)

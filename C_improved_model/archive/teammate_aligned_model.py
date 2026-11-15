"""Teammate-aligned shared LightGBM and three 16-head PyTorch networks.

All parameters and feature scalers are fitted from data. Codex-assisted implementation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMClassifier

KEYS = ['patient_id', 'visit_month']
TARGETS = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']
HORIZONS = [0, 6, 12, 24]
MONTHS = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]
TREE_INDICES = [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14]
NN_INDICES = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14]
ALPHA = np.array([0.625, 0.75, 0.75, 0.375])


class History:
    def __init__(self):
        self.visits, self.proteins = {}, {}

    def update(self, visits, proteins):
        for frame, cache in [(visits, self.visits), (proteins, self.proteins)]:
            for patient, month in frame[KEYS].drop_duplicates().itertuples(index=False, name=None):
                cache.setdefault(int(patient), set()).add(int(month))

    def features(self, patient, month, horizon, source=0):
        known = sorted(v for v in self.visits.get(int(patient), set()) if v <= month)
        before = [v for v in known if v < month]
        blood = [v for v in self.proteins.get(int(patient), set()) if v <= month]
        nonzero = [v for v in known if v > 0]
        return [month, horizon, month + horizon, len(known),
                month - before[-1] if before else 0, month - max(blood) if blood else 0,
                int(bool(blood)), int(month in blood), int(6 in known), int(18 in known),
                int(48 in known), nonzero[0] if nonzero else 0, int(bool(nonzero)),
                int(month >= 12 and 6 not in known and 18 not in known), source]


def make_training_data(clinical, proteins, source):
    history = History()
    history.update(clinical[KEYS], proteins)
    origins = clinical.loc[clinical.visit_month.isin(MONTHS), KEYS].drop_duplicates()
    requests = pd.concat([origins.assign(horizon=h) for h in HORIZONS], ignore_index=True)
    requests['target_month'] = requests.visit_month + requests.horizon
    labels = clinical.melt(id_vars=KEYS, value_vars=TARGETS,
                           var_name='target', value_name='rating').dropna(subset=['rating'])
    labels = labels.rename(columns={'visit_month': 'target_month'})
    rows = requests.merge(labels, on=['patient_id', 'target_month'], validate='many_to_many')
    rows['target_index'] = rows.target.map({name: i for i, name in enumerate(TARGETS)})
    rows['head'] = rows.horizon.map({h: i * 4 for i, h in enumerate(HORIZONS)}) + rows.target_index
    rows['source'] = source
    features = np.array([history.features(p, t, h, source) for p, t, h in
                         rows[KEYS + ['horizon']].itertuples(index=False, name=None)], dtype=np.float32)
    return rows.reset_index(drop=True), features


class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(NN_INDICES), 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16), nn.Softplus())

    def forward(self, features):
        return self.layers(features)


class AlignedModel:
    def fit(self, rows, features, epochs=200):
        torch.set_num_threads(1)
        tree_x = np.column_stack([features[:, TREE_INDICES], rows.target_index.to_numpy()])
        self.classifier = LGBMClassifier(n_estimators=300, num_leaves=7, learning_rate=0.03,
                                         min_child_samples=50, reg_lambda=10,
                                         random_state=4486, n_jobs=1, verbosity=-1)
        self.classifier.fit(tree_x, rows.rating.astype(int))
        # One origin supplies up to sixteen labels; missing labels remain masked.
        origin_keys = rows[['patient_id', 'visit_month', 'source']]
        unique = ~origin_keys.duplicated()
        origin_index = pd.MultiIndex.from_frame(origin_keys.loc[unique])
        row_origin = origin_index.get_indexer(pd.MultiIndex.from_frame(origin_keys))
        values = features[unique][:, NN_INDICES]
        self.mean, self.scale = values.mean(axis=0), values.std(axis=0)
        self.scale[self.scale == 0] = 1
        x = torch.tensor((values - self.mean) / self.scale, dtype=torch.float32)
        y = torch.zeros((len(values), 16), dtype=torch.float32)
        observed = torch.zeros_like(y, dtype=torch.bool)
        heads = rows['head'].to_numpy()
        y[row_origin, heads] = torch.tensor(rows.rating.to_numpy(), dtype=torch.float32)
        observed[row_origin, heads] = True
        self.networks = []
        for seed in [4486, 4487, 4488]:
            torch.manual_seed(seed)
            network = ScoreNetwork()
            optimizer = torch.optim.AdamW(network.parameters(), lr=0.003, weight_decay=0.01)
            for epoch in range(epochs):
                for batch in torch.randperm(len(x)).split(256):
                    prediction = network(x[batch])
                    loss = (200 * abs(prediction - y[batch]) / (prediction + y[batch] + 2))[observed[batch]].mean()
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            network.eval()
            self.networks.append(network)
            print('network seed', seed, 'trained', flush=True)
        self.history = History()
        return self

    def predict_features(self, features, target_indices, heads):
        tree_x = np.column_stack([features[:, TREE_INDICES], target_indices])
        probability = self.classifier.predict_proba(tree_x)
        classes = self.classifier.classes_.astype(int)
        candidates = np.arange(int(classes.max()) + 1)
        risk = 200 * abs(classes[:, None] - candidates) / (classes[:, None] + candidates + 2)
        tree_prediction = candidates[(probability @ risk).argmin(axis=1)]
        x = torch.tensor((features[:, NN_INDICES] - self.mean) / self.scale, dtype=torch.float32)
        with torch.no_grad():
            predictions = np.mean([net(x).numpy() for net in self.networks], axis=0)
        neural_prediction = predictions[np.arange(len(features)), heads]
        return (1 - ALPHA[target_indices]) * tree_prediction + ALPHA[target_indices] * neural_prediction

    def predict_batch(self, test, peptides, proteins, sample):
        self.history.update(test, proteins)
        parsed = sample.prediction_id.str.extract(r'^(\d+)_(\d+)_updrs_([1-4])_plus_(0|6|12|24)_months$')
        if parsed.isna().any().any():
            raise ValueError('Unexpected prediction identifier')
        parsed = parsed.astype(int)
        features = np.array([self.history.features(p, t, h) for p, t, k, h in
                             parsed.itertuples(index=False, name=None)], dtype=np.float32)
        targets = parsed[2].to_numpy() - 1
        heads = parsed[3].map({h: i * 4 for i, h in enumerate(HORIZONS)}).to_numpy() + targets
        output = sample.copy()
        output['rating'] = self.predict_features(features, targets, heads)
        if not np.isfinite(output.rating).all() or output.rating.lt(0).any():
            raise ValueError('Invalid predictions')
        return output


def load_training(data_dir):
    data_dir = Path(data_dir)
    clinical = pd.read_csv(data_dir / 'train_clinical_data.csv')
    supplemental = pd.read_csv(data_dir / 'supplemental_clinical_data.csv')
    supplemental.loc[supplemental.visit_month.eq(5), 'visit_month'] = 6
    proteins = pd.read_csv(data_dir / 'train_proteins.csv', usecols=KEYS)
    main_rows, main_x = make_training_data(clinical, proteins, 0)
    extra_rows, extra_x = make_training_data(supplemental, pd.DataFrame(columns=KEYS), 1)
    return pd.concat([main_rows, extra_rows], ignore_index=True), np.concatenate([main_x, extra_x])

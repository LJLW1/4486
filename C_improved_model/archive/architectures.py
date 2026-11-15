"""Three candidate neural architectures using V12 features and two proteins."""
from teammate_neural_proteins import *
import teammate_neural_proteins as direct
import torch.nn.functional as F


class ResidualNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ScoreNetwork()
        self.residual_scale = 1.0  # V18 default; adjust only during held-out evaluation
        self.correction = nn.Sequential(nn.Linear(len(NN_INDICES) + 4, 16), nn.ReLU(), nn.Linear(16, 16))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, features):
        base = self.base(features[:, :len(NN_INDICES)])
        available = features[:, -2:].amax(dim=1, keepdim=True)
        delta = self.correction(features) * available * self.residual_scale
        return (base + delta).clamp_min(0)


class SplitNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(len(NN_INDICES) + 4, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, 4) for _ in range(4)])

    def forward(self, features):
        hidden = self.shared(features)
        # Columns are horizon-major, matching the original sixteen heads.
        return F.softplus(torch.stack([head(hidden) for head in self.heads], dim=2).reshape(-1, 16))


class DistributionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(NN_INDICES) + 4, 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16 * 87))

    def forward(self, features):
        return self.layers(features).reshape(-1, 16, 87)


class ArchitectureModel(ProteinModel):
    def __init__(self, variant):
        self.variant = variant

    def fit(self, rows, features, protein_values, epochs=200, base_epochs=100):
        if self.variant in ('residual', 'residual_two_stage', 'split'):
            original = direct.ProteinNetwork
            direct.ProteinNetwork = {'residual': ResidualNetwork, 'residual_two_stage': ResidualNetwork, 'split': SplitNetwork}[self.variant]
            try:
                if self.variant == 'residual_two_stage':
                    super().fit(rows, features, protein_values, epochs, training_mode='two_stage', base_epochs=base_epochs)
                else:
                    super().fit(rows, features, protein_values, epochs)
            finally:
                direct.ProteinNetwork = original
            return self
        torch.set_num_threads(1)
        tree_x = np.column_stack([features[:, TREE_INDICES], rows.target_index.to_numpy()])
        self.classifier = LGBMClassifier(n_estimators=300, num_leaves=7, learning_rate=0.03,
                                         min_child_samples=50, reg_lambda=10,
                                         random_state=4486, n_jobs=1, verbosity=-1)
        self.classifier.fit(tree_x, rows.rating.astype(int))
        origin_keys = rows[['patient_id', 'visit_month', 'source']]
        unique = ~origin_keys.duplicated()
        index = pd.MultiIndex.from_frame(origin_keys.loc[unique])
        origins = index.get_indexer(pd.MultiIndex.from_frame(origin_keys))
        values, protein = features[unique][:, NN_INDICES], protein_values[unique]
        self.mean, self.scale = values.mean(axis=0), values.std(axis=0)
        self.scale[self.scale == 0] = 1
        self.protein_mean = np.nan_to_num(np.nanmean(protein, axis=0), nan=0.0)
        stdev = np.nanstd(protein, axis=0)
        self.protein_scale = np.where(np.isfinite(stdev) & (stdev > 0), stdev, 1)
        x = torch.tensor(self.neural_features(values, protein), dtype=torch.float32)
        y = torch.full((len(values), 16), -100, dtype=torch.long)
        y[origins, rows['head'].to_numpy()] = torch.tensor(rows.rating.to_numpy(), dtype=torch.long)
        self.networks = []
        for seed in [4486, 4487, 4488]:
            torch.manual_seed(seed)
            net = DistributionNetwork()
            optim = torch.optim.AdamW(net.parameters(), lr=0.003, weight_decay=0.01)
            for epoch in range(epochs):
                for batch in torch.randperm(len(x)).split(256):
                    loss = F.cross_entropy(net(x[batch]).reshape(-1, 87), y[batch].reshape(-1), ignore_index=-100)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite distribution loss')
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
            net.eval()
            self.networks.append(net)
            print(self.variant, 'seed', seed, 'trained', flush=True)
        self.history = History()
        return self

    def predict_features(self, features, target_indices, heads, protein_values):
        if self.variant != 'distribution':
            return super().predict_features(features, target_indices, heads, protein_values)
        tree_x = np.column_stack([features[:, TREE_INDICES], target_indices])
        probabilities = self.classifier.predict_proba(tree_x)
        classes = self.classifier.classes_.astype(int)
        candidates = np.arange(87)
        tree_risk = 200 * abs(classes[:, None] - candidates) / (classes[:, None] + candidates + 2)
        tree = candidates[(probabilities @ tree_risk).argmin(axis=1)]
        x = torch.tensor(self.neural_features(features[:, NN_INDICES], protein_values), dtype=torch.float32)
        with torch.no_grad():
            probs = np.mean([torch.softmax(net(x), dim=-1).numpy() for net in self.networks], axis=0)
        chosen = probs[np.arange(len(features)), heads]
        score = 200 * abs(candidates[:, None] - candidates) / (candidates[:, None] + candidates + 2)
        neural = candidates[(chosen @ score).argmin(axis=1)]
        return (1 - ALPHA[target_indices]) * tree + ALPHA[target_indices] * neural

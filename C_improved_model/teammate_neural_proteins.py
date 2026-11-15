"""Direct protein inputs for the teammate-aligned neural ensemble."""
from teammate_aligned_model import *

PROTEINS = ['P05060', 'O15240']


def protein_matrix(frame, protein_table):
    selected = protein_table[protein_table.UniProt.isin(PROTEINS)]
    wide = selected.pivot(index=KEYS, columns='UniProt', values='NPX').reindex(columns=PROTEINS)
    return np.log1p(wide.reindex(pd.MultiIndex.from_frame(frame[KEYS])).to_numpy(dtype=np.float32))


class ProteinNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(len(NN_INDICES) + 2 * len(PROTEINS), 128), nn.ReLU(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 16), nn.Softplus())

    def forward(self, features):
        return self.layers(features)


class ProteinModel(AlignedModel):
    def fit(self, rows, features, protein_values, epochs=200):
        torch.set_num_threads(1)
        tree_x = np.column_stack([features[:, TREE_INDICES], rows.target_index.to_numpy()])
        self.classifier = LGBMClassifier(n_estimators=300, num_leaves=7, learning_rate=0.03,
                                         min_child_samples=50, reg_lambda=10,
                                         random_state=4486, n_jobs=1, verbosity=-1)
        self.classifier.fit(tree_x, rows.rating.astype(int))
        origin_keys = rows[['patient_id', 'visit_month', 'source']]
        unique = ~origin_keys.duplicated()
        origin_index = pd.MultiIndex.from_frame(origin_keys.loc[unique])
        row_origin = origin_index.get_indexer(pd.MultiIndex.from_frame(origin_keys))
        values = features[unique][:, NN_INDICES]
        self.mean, self.scale = values.mean(axis=0), values.std(axis=0)
        self.scale[self.scale == 0] = 1
        protein = protein_values[unique]
        self.protein_mean = np.nanmean(protein, axis=0)
        self.protein_scale = np.nanstd(protein, axis=0)
        self.protein_mean = np.nan_to_num(self.protein_mean, nan=0.0)
        self.protein_scale = np.where(np.isfinite(self.protein_scale) & (self.protein_scale > 0), self.protein_scale, 1)
        x = torch.tensor(self.neural_features(values, protein), dtype=torch.float32)
        y = torch.zeros((len(values), 16), dtype=torch.float32)
        observed = torch.zeros_like(y, dtype=torch.bool)
        y[row_origin, rows['head'].to_numpy()] = torch.tensor(rows.rating.to_numpy(), dtype=torch.float32)
        observed[row_origin, rows['head'].to_numpy()] = True
        self.networks = []
        for seed in [4486, 4487, 4488]:
            torch.manual_seed(seed)
            network = ProteinNetwork()
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
            print('protein network seed', seed, 'trained', flush=True)
        self.history = History()
        return self

    def neural_features(self, values, proteins):
        mask = np.isfinite(proteins).astype(np.float32)
        standardized = (proteins - self.protein_mean) / self.protein_scale
        standardized = np.nan_to_num(standardized, nan=0.0)
        return np.column_stack([(values - self.mean) / self.scale, standardized, mask]).astype(np.float32)

    def predict_features(self, features, target_indices, heads, protein_values):
        tree_x = np.column_stack([features[:, TREE_INDICES], target_indices])
        probabilities = self.classifier.predict_proba(tree_x)
        classes = self.classifier.classes_.astype(int)
        candidates = np.arange(int(classes.max()) + 1)
        risk = 200 * abs(classes[:, None] - candidates) / (classes[:, None] + candidates + 2)
        tree_prediction = candidates[(probabilities @ risk).argmin(axis=1)]
        x = torch.tensor(self.neural_features(features[:, NN_INDICES], protein_values), dtype=torch.float32)
        with torch.no_grad():
            neural = np.mean([net(x).numpy() for net in self.networks], axis=0)
        prediction = neural[np.arange(len(features)), heads]
        return (1 - ALPHA[target_indices]) * tree_prediction + ALPHA[target_indices] * prediction

    def predict_batch(self, test, peptides, proteins, sample):
        self.history.update(test, proteins)
        parsed = sample.prediction_id.str.extract(r'^(\d+)_(\d+)_updrs_([1-4])_plus_(0|6|12|24)_months$')
        if parsed.isna().any().any():
            raise ValueError('Unexpected prediction identifier')
        parsed = parsed.astype(int)
        parsed.columns = ['patient_id', 'visit_month', 'target', 'horizon']
        features = np.array([self.history.features(p, t, h) for p, t, k, h in
                             parsed.itertuples(index=False, name=None)], dtype=np.float32)
        targets = parsed.target.to_numpy() - 1
        heads = parsed.horizon.map({h: i * 4 for i, h in enumerate(HORIZONS)}).to_numpy() + targets
        current = protein_matrix(parsed, proteins)
        output = sample.copy()
        output['rating'] = self.predict_features(features, targets, heads, current)
        if not np.isfinite(output.rating).all() or output.rating.lt(0).any():
            raise ValueError('Invalid predictions')
        return output

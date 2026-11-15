"""Train a small pairwise protein-similarity network on patient-held-out folds.

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 C_improved_model/neural_protein_similarity.py
The candidate pair is fixed from earlier exploratory work; evaluation remains exploratory.
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows
from protein_history_experiment import protein_values, predict_with_proteins

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/neural_protein_results'
PROTEINS = ['P05060', 'O15240']
STEPS = 600
QUERY_BATCH = 32
REFERENCE_BATCH = 512
TEMPERATURE = 1.0
SEED = 4486


class SimilarityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(6, 12), nn.Tanh(), nn.Linear(12, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, query, reference):
        # Shapes: query [Q,1,2], reference [1,R,2].
        q_ok, r_ok = torch.isfinite(query), torch.isfinite(reference)
        delta = torch.where(q_ok & r_ok, (query - reference).abs(), 0.0)
        q_missing = (~q_ok).to(delta.dtype).expand_as(delta)
        r_missing = (~r_ok).to(delta.dtype).expand_as(delta)
        features = torch.cat((delta, q_missing, r_missing), dim=-1)
        weights = 1.0 + 0.5 * torch.tanh(self.layers(features).squeeze(-1))
        # No query measurement: use the original time/history weights unchanged.
        return torch.where(q_ok.any(dim=-1), weights, torch.ones_like(weights))


def arrays(rows, z):
    values = z.reindex(pd.MultiIndex.from_frame(rows[KEYS])).to_numpy(dtype=np.float32)
    result = {name: rows[name].to_numpy() for name in ['patient_id', 'target', 'target_month', 'history']}
    if 'rating' in rows:
        result['rating'] = rows.rating.to_numpy()
    return result | {'protein': values}


def train_network(train, normalized, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = SimilarityNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.01)
    data = arrays(train, normalized)
    groups = {target: np.flatnonzero(data['target'] == target) for target in sorted(train.target.unique())}
    skipped = 0
    for step in range(STEPS):
        target = list(groups)[step % len(groups)]
        indices = groups[target]
        qi = rng.choice(indices, QUERY_BATCH, replace=True)
        ri = rng.choice(indices, min(REFERENCE_BATCH, len(indices)), replace=False)
        # Exclude every reference request belonging to the query patient.
        eligible = data['patient_id'][qi, None] != data['patient_id'][None, ri]
        same_history = eligible & (data['history'][qi, None] == data['history'][None, ri])
        mask = np.where(same_history.any(axis=1, keepdims=True), same_history, eligible)
        active = mask.any(axis=1)
        if not active.any():
            skipped += 1
            continue
        qi, mask = qi[active], mask[active]
        assert not ((data['patient_id'][qi, None] == data['patient_id'][None, ri]) & mask).any()
        query = torch.from_numpy(data['protein'][qi])[:, None, :]
        ref = torch.from_numpy(data['protein'][ri])[None, :, :]
        dt = (data['target_month'][qi, None] - data['target_month'][None, ri]) / 12.0
        time = torch.from_numpy(np.exp(-0.5 * dt ** 2).astype(np.float32))
        weights = net(query, ref) * time * torch.from_numpy(mask.astype(np.float32))
        candidates = torch.arange(int(data['rating'][indices].max()) + 1, dtype=torch.float32)
        truth = torch.from_numpy(data['rating'][ri].astype(np.float32))[:, None]
        candidate_loss = 200 * (truth - candidates).abs() / (truth + candidates + 2)
        risk = (weights @ candidate_loss) / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        prediction = (torch.softmax(-risk / TEMPERATURE, dim=1) * candidates).sum(dim=1)
        actual = torch.from_numpy(data['rating'][qi].astype(np.float32))
        loss = (200 * (actual - prediction).abs() / (actual + prediction + 2)).mean()
        if not torch.isfinite(loss):
            raise ValueError('Non-finite training loss')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return net.eval(), skipped


def predict_network(net, train, valid, normalized, soft=False):
    tr, va = arrays(train, normalized), arrays(valid, normalized)
    output = np.empty(len(valid), dtype=float)
    groups = valid.reset_index(drop=True).groupby(['target', 'target_month', 'history'], sort=False).indices
    with torch.no_grad():
        for (target, month, history), positions in groups.items():
            target_mask = tr['target'] == target
            selected = target_mask & (tr['history'] == history)
            mask = selected if selected.any() else target_mask
            truth = tr['rating'][mask]
            candidates = np.arange(int(tr['rating'][target_mask].max()) + 1, dtype=np.float32)
            losses = 200 * abs(truth[:, None] - candidates) / (truth[:, None] + candidates + 2)
            times = np.exp(-0.5 * ((tr['target_month'][mask] - month) / 12.0) ** 2)
            refs = torch.from_numpy(tr['protein'][mask])[None, :, :]
            for start in range(0, len(positions), 64):
                idx = positions[start:start+64]
                q = torch.from_numpy(va['protein'][idx])[:, None, :]
                weights = net(q, refs).numpy() * times
                risk = (weights @ losses) / weights.sum(axis=1, keepdims=True).clip(min=1e-12)
                if soft:
                    prob = torch.softmax(-torch.from_numpy(risk) / TEMPERATURE, dim=1).numpy()
                    output[idx] = prob @ candidates
                else:
                    output[idx] = candidates[risk.argmin(axis=1)]
    return output


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    rows = training_rows(clinical, omics, 'union').merge(
        pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv'), on='patient_id', validate='many_to_one')
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    output = []
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        assert set(train.patient_id).isdisjoint(valid.patient_id)
        z = protein_values(wide, set(train.patient_id), PROTEINS)
        net, skipped = train_network(train, z, SEED + int(fold))
        predictions = {'history': VisitHistoryModel(train).predict_rows(valid).to_numpy(),
                       'fixed_protein': predict_with_proteins(train, valid, z),
                       'neural_hard': predict_network(net, train, valid, z),
                       'neural_soft': predict_network(net, train, valid, z, soft=True)}
        for name, pred in predictions.items():
            part = valid[['patient_id', 'visit_month', 'target', 'target_month', 'horizon', 'rating', 'fold']].copy()
            part['model'], part['prediction'] = name, pred
            part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            output.append(part)
            print(f'fold={fold} {name}: {part.loss.mean():.4f}', flush=True)
        print(f'fold={fold} training steps={STEPS}, skipped={skipped}', flush=True)
        torch.save(net.state_dict(), OUT / f'fold_{fold}_network.pt')
    full = pd.concat(output, ignore_index=True)
    assert not full.duplicated(['model', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    assert full.groupby('model').size().eq(len(rows)).all()
    full.to_csv(OUT / 'oof_predictions.csv', index=False)
    fold_scores = full.groupby(['model', 'fold']).loss.mean().unstack()
    scores = full.groupby('model').loss.agg(oof_score='mean', n='size')
    scores['fold_mean'] = fold_scores.mean(axis=1)
    scores['fold_std'] = fold_scores.std(axis=1, ddof=0)
    scores.join(fold_scores.add_prefix('fold_')).to_csv(OUT / 'comparison.csv')
    full.groupby(['model', 'target', 'horizon']).loss.agg(['mean', 'size']).to_csv(OUT / 'breakdown.csv')
    print(scores.to_string(), flush=True)


if __name__ == '__main__':
    main()

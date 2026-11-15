"""Compare reference sampling and fixed-formula residual initialization."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from neural_protein_similarity import (SimilarityNet, PROTEINS, DATA, ROOT, STEPS, SEED,
    QUERY_BATCH, REFERENCE_BATCH, TEMPERATURE, arrays, predict_network, train_network)
from protein_history_experiment import protein_values, predict_with_proteins
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows

OUT = ROOT / 'C_improved_model/neural_ablation_results'


class FixedResidualNet(SimilarityNet):
    def forward(self, query, reference):
        q_ok, r_ok = torch.isfinite(query), torch.isfinite(reference)
        diff = torch.where(q_ok & r_ok, query - reference, 0.0)
        similarity = torch.where(q_ok & r_ok, torch.exp(-0.5 * diff.square()), 1.0).mean(dim=-1)
        fixed = 0.5 + 0.5 * similarity
        # Superclass produces [0.5, 1.5]; residual adjustment is at most +/-20%.
        adjustment = 1.0 + 0.4 * (super().forward(query, reference) - 1.0)
        return fixed * adjustment


def train_variant(train, normalized, seed, match_sampling, residual):
    if not match_sampling and not residual:
        return train_network(train, normalized, seed)[0]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = FixedResidualNet() if residual else SimilarityNet()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.003, weight_decay=0.01)
    data = arrays(train, normalized)
    target_groups = {t: np.flatnonzero(data['target'] == t) for t in sorted(train.target.unique())}
    grouped = {}
    for t, ix in target_groups.items():
        for history in np.unique(data['history'][ix]):
            grouped[(t, history)] = ix[data['history'][ix] == history]
    for step in range(STEPS):
        if match_sampling:
            target = list(target_groups)[step % len(target_groups)]
            groups = [g for g in grouped if g[0] == target]
            probabilities = np.array([len(grouped[g]) for g in groups], dtype=float)
            probabilities /= probabilities.sum()
            target, hist = groups[rng.choice(len(groups), p=probabilities)]
            qpool = grouped[(target, hist)]
            qi = rng.choice(qpool, QUERY_BATCH, replace=True)
            # If an entire history group belongs to one patient, no query in it
            # can use a same-history reference after patient exclusion.
            rpool = qpool if len(np.unique(data['patient_id'][qpool])) > 1 else target_groups[target]
            ri = rng.choice(rpool, min(REFERENCE_BATCH, len(rpool)), replace=False)
            eligible = data['patient_id'][qi, None] != data['patient_id'][None, ri]
            if rpool is qpool:
                mask = eligible
            else:
                mask = eligible  # no same-history references exist for these patients
        else:
            target = list(target_groups)[step % len(target_groups)]
            indices = target_groups[target]
            qi = rng.choice(indices, QUERY_BATCH, replace=True)
            ri = rng.choice(indices, min(REFERENCE_BATCH, len(indices)), replace=False)
            eligible = data['patient_id'][qi, None] != data['patient_id'][None, ri]
            same = eligible & (data['history'][qi, None] == data['history'][None, ri])
            mask = np.where(same.any(axis=1, keepdims=True), same, eligible)
        active = mask.any(axis=1)
        if not active.any():
            continue
        qi, mask = qi[active], mask[active]
        assert not ((data['patient_id'][qi, None] == data['patient_id'][None, ri]) & mask).any()
        query = torch.from_numpy(data['protein'][qi])[:, None, :]
        ref = torch.from_numpy(data['protein'][ri])[None, :, :]
        dt = (data['target_month'][qi, None] - data['target_month'][None, ri]) / 12.0
        time = torch.from_numpy(np.exp(-0.5 * dt ** 2).astype(np.float32))
        weights = net(query, ref) * time * torch.from_numpy(mask.astype(np.float32))
        candidates = torch.arange(int(data['rating'][target_groups[target]].max()) + 1, dtype=torch.float32)
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
    return net.eval()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    rows = training_rows(clinical, omics, 'union').merge(
        pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv'), on='patient_id', validate='many_to_one')
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    wide = protein.pivot(index=KEYS, columns='UniProt', values='NPX').sort_index()
    configs = [('original', False, False), ('matched_sampling', True, False),
               ('fixed_residual', False, True), ('both', True, True)]
    output=[]
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        assert set(train.patient_id).isdisjoint(valid.patient_id)
        z = protein_values(wide, set(train.patient_id), PROTEINS)
        predictions = {'history': VisitHistoryModel(train).predict_rows(valid).to_numpy(),
                       'fixed_protein': predict_with_proteins(train, valid, z)}
        for name, matched, residual in configs:
            net = train_variant(train, z, SEED + int(fold), matched, residual)
            predictions[name] = predict_network(net, train, valid, z)
            torch.save(net.state_dict(), OUT / f'{name}_fold_{fold}.pt')
        for name, pred in predictions.items():
            part = valid[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
            part['model'], part['prediction'] = name, pred
            part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
            assert np.isfinite(pred).all() and (pred >= 0).all()
            output.append(part)
            print(f'fold={fold} {name}: {part.loss.mean():.4f}',flush=True)
    full = pd.concat(output, ignore_index=True)
    assert not full.duplicated(['model','patient_id','visit_month','target','horizon']).any()
    assert full.groupby('model').size().eq(len(rows)).all()
    full.to_csv(OUT / 'oof_predictions.csv',index=False)
    folds = full.groupby(['model','fold']).loss.mean().unstack()
    scores = full.groupby('model').loss.agg(oof_score='mean',n='size')
    scores['fold_mean'], scores['fold_std'] = folds.mean(axis=1), folds.std(axis=1, ddof=0)
    scores.join(folds.add_prefix('fold_')).to_csv(OUT / 'comparison.csv')
    full.groupby(['model','target','horizon']).loss.agg(['mean','size']).to_csv(OUT / 'breakdown.csv')
    print(scores.join(folds.add_prefix('fold_')).to_string(), flush=True)


if __name__ == '__main__':
    main()

"""Patient-fold LightGBM plus three neural distribution models and equal-weight fusion.

See fusion_results/实验说明.md for scope and limits. Runs locally, not a submission.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMClassifier
from visit_history_model import KEYS, VisitHistoryModel, read_data, training_rows
from protein_history_experiment import ranking, protein_values, predict_with_proteins

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/fusion_results'
SMALL = ['P05060', 'O15240', 'P17174']
SEED = 4486
EPOCHS = 100
PARAMS = dict(n_estimators=150, learning_rate=.03, num_leaves=7, max_depth=3,
              min_child_samples=50, reg_lambda=10.0, random_state=SEED,
              n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)


def history_features(rows, clinical, omics, proteins):
    """Causal visit and protein history; no clinical scores are inputs."""
    union = pd.concat([clinical[KEYS], omics]).drop_duplicates()
    pvisits = proteins[KEYS].drop_duplicates()
    visit_map = {pid: sorted(group.visit_month.unique()) for pid, group in union.groupby('patient_id')}
    protein_map = {pid: sorted(group.visit_month.unique()) for pid, group in pvisits.groupby('patient_id')}
    unique = rows[KEYS].drop_duplicates().copy()
    features = []
    for row in unique.itertuples(index=False):
        history = [m for m in visit_map.get(row.patient_id, []) if m <= row.visit_month]
        measured = [m for m in protein_map.get(row.patient_id, []) if m <= row.visit_month]
        features.append([int(6 in history), int(18 in history), int(bool(measured)),
                         int(row.visit_month in measured)])
    unique[['seen_6', 'seen_18', 'protein_known', 'protein_now']] = np.asarray(features)
    out = rows[KEYS].merge(unique, on=KEYS, how='left', validate='many_to_one')
    out[['visit_month', 'horizon', 'target_month']] = rows[['visit_month', 'horizon', 'target_month']].to_numpy()
    return out[['visit_month', 'horizon', 'target_month', 'seen_6', 'seen_18', 'protein_known', 'protein_now']]


def decision(prob):
    assert np.isfinite(prob).all() and (prob >= 0).all()
    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
    classes = np.arange(prob.shape[1])
    loss = 200 * abs(classes[:, None] - classes) / (classes[:, None] + classes + 2)
    return classes[(prob @ loss).argmin(axis=1)]


def spread_prob(prob, classes, nclasses):
    result = np.zeros((len(prob), nclasses), dtype=np.float32)
    result[:, np.asarray(classes, dtype=int)] = prob
    return result


class HistoryMLP(nn.Module):
    def __init__(self, inputs, nclasses):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 24), nn.ReLU(), nn.Linear(24, 16),
                                 nn.ReLU(), nn.Linear(16, nclasses))
    def forward(self, base, protein, mask):
        return self.net(base)


class SmallMLP(nn.Module):
    def __init__(self, inputs, nclasses, nproteins):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs + 2*nproteins, 24), nn.ReLU(),
                                 nn.Linear(24, 16), nn.ReLU(), nn.Linear(16, nclasses))
    def forward(self, base, protein, mask):
        return self.net(torch.cat([base, protein, mask], dim=1))


class SetMLP(nn.Module):
    def __init__(self, inputs, nclasses, nproteins):
        super().__init__()
        self.embedding = nn.Embedding(nproteins, 3)
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4), nn.ReLU())
        self.net = nn.Sequential(nn.Linear(inputs+4, 24), nn.ReLU(), nn.Linear(24, nclasses))
    def forward(self, base, protein, mask):
        ids = torch.arange(protein.shape[1], device=protein.device)
        identity = self.embedding(ids).unsqueeze(0).expand(len(base), -1, -1)
        encoded = self.encoder(torch.cat([protein.unsqueeze(-1), identity], dim=-1))
        pooled = (encoded*mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.net(torch.cat([base, pooled], dim=1))


def net_predict(name, base_train, base_valid, ztrain, zvalid, y, classes, seed):
    torch.manual_seed(seed)
    inputs, nclasses, nproteins = base_train.shape[1], len(classes), ztrain.shape[1]
    if name == 'nn_history': net = HistoryMLP(inputs, nclasses)
    elif name == 'nn_small': net = SmallMLP(inputs, nclasses, nproteins)
    else: net = SetMLP(inputs, nclasses, nproteins)
    def tensor(z):
        mask = np.isfinite(z).astype(np.float32)
        return torch.from_numpy(np.nan_to_num(z, nan=0.0).astype(np.float32)), torch.from_numpy(mask)
    trainp, trainmask = tensor(ztrain)
    validp, validmask = tensor(zvalid)
    xtrain = torch.from_numpy(base_train.astype(np.float32))
    xvalid = torch.from_numpy(base_valid.astype(np.float32))
    labels = torch.from_numpy(np.searchsorted(classes, y).astype(np.int64))
    optim = torch.optim.AdamW(net.parameters(), lr=.01, weight_decay=.01)
    for _ in range(EPOCHS):
        net.train()
        loss = nn.functional.cross_entropy(net(xtrain, trainp, trainmask), labels)
        if not torch.isfinite(loss): raise ValueError('Nonfinite network loss')
        optim.zero_grad(); loss.backward(); optim.step()
    net.eval()
    with torch.no_grad():
        return torch.softmax(net(xvalid, validp, validmask), dim=1).numpy()


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    clinical, omics = read_data(DATA)
    rows = training_rows(clinical, omics, 'union').merge(
        pd.read_csv(ROOT/'artifacts/shared/patient_folds.csv'), on='patient_id', validate='many_to_one')
    assert np.equal(rows.rating, np.floor(rows.rating)).all()
    protein = pd.read_csv(DATA/'train_proteins.csv', usecols=KEYS+['UniProt','NPX'])
    wide = protein.pivot(index=KEYS,columns='UniProt',values='NPX').sort_index()
    features = history_features(rows,clinical,omics,protein)
    rng = np.random.default_rng(SEED)
    for month in rows.visit_month.unique():
        select = rows.visit_month.eq(month)
        truncated = history_features(rows.loc[select].reset_index(drop=True),
                                     clinical[clinical.visit_month.le(month)], omics[omics.visit_month.le(month)],
                                     protein[protein.visit_month.le(month)])
        np.testing.assert_array_equal(features.loc[select].to_numpy(),truncated.to_numpy())
    results, selected = [], []
    for fold in sorted(rows.fold.unique()):
        train, valid = rows.loc[rows.fold.ne(fold)], rows.loc[rows.fold.eq(fold)]
        assert set(train.patient_id).isdisjoint(valid.patient_id)
        patients = set(train.patient_id)
        twide = wide.loc[wide.index.get_level_values('patient_id').isin(patients)]
        ranked = ranking(clinical[clinical.patient_id.isin(patients)],twide)
        top = ranked.head(10).protein.tolist()
        assert len(top) == 10
        selected.append({'fold':int(fold),'top10':top})
        protein_columns = list(dict.fromkeys(SMALL+top))
        z = protein_values(wide,patients,protein_columns)
        # Fit temporal feature scaler on unique training requests (not validation).
        base = features.to_numpy(dtype=float)
        means, scales = base[train.index].mean(0), base[train.index].std(0)
        scales[scales==0] = 1
        base = np.clip((base-means)/scales,-8,8)
        z_all = z.reindex(pd.MultiIndex.from_frame(rows[KEYS])).reset_index(drop=True)
        fixed_z = z[SMALL[:2]]
        baseline_history = VisitHistoryModel(train).predict_rows(valid).to_numpy()
        baseline_fixed = predict_with_proteins(train,valid,fixed_z)
        output = {name: pd.Series(index=valid.index,dtype=float) for name in
                  ['lightgbm','nn_history','nn_small','nn_set','mean_predictions','mean_probabilities']}
        for target in sorted(rows.target.unique()):
            tr = train[train.target.eq(target)]
            va = valid[valid.target.eq(target)]
            y = tr.rating.to_numpy(dtype=int)
            classes = np.unique(y)
            nclasses = int(classes.max()) + 1
            classifier = LGBMClassifier(**PARAMS)
            classifier.fit(features.loc[tr.index], y)
            probs = {'lightgbm': spread_prob(classifier.predict_proba(features.loc[va.index]),classifier.classes_,nclasses)}
            for offset,(name,cols) in enumerate([('nn_history',[]),('nn_small',SMALL),('nn_set',top)]):
                ztr = z_all.loc[tr.index,cols].to_numpy(dtype=np.float32)
                zva = z_all.loc[va.index,cols].to_numpy(dtype=np.float32)
                p = net_predict(name,base[tr.index],base[va.index],ztr,zva,y,classes,SEED+int(fold)*10+offset)
                probs[name] = spread_prob(p,classes,nclasses)
            forecasts = {name:decision(p) for name,p in probs.items()}
            forecasts['mean_predictions'] = np.mean(list(forecasts.values()),axis=0)
            forecasts['mean_probabilities'] = decision(np.mean(list(probs.values()),axis=0))
            for name,pred in forecasts.items(): output[name].loc[va.index] = pred
            np.savez_compressed(OUT/f'fold{fold}_{target}_probabilities.npz',
                                patient_id=va.patient_id.to_numpy(),visit_month=va.visit_month.to_numpy(),
                                horizon=va.horizon.to_numpy(),**probs)
            print(f'fold={fold} {target} complete',flush=True)
        output['history'] = pd.Series(baseline_history,index=valid.index)
        output['fixed_protein'] = pd.Series(baseline_fixed,index=valid.index)
        for name,pred in output.items():
            assert pred.notna().all() and np.isfinite(pred).all() and pred.ge(0).all()
            frame=valid[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
            frame['model'],frame['prediction']=name,pred
            frame['loss']=200*abs(frame.rating-frame.prediction)/(frame.rating+frame.prediction+2)
            results.append(frame)
            print(f'fold={fold} {name}: {frame.loss.mean():.4f}',flush=True)
    full=pd.concat(results,ignore_index=True)
    assert not full.duplicated(['model','patient_id','visit_month','target','horizon']).any()
    assert full.groupby('model').size().eq(len(rows)).all()
    full.to_csv(OUT/'oof_predictions.csv',index=False)
    folds=full.groupby(['model','fold']).loss.mean().unstack()
    scores=full.groupby('model').loss.agg(oof_score='mean',n='size')
    scores['fold_mean'],scores['fold_std']=folds.mean(axis=1),folds.std(axis=1,ddof=0)
    scores.join(folds.add_prefix('fold_')).sort_values('oof_score').to_csv(OUT/'comparison.csv')
    full.groupby(['model','target','horizon']).loss.agg(['mean','size']).to_csv(OUT/'breakdown.csv')
    (OUT/'fold_proteins.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(scores.sort_values('oof_score').to_string(),flush=True)


if __name__ == '__main__': main()

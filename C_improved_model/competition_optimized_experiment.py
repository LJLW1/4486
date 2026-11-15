"""Competition-inspired clinical-visit, supplemental and SMAPE neural ablations.

Inspired by Kaggle's first-place writeup; this is not a reproduction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMClassifier
from fusion_experiment import DATA, ROOT, SEED, PARAMS, decision, spread_prob, history_features
from visit_history_model import KEYS, TARGETS, read_data, training_rows, VisitHistoryModel
from protein_history_experiment import protein_values, predict_with_proteins

OUT = ROOT / 'C_improved_model/competition_optimized_results'


def make_features(rows, clinical, origins, protein):
    out = history_features(rows, clinical, origins, protein)
    visits = pd.concat([clinical[KEYS], origins[KEYS]]).drop_duplicates()
    seen48 = set(visits.loc[visits.visit_month.eq(48), 'patient_id'])
    out['seen_48'] = (rows.visit_month.ge(48) & rows.patient_id.isin(seen48)).to_numpy().astype(float)
    out['seen_6_18_count'] = out.seen_6 + out.seen_18
    return out


class ScoreNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self, x):
        return nn.functional.softplus(self.layers(x).squeeze(1))


def fit_neural(x, y, xv, seed):
    torch.manual_seed(seed)
    net=ScoreNet(x.shape[1])
    opt=torch.optim.AdamW(net.parameters(),lr=.005,weight_decay=.02)
    xt=torch.tensor(x,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    for _ in range(400):
        pred=net(xt)
        loss=(200*abs(yt-pred)/(yt+pred+2)).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite neural loss')
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): return net(torch.tensor(xv,dtype=torch.float32)).numpy()


def main():
    torch.set_num_threads(1); OUT.mkdir(parents=True,exist_ok=True)
    clinical,omics=read_data(DATA)
    supplemental=pd.read_csv(DATA/'supplemental_clinical_data.csv')
    proteins=pd.read_csv(DATA/'train_proteins.csv',usecols=KEYS+['UniProt','NPX'])
    folds=pd.read_csv(ROOT/'artifacts/shared/patient_folds.csv')
    evaluation=training_rows(clinical,omics).merge(folds,on='patient_id',validate='many_to_one')
    full=training_rows(clinical,clinical[KEYS]).merge(folds,on='patient_id',validate='many_to_one')
    empty=pd.DataFrame(columns=KEYS)
    extra=training_rows(supplemental,supplemental[KEYS])
    print('rows evaluation, clinical origins, supplemental:',len(evaluation),len(full),len(extra),flush=True)
    evf=make_features(evaluation,clinical,omics,proteins); evf['supplemental']=0
    fullf=make_features(full,clinical,omics,proteins); fullf['supplemental']=0
    extraf=make_features(extra,supplemental,empty,empty); extraf['supplemental']=1
    # Evaluation folds use all rows of a patient on one side. Supplemental overlap is still excluded per fold.
    for month in evaluation.visit_month.unique():
        mask=evaluation.visit_month.eq(month)
        actual=make_features(evaluation.loc[mask].reset_index(drop=True),
            clinical[clinical.visit_month.le(month)],omics[omics.visit_month.le(month)],
            proteins[proteins.visit_month.le(month)])
        np.testing.assert_array_equal(evf.loc[mask].drop(columns='supplemental').to_numpy(),actual.to_numpy())
    wide=proteins.pivot(index=KEYS,columns='UniProt',values='NPX').sort_index()
    outputs=[]
    for fold in sorted(evaluation.fold.unique()):
        valid=evaluation[evaluation.fold.eq(fold)]; held=set(valid.patient_id)
        for config in ('omics_origins','clinical_origins','clinical_plus_supplemental'):
            src=evaluation if config=='omics_origins' else full
            sf=evf if config=='omics_origins' else fullf
            keep=src.fold.ne(fold)
            train=src.loc[keep].reset_index(drop=True)
            tf=sf.loc[keep].reset_index(drop=True)
            if config=='clinical_plus_supplemental':
                safe=~extra.patient_id.isin(held)
                train=pd.concat([train,extra.loc[safe]],ignore_index=True)
                tf=pd.concat([tf,extraf.loc[safe]],ignore_index=True)
            assert set(train.patient_id).isdisjoint(held)
            xf=evf.loc[valid.index].reset_index(drop=True)
            gbm=np.empty(len(valid));
            for target in TARGETS:
                it=train.target.eq(target).to_numpy(); iv=valid.target.eq(target).to_numpy()
                clf=LGBMClassifier(**PARAMS); clf.fit(tf.loc[it],train.loc[it,'rating'].astype(int))
                prob=spread_prob(clf.predict_proba(xf.loc[iv]),clf.classes_,int(clf.classes_.max())+1)
                gbm[iv]=decision(prob)
            # Shared network across targets, with one-hot UPDRS input, as in the winner's pooled scheme.
            tr_base=tf.to_numpy(dtype=float); va_base=xf.to_numpy(dtype=float)
            means=tr_base.mean(0); std=tr_base.std(0); std[std==0]=1
            tr_base=np.clip((tr_base-means)/std,-8,8); va_base=np.clip((va_base-means)/std,-8,8)
            tr_target=np.column_stack([train.target.eq(t).to_numpy(dtype=float) for t in TARGETS])
            va_target=np.column_stack([valid.target.eq(t).to_numpy(dtype=float) for t in TARGETS])
            net=fit_neural(np.column_stack([tr_base,tr_target]),train.rating.to_numpy(),
                           np.column_stack([va_base,va_target]),SEED+int(fold))
            for model,pred in [('lgbm',gbm),('nn_smape',net),('blend',.5*(gbm+net))]:
                f=valid[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
                f['model']=config+'_'+model; f['prediction']=pred
                f['loss']=200*abs(f.rating-pred)/(f.rating+pred+2)
                assert np.isfinite(pred).all() and (pred>=0).all()
                outputs.append(f)
                print('fold',fold,config,model,round(f.loss.mean(),4),flush=True)
        train=evaluation[evaluation.fold.ne(fold)]
        z=protein_values(wide,set(train.patient_id),['P05060','O15240'])
        for name,pred in [('history',VisitHistoryModel(train).predict_rows(valid).to_numpy()),
                          ('fixed_protein',predict_with_proteins(train,valid,z))]:
            f=valid[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
            f['model']=name; f['prediction']=pred
            f['loss']=200*abs(f.rating-pred)/(f.rating+pred+2); outputs.append(f)
    out=pd.concat(outputs,ignore_index=True)
    assert not out.duplicated(['model','patient_id','visit_month','target','horizon']).any()
    assert out.groupby('model').size().eq(len(evaluation)).all()
    out.to_csv(OUT/'oof_predictions.csv',index=False)
    fs=out.groupby(['model','fold']).loss.mean().unstack()
    score=out.groupby('model').loss.agg(oof_score='mean',n='size')
    score['fold_mean'],score['fold_std']=fs.mean(axis=1),fs.std(axis=1,ddof=0)
    score.join(fs.add_prefix('fold_')).sort_values('oof_score').to_csv(OUT/'comparison.csv')
    out.groupby(['model','target','horizon']).loss.agg(['mean','size']).to_csv(OUT/'breakdown.csv')
    print(score.sort_values('oof_score').to_string(),flush=True)

if __name__=='__main__': main()

"""Patient-fold ablation: condition the existing SMAPE kernel on forecast horizon."""
from pathlib import Path
import numpy as np
import pandas as pd
from visit_history_model import KEYS, read_data, training_rows, VisitHistoryModel
from protein_history_experiment import protein_values, predict_with_proteins
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'amp-parkinsons-disease-progression-prediction'
OUT=ROOT/'C_improved_model/horizon_kernel_results'


def predict(train,valid,z,mode):
    tr=z.reindex(pd.MultiIndex.from_frame(train[KEYS])).to_numpy()
    va=z.reindex(pd.MultiIndex.from_frame(valid[KEYS])).to_numpy()
    y=train.rating.to_numpy(); t=train.target.to_numpy(); h=train.history.to_numpy()
    tm=train.target_month.to_numpy(); horizon=train.horizon.to_numpy()
    out=np.empty(len(valid))
    for (target,month,history,forecast),idx in valid.reset_index(drop=True).groupby(
            ['target','target_month','history','horizon'],sort=False).indices.items():
        target_mask=t==target; same=target_mask & (h==history)
        mask=same if same.any() else target_mask
        truth=y[mask]; candidates=np.arange(int(y[target_mask].max())+1)
        loss=200*abs(truth[:,None]-candidates)/(truth[:,None]+candidates+2)
        weights=np.exp(-.5*((tm[mask]-month)/12)**2)
        if mode=='horizon6': weights*=np.where((horizon[mask]==6)==(forecast==6),1.,.25)
        if mode=='horizon_exact': weights*=np.where(horizon[mask]==forecast,1.,.25)
        for start in range(0,len(idx),64):
            ix=idx[start:start+64]; d=va[ix,None,:]-tr[mask][None,:,:]
            sim=np.where(np.isfinite(d),np.exp(-.5*d*d),1.).mean(2)
            risk=((.5+.5*sim)*weights)@loss
            out[ix]=candidates[risk.argmin(1)]
    return out


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    clinical,omics=read_data(DATA)
    data=training_rows(clinical,omics).merge(pd.read_csv(ROOT/'artifacts/shared/patient_folds.csv'),on='patient_id',validate='many_to_one')
    protein=pd.read_csv(DATA/'train_proteins.csv',usecols=KEYS+['UniProt','NPX'])
    wide=protein.pivot(index=KEYS,columns='UniProt',values='NPX').sort_index()
    out=[]
    for fold in sorted(data.fold.unique()):
        tr=data[data.fold.ne(fold)]; va=data[data.fold.eq(fold)]
        assert set(tr.patient_id).isdisjoint(va.patient_id)
        z=protein_values(wide,set(tr.patient_id),['P05060','O15240'])
        p={'history':VisitHistoryModel(tr).predict_rows(va).to_numpy(),
           'fixed_protein':predict_with_proteins(tr,va,z)}
        for mode in ('horizon6','horizon_exact'): p[mode]=predict(tr,va,z,mode)
        for name,pred in p.items():
            f=va[['patient_id','visit_month','target','target_month','horizon','rating','fold']].copy()
            f['model']=name;f['prediction']=pred;f['loss']=200*abs(f.rating-pred)/(f.rating+pred+2)
            assert np.isfinite(pred).all() and (pred>=0).all()
            out.append(f);print(f'fold={fold} {name}: {f.loss.mean():.4f}',flush=True)
    res=pd.concat(out,ignore_index=True)
    assert not res.duplicated(['model','patient_id','visit_month','target','horizon']).any()
    assert res.groupby('model').size().eq(len(data)).all()
    res.to_csv(OUT/'oof_predictions.csv',index=False)
    fs=res.groupby(['model','fold']).loss.mean().unstack()
    scores=res.groupby('model').loss.agg(oof_score='mean',n='size')
    scores['fold_mean'],scores['fold_std']=fs.mean(axis=1),fs.std(axis=1,ddof=0)
    scores.join(fs.add_prefix('fold_')).to_csv(OUT/'comparison.csv')
    res.groupby(['model','target','horizon']).loss.agg(['mean','size']).to_csv(OUT/'breakdown.csv')
    print(scores.to_string(),flush=True)

if __name__=='__main__': main()

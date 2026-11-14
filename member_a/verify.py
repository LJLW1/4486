"""Check shared hashes, OOF scores, patient isolation and saved A model predictions."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from common.data import load_raw
from common.interface import model_folds
from common.metric import evaluate_oof
from common.prepare import prepare,sha256
from .features import feature_frame,predict_bundle
from .run import EXPERIMENTS,write_json


def verify(data_dir,shared_dir,results_dir,require_models=False):
    data_dir,shared_dir,out=map(Path,[data_dir,shared_dir,results_dir])
    shared=prepare(data_dir,shared_dir)
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['split_id']!=shared['split_id']:
        raise ValueError('Cannot compare different frozen splits')
    samples=pd.read_csv(shared_dir/'samples.csv');patients=pd.read_csv(shared_dir/'patient_folds.csv')
    comparison=pd.read_csv(out/'comparison.csv')
    scores={}
    for name in EXPERIMENTS:
        pred=pd.read_csv(out/name/'oof.csv')
        evaluation=evaluate_oof(samples,pred)
        row=comparison.set_index('experiment').loc[name]
        for field in ['mean_score','std_score','oof_score']:
            if abs(evaluation[field]-float(row[field]))>1e-10:
                raise ValueError(f'Incorrect logged {field} for {name}')
        if len(pred)!=12697 or pred.prediction_id.duplicated().any():
            raise ValueError('Incorrect frozen prediction coverage')
        scores[name]=evaluation
    files=[out/name/f'fold_{fold}.joblib' for name in EXPERIMENTS for fold in range(5)]
    present=all(p.exists() for p in files)
    if require_models and not present:
        raise FileNotFoundError('Run training locally to regenerate the ignored joblib files')
    max_error=0.;model_count=0
    if present:
        raw=load_raw(data_dir)
        with threadpool_limits(limits=4):
            for data in model_folds(raw,samples,patients):
                for name in EXPERIMENTS:
                    bundle=joblib.load(out/name/f'fold_{data.fold}.joblib')
                    if bundle['split_id']!=shared['split_id']:
                        raise ValueError('Saved model split differs')
                    if set(bundle['training_patients'])&set(data.valid_samples.patient_id):
                        raise ValueError('Saved model trained on validation patient')
                    if bundle['pca'] is not None:
                        if set(bundle['pca']['fit_visit_ids'])&set(data.valid_samples.visit_id):
                            raise ValueError('PCA fitted on validation visit')
                    features=feature_frame(data.valid_features,raw,bundle['protein_columns'],bundle['peptide_columns'])
                    pred=predict_bundle(bundle,features,data.valid_features)
                    stored=pd.read_csv(out/name/'oof.csv').set_index('prediction_id').loc[data.valid_samples.prediction_id,'rating'].to_numpy(float)
                    np.testing.assert_allclose(pred,stored,rtol=1e-12,atol=1e-12)
                    max_error=max(max_error,float(np.max(np.abs(pred-stored))));model_count+=1
    result={'status':'passed','split_id':shared['split_id'],'n_requests':len(samples),
            'primary_score':'mean_score','scores':scores,
            'saved_fold_models_checked':model_count,'max_prediction_error':max_error if present else None,
            'model_check_status':'passed' if present else 'not_run_regenerate_ignored_models',
            'frozen_outputs_unchanged':True,
            'sources':{p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(out/'verification.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='amp-parkinsons-disease-progression-prediction')
    parser.add_argument('--shared-dir',default='artifacts/shared')
    parser.add_argument('--results-dir',default='member_a/results')
    parser.add_argument('--require-models',action='store_true')
    a=parser.parse_args();verify(a.data_dir,a.shared_dir,a.results_dir,a.require_models)

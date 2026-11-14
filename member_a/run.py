"""Real Member A ablations using common.model_folds and the frozen shared split.

Run at repository root: python -m member_a.run
Outputs never overwrite an earlier run. Shared logging is an explicit option.
"""
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs4486-member-a-repo-mpl')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from common import SEED, TARGETS
from common.data import load_raw
from common.interface import model_folds
from common.metric import evaluate_oof
from common.prepare import prepare, sha256
from common.experiments import log_experiment
from .features import feature_frame, fit_protein_pca, transform_protein_pca, selected_columns

EXPERIMENTS = ['A0', 'A1', 'A2', 'A3']
DESCRIPTIONS = {
    'A0': 'Known times: visit_month, horizon, target_month',
    'A1': 'A0 + 14 contemporaneous molecular summary features',
    'A2': 'A1 + 22 past-and-current molecular history features',
    'A3': 'A2 + 16 training-fold protein principal components',
}
MODEL = dict(loss='absolute_error', learning_rate=.05, max_iter=120,
             max_leaf_nodes=7, max_depth=3, min_samples_leaf=20,
             l2_regularization=5., early_stopping=False, random_state=SEED)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def log_completed(log, experiment, evaluation, manifest, runtime, code_sha):
    log_experiment(log, experiment_id=f'memberA_{experiment}_omics_v1',
        model='Four target-specific HistGradientBoostingRegressor models',
        features=DESCRIPTIONS[experiment]+'; supplemental=false', evaluation=evaluation,
        seed=SEED, runtime_seconds=runtime, split_id=manifest['split_id'],
        config={'model':MODEL, 'code_sha256':code_sha, 'origin':'frozen common omics visits',
                'prediction_postprocessing':'clip at zero', 'supplemental':False,
                'primary_score':'mean_score', 'pca_training_only':experiment=='A3'},
        notes='AI-assisted Member A controlled feature ablation. Development CV only; no official submission.')


def run(data_dir, shared_dir, output_dir, append_shared=False):
    data_dir, shared_dir, out = map(Path, [data_dir, shared_dir, output_dir])
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite run: {out}')
    manifest = prepare(data_dir, shared_dir)
    raw = load_raw(data_dir)
    samples = pd.read_csv(shared_dir/'samples.csv')
    patients = pd.read_csv(shared_dir/'patient_folds.csv')
    if len(samples) != manifest['n_samples']:
        raise ValueError('Frozen request count differs')
    out.mkdir(parents=True)
    code_sha = {p.name:sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    write_json(out/'plan.json', {'created_utc':datetime.now(timezone.utc).isoformat(),
        'experiments':DESCRIPTIONS, 'model':MODEL, 'seed':SEED, 'split_id':manifest['split_id'],
        'primary_score':'mean_score', 'selection':'Report all four fixed configurations without tuning',
        'expected_requests':len(samples), 'sources':code_sha})
    predictions={name:[] for name in EXPERIMENTS}
    fold_rows=[];runtimes={name:0. for name in EXPERIMENTS};model_checks=[]
    for name in EXPERIMENTS:
        (out/name).mkdir()
    with threadpool_limits(limits=4):
        for data in model_folds(raw, samples, patients):
            fold = data.fold
            protein_cols = [c for c in data.train_features if c.startswith('protein::')]
            peptide_cols = [c for c in data.train_features if c.startswith('peptide::')]
            start = time.monotonic()
            train_x = feature_frame(data.train_features, raw, protein_cols, peptide_cols)
            valid_x = feature_frame(data.valid_features, raw, protein_cols, peptide_cols)
            feature_seconds = time.monotonic()-start
            # This branch consumes no supplemental labels, clinical scores or clinical schedule.
            assert list(data.train_features.prediction_id) == list(data.train_samples.prediction_id)
            assert list(data.valid_features.prediction_id) == list(data.valid_samples.prediction_id)
            pca = None
            for name in EXPERIMENTS:
                start = time.monotonic()
                cols = selected_columns(train_x, name)
                tr, va = train_x[cols].to_numpy(float), valid_x[cols].to_numpy(float)
                if name == 'A3':
                    pca = fit_protein_pca(data.train_features)
                    tr = np.column_stack([tr, transform_protein_pca(data.train_features,pca)])
                    va = np.column_stack([va, transform_protein_pca(data.valid_features,pca)])
                models = {};pred = np.full(len(va), np.nan)
                for target in TARGETS:
                    train_mask = data.train_samples.target.eq(target).to_numpy()
                    valid_mask = data.valid_samples.target.eq(target).to_numpy()
                    model = HistGradientBoostingRegressor(**MODEL)
                    model.fit(tr[train_mask], data.train_samples.loc[train_mask,'rating'].to_numpy(float))
                    pred[valid_mask] = np.maximum(0,model.predict(va[valid_mask]))
                    models[target] = model
                if not np.isfinite(pred).all():
                    raise ValueError('Missing request prediction')
                bundle = dict(models=models, columns=cols, pca=pca if name=='A3' else None,
                    training_patients=sorted(data.train_samples.patient_id.unique().tolist()),
                    protein_columns=protein_cols, peptide_columns=peptide_cols,
                    experiment=name, split_id=manifest['split_id'], model=MODEL)
                joblib.dump(bundle, out/name/f'fold_{fold}.joblib', compress=3)
                result = data.valid_samples[['prediction_id']].assign(rating=pred)
                evaluation = evaluate_oof(data.valid_samples,result)
                predictions[name].append(result)
                elapsed = time.monotonic()-start
                # Shared feature construction is recorded separately, not silently charged to one branch.
                runtimes[name] += elapsed
                row = dict(experiment=name,fold=fold,score=evaluation['oof_score'],
                           requests=len(result),features=tr.shape[1],runtime_seconds=elapsed,
                           shared_feature_seconds=feature_seconds)
                fold_rows.append(row)
                print(json.dumps(row),flush=True)
            model_checks.append({'fold':fold,'valid_patients':int(data.valid_samples.patient_id.nunique()),
                'patient_intersection':len(set(data.train_samples.patient_id)&set(data.valid_samples.patient_id)),
                'validation_request_count':len(data.valid_samples)})
    rows=[];by_target=[]
    for name in EXPERIMENTS:
        oof = pd.concat(predictions[name],ignore_index=True).sort_values('prediction_id').reset_index(drop=True)
        oof.to_csv(out/name/'oof.csv',index=False)
        evaluation = evaluate_oof(samples,oof)
        write_json(out/name/'evaluation.json',evaluation)
        for target in TARGETS:
            subset=samples.loc[samples.target==target]
            pred=oof.loc[oof.prediction_id.isin(subset.prediction_id)]
            values=evaluate_oof(subset,pred)
            by_target.append({'experiment':name,'target':target,'oof_score':values['oof_score'],'requests':len(subset)})
        for horizon in sorted(samples.horizon.unique()):
            subset=samples.loc[samples.horizon==horizon];pred=oof.loc[oof.prediction_id.isin(subset.prediction_id)]
            values=evaluate_oof(subset,pred)
            by_target.append({'experiment':name,'target':f'horizon_{horizon}','oof_score':values['oof_score'],'requests':len(subset)})
        row={'experiment':name,'description':DESCRIPTIONS[name],
             'mean_score':evaluation['mean_score'],'std_score':evaluation['std_score'],
             'oof_score':evaluation['oof_score'],'runtime_seconds':runtimes[name],
             **{f'fold_{k}':v for k,v in evaluation['fold_scores'].items()}}
        rows.append(row)
        log_completed(out/'experiment_results.csv',name,evaluation,manifest,runtimes[name],code_sha)
        if append_shared:
            log_completed(shared_dir/'experiment_results.csv',name,evaluation,manifest,runtimes[name],code_sha)
    comparison=pd.DataFrame(rows)
    comparison['delta_vs_A0']=comparison.mean_score-comparison.mean_score.iloc[0]
    comparison.to_csv(out/'comparison.csv',index=False)
    pd.DataFrame(fold_rows).to_csv(out/'fold_scores.csv',index=False)
    pd.DataFrame(by_target).to_csv(out/'target_horizon_scores.csv',index=False)
    write_json(out/'manifest.json',{'split_id':manifest['split_id'],'seed':SEED,'n_requests':len(samples),
        'primary_score':'mean_score','shared_frozen_hashes':manifest['output_sha256'],
        'shared_checks':model_checks,'sources':code_sha,'model':MODEL,
        'environment':{'python':platform.python_version(),'machine':platform.machine(),
                       **{p:version(p) for p in ['numpy','pandas','scikit-learn','filelock','matplotlib','scipy','joblib','threadpoolctl']}},
        'runtime_definition':'Per-configuration measured fit, prediction and model-save time; common interface and feature construction excluded and recorded per fold',
        'official_submission':False,'old_seed42_scores_included':False})
    print(comparison[['experiment','mean_score','std_score','oof_score','delta_vs_A0']].to_string(index=False),flush=True)
    return comparison


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',default='amp-parkinsons-disease-progression-prediction')
    parser.add_argument('--shared-dir',default='artifacts/shared')
    parser.add_argument('--output-dir',default='member_a/results')
    parser.add_argument('--append-shared-log',action='store_true')
    args=parser.parse_args()
    run(args.data_dir,args.shared_dir,args.output_dir,args.append_shared_log)


if __name__=='__main__':main()

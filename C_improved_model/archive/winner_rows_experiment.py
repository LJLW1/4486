"""Compare winner-inspired training-row construction with a fixed classifier."""
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from visit_history_model import KEYS, TARGETS, read_data, training_rows
from fusion_experiment import PARAMS, decision, spread_prob
from competition_optimized_experiment import make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'amp-parkinsons-disease-progression-prediction'
OUT = ROOT / 'C_improved_model/top5_results'
MONTHS = [0, 6, 12, 18, 24, 36, 48, 60, 72, 84]


def main():
    clinical, omics = read_data(DATA)
    supplemental = pd.read_csv(DATA / 'supplemental_clinical_data.csv')
    supplemental.loc[supplemental.visit_month.eq(5), 'visit_month'] = 6
    protein = pd.read_csv(DATA / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
    folds = pd.read_csv(ROOT / 'artifacts/shared/patient_folds.csv')
    evaluation = training_rows(clinical, omics).merge(folds, on='patient_id', validate='many_to_one')
    full = training_rows(clinical, clinical[KEYS]).merge(folds, on='patient_id', validate='many_to_one')
    full = full[full.visit_month.isin(MONTHS)].reset_index(drop=True)
    extra = training_rows(supplemental, supplemental[KEYS])
    extra = extra[extra.visit_month.isin(MONTHS)].reset_index(drop=True)
    empty = pd.DataFrame(columns=KEYS)
    evf = make_features(evaluation, clinical, omics, protein).assign(supplemental=0)
    fullf = make_features(full, clinical, clinical[KEYS], protein).assign(supplemental=0)
    extraf = make_features(extra, supplemental, empty, empty).assign(supplemental=1)
    results = []
    for fold in range(5):
        valid = evaluation[evaluation.fold.eq(fold)]
        held = set(valid.patient_id)
        base = evaluation.fold.ne(fold)
        mask = full.fold.ne(fold)
        safe = ~extra.patient_id.isin(held)
        configs = {
            'omics': (evaluation.loc[base], evf.loc[base]),
            'winner_origins': (full.loc[mask], fullf.loc[mask]),
            'winner_origins_supplemental': (pd.concat([full.loc[mask], extra.loc[safe]]), pd.concat([fullf.loc[mask], extraf.loc[safe]])),
        }
        for name, (train, features) in configs.items():
            assert set(train.patient_id).isdisjoint(held)
            pred = np.empty(len(valid))
            valid_features = evf.loc[valid.index].reset_index(drop=True)
            features = features.reset_index(drop=True)
            train = train.reset_index(drop=True)
            for target in TARGETS:
                tr = train.target.eq(target).to_numpy()
                va = valid.target.eq(target).to_numpy()
                clf = LGBMClassifier(**PARAMS)
                clf.fit(features.loc[tr], train.loc[tr, 'rating'].astype(int))
                probabilities = spread_prob(clf.predict_proba(valid_features.loc[va]), clf.classes_, int(clf.classes_.max()) + 1)
                pred[va] = decision(probabilities)
            part = valid[['patient_id', 'visit_month', 'target', 'horizon', 'rating', 'fold']].copy()
            part['model'], part['prediction'] = name, pred
            part['loss'] = 200 * abs(part.rating - pred) / (part.rating + pred + 2)
            results.append(part)
            print(fold, name, len(train), round(part.loss.mean(), 4), flush=True)
    result = pd.concat(results, ignore_index=True)
    assert not result.duplicated(['model', 'patient_id', 'visit_month', 'target', 'horizon']).any()
    result.to_csv(OUT / 'winner_rows_oof.csv', index=False)
    result.groupby('model').loss.agg(['mean', 'size']).to_csv(OUT / 'winner_rows_summary.csv')


if __name__ == '__main__':
    main()

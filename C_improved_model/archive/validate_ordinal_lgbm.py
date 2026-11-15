"""Minimal ordinal-LightGBM ablation against original multiclass classifier.

Default: one frozen patient fold and one UPDRS target (8 binary trees plus
one multiclass tree). Expand ONLY after inspecting the pilot; no grid search.
This is offline development CV, not a Kaggle submission.

From repository root:
    python C_improved_model/validate_ordinal_lgbm.py
    python C_improved_model/validate_ordinal_lgbm.py --folds 0 1 2 3 4 --targets updrs_1 updrs_2 updrs_3 updrs_4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from fusion_experiment import PARAMS, decision, spread_prob
from visit_history_model import KEYS, TARGETS, read_data, training_rows
from competition_optimized_experiment import make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "amp-parkinsons-disease-progression-prediction"
OUTPUT = ROOT / "C_improved_model/ordinal_results"


def smape(y, prediction):
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return 200 * np.abs(y - prediction) / (y + prediction + 2)


def ordinal_thresholds(y, max_thresholds=8):
    """Select thresholds from training labels ONLY; no validation data."""
    distinct = np.unique(np.asarray(y, dtype=int))
    if len(distinct) < 2:
        raise ValueError("Ordinal classification requires at least two observed values")
    quantiles = np.linspace(0, 1, max_thresholds + 2)[1:-1]
    suggested = np.quantile(y, quantiles)
    thresholds = sorted(set(int(v) for v in suggested if v < distinct[-1]))
    # Every threshold must split into two observed label classes.
    return np.array([v for v in thresholds if (y <= v).any() and (y > v).any()], dtype=int)


def train_ordinal(x, y, thresholds):
    """Fit P(Y > threshold | X) with the same tree hyperparameters."""
    models = []
    binary_params = dict(PARAMS)
    binary_params["objective"] = "binary"
    for boundary in thresholds:
        label = (y > boundary).astype(int)
        if len(np.unique(label)) != 2:
            raise AssertionError("Threshold has constant labels")
        model = LGBMClassifier(**binary_params)
        model.fit(x, label)
        models.append(model)
    return models


def ordinal_probabilities(models, thresholds, y_train, x_valid):
    """Recover ordered-bin probabilities, then training-only within-bin PMFs.

    Cumulative binary models can cross; enforce P(Y>k) nonincreasing.
    Within-bin frequencies are training-label empirical probabilities (Laplace
    smoothed); no outer validation labels are involved in reconstruction.
    """
    if not models:
        raise ValueError("No ordinal models")
    max_label = int(np.max(y_train))
    survival = np.column_stack([m.predict_proba(x_valid)[:, 1] for m in models])
    survival = np.minimum.accumulate(np.clip(survival, 0, 1), axis=1)
    n = len(x_valid)
    bin_mass = np.column_stack([
        1 - survival[:, 0],
        survival[:, :-1] - survival[:, 1:] if len(models) > 1
        else np.empty((n, 0), dtype=float),
        survival[:, -1],
    ])
    bin_mass = np.maximum(bin_mass, 0)
    output = np.zeros((n, max_label + 1), dtype=float)
    boundaries = [-1] + list(thresholds) + [max_label]
    counts = np.bincount(y_train, minlength=max_label + 1).astype(float)
    for j, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        values = np.arange(left + 1, right + 1, dtype=int)
        # Positive smoothing prevents unseen integers receiving zero probability.
        local = counts[values] + 0.1
        local /= local.sum()
        output[:, values] += bin_mass[:, [j]] * local[None, :]
    if not np.allclose(output.sum(axis=1), 1, atol=1e-6):
        raise AssertionError("Ordinal probabilities do not sum to one")
    return output


def run(folds=(0,), targets=("updrs_1",), data_dir=DATA, output_dir=OUTPUT, max_thresholds=8):
    if not folds or not targets or max_thresholds < 1:
        raise ValueError("Empty folds/targets or invalid threshold count")
    if any(f not in range(5) for f in folds) or any(t not in TARGETS for t in targets):
        raise ValueError("Unknown fold or UPDRS target")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    clinical, omics = read_data(data_dir)
    proteins = pd.read_csv(data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    fold_map = pd.read_csv(ROOT / "artifacts/shared/patient_folds.csv")
    rows = training_rows(clinical, omics).merge(
        fold_map, on="patient_id", validate="many_to_one"
    )
    features = make_features(rows, clinical, omics, proteins).assign(supplemental=0)
    if len(rows) != len(features):
        raise AssertionError("Row/feature mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries, records = [], []
    for fold in folds:
        valid_patients = set(fold_map.loc[fold_map.fold.eq(fold), "patient_id"])
        for target in targets:
            train_mask = rows.fold.ne(fold) & rows.target.eq(target)
            valid_mask = rows.fold.eq(fold) & rows.target.eq(target)
            train, valid = rows.loc[train_mask], rows.loc[valid_mask]
            if not len(train) or not len(valid):
                raise ValueError("Empty train or validation target")
            if set(train.patient_id) & valid_patients:
                raise AssertionError("Patient leakage")
            x_train, x_valid = features.loc[train_mask], features.loc[valid_mask]
            y = train.rating.to_numpy(dtype=int)
            reference = LGBMClassifier(**PARAMS)
            reference.fit(x_train, y)
            baseline_prob = spread_prob(
                reference.predict_proba(x_valid), reference.classes_,
                int(reference.classes_.max()) + 1,
            )
            baseline = decision(baseline_prob)
            thresholds = ordinal_thresholds(y, max_thresholds)
            ordinal_models = train_ordinal(x_train, y, thresholds)
            ordinal_prob = ordinal_probabilities(ordinal_models, thresholds, y, x_valid)
            prediction = decision(ordinal_prob)
            truth = valid.rating.to_numpy(float)
            frame = valid[["patient_id", "visit_month", "target", "horizon",
                           "rating", "fold"]].copy()
            frame["prediction_multiclass"] = baseline
            frame["prediction_ordinal"] = prediction
            frame["loss_multiclass"] = smape(truth, baseline)
            frame["loss_ordinal"] = smape(truth, prediction)
            frame["delta_loss"] = frame.loss_ordinal - frame.loss_multiclass
            entries.append(frame)
            record = {
                "fold": int(fold), "target": target, "n_valid": len(valid),
                "n_train": len(train), "n_binary_models": len(ordinal_models),
                "thresholds": json.dumps(thresholds.tolist()),
                "multiclass_smape1": float(frame.loss_multiclass.mean()),
                "ordinal_smape1": float(frame.loss_ordinal.mean()),
                "delta_vs_multiclass": float(frame.delta_loss.mean()),
                "prediction_flip_rate": float((baseline != prediction).mean()),
            }
            records.append(record)
            # Persist after every fold-target to survive interruption.
            pd.concat(entries, ignore_index=True).to_csv(
                output_dir / "ordinal_predictions.csv", index=False
            )
            pd.DataFrame(records).to_csv(output_dir / "ordinal_comparison.csv", index=False)
            print(
                f"Fold {fold} {target}: multiclass={record['multiclass_smape1']:.4f}, "
                f"ordinal={record['ordinal_smape1']:.4f}, "
                f"delta={record['delta_vs_multiclass']:+.4f}; "
                f"{len(ordinal_models)} thresholds", flush=True
            )
    result = pd.DataFrame(records)
    # When all five folds of a given target are present, mean is equally weighted
    # across folds. Pilot results are NOT a five-fold overall score.
    summary = result.groupby("target").agg(
        evaluated_folds=("fold", "nunique"),
        mean_fold_multiclass=("multiclass_smape1", "mean"),
        mean_fold_ordinal=("ordinal_smape1", "mean"),
        mean_delta=("delta_vs_multiclass", "mean"),
        fold_win_rate=("delta_vs_multiclass", lambda z: (z < 0).mean()),
    )
    summary.to_csv(output_dir / "ordinal_summary.csv")
    (output_dir / "ordinal_metadata.json").write_text(json.dumps({
        "experiment": "Ordinal threshold LightGBM vs multiclass",
        "folds": list(folds), "targets": list(targets),
        "max_thresholds": max_thresholds,
        "training": "omics origins, uniform sample weights, original features",
        "caution": "Pilot or development CV only; not an independent Kaggle score. "
                   "Original features may rely on historical clinical visits "
                   "unavailable in online iteration.",
        "ai_usage": "Implementation assisted by ChatGPT; independently review before submission.",
    }, indent=2) + "\n", encoding="utf-8")
    print(summary.to_string(), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--targets", choices=TARGETS, nargs="+", default=["updrs_1"])
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-thresholds", type=int, default=8)
    args = parser.parse_args()
    run(**vars(args))

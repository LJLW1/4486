"""Frozen five-fold validation of ordinal-regression trees blended into V18.

Train the original V18 neural network once per fold; replace/supplement only its
LightGBM multiclass tree component using continuous-score LGBM regressors.
Candidates share frozen patient splits, training labels, and neural predictions.
Lower SMAPE+1 is better. This is NOT a Kaggle submission.

python C_improved_model/validate_tree_regression.py
python C_improved_model/validate_tree_regression.py --epochs 2  # smoke test only
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from validate_residual_scale import (
    ROOT, ArchitectureModel, KEYS, load_frozen_validation, load_training,
    protein_matrix, evaluate_oof, smape_plus_one,
)
from teammate_aligned_model import ALPHA, TREE_INDICES

TREE_OBJECTIVES = ("regression_l1", "huber", "poisson")
TREE_WEIGHTS = (0.25, 0.5, 0.75, 1.0)


def tree_classifier_prediction(classifier, tree_x):
    """Original V18 minimum expected SMAPE+1 class decision."""
    probabilities = classifier.predict_proba(tree_x)
    classes = classifier.classes_.astype(int)
    candidates = np.arange(int(classes.max()) + 1)
    risk = 200 * abs(classes[:, None] - candidates) / (classes[:, None] + candidates + 2)
    return candidates[(probabilities @ risk).argmin(axis=1)].astype(float)


def tree_regressor(objective):
    """Comparable-capacity continuous-target tree with same regularization."""
    if objective not in TREE_OBJECTIVES:
        raise ValueError(f"Unexpected tree objective: {objective}")
    return LGBMRegressor(
        objective=objective,
        n_estimators=300,
        num_leaves=7,
        learning_rate=0.03,
        min_child_samples=50,
        reg_lambda=10,
        random_state=4486,
        n_jobs=1,
        verbosity=-1,
    )


def blended_prediction(v18_pred, original_tree, new_tree, alpha, tree_weight):
    """V18 + (1-alpha) * beta * (alternative_tree-original_tree)."""
    if not 0 <= tree_weight <= 1:
        raise ValueError("tree_weight must be in [0,1]")
    if len(v18_pred) != len(original_tree) or len(v18_pred) != len(new_tree):
        raise ValueError("Mismatched prediction lengths")
    return np.maximum(0, v18_pred + (1 - alpha) * tree_weight * (new_tree - original_tree))


def run(data_dir=None, output_dir=None, epochs=200, weights=TREE_WEIGHTS):
    if epochs < 1 or not weights or not all(0 < w <= 1 for w in weights):
        raise ValueError("Invalid epochs or tree weights")
    data_dir = Path(data_dir) if data_dir else ROOT / "amp-parkinsons-disease-progression-prediction"
    output_dir = Path(output_dir) if output_dir else ROOT / "C_improved_model" / "tree_regression_results"

    samples, folds, manifest, features, proteins, targets, heads = load_frozen_validation(ROOT, data_dir)
    train_rows, train_features = load_training(data_dir)
    protein_table = pd.read_csv(data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    train_proteins = protein_matrix(train_rows, protein_table)
    train_proteins[train_rows.source.eq(1).to_numpy()] = np.nan
    names = ["v18"] + [
        f"{objective}_w{str(weight).replace('.', 'p')}"
        for objective in TREE_OBJECTIVES for weight in weights
    ]
    oof = {name: np.full(len(samples), np.nan) for name in names}
    records = []

    for fold in range(5):
        heldout = set(folds.loc[folds.fold.eq(fold), "patient_id"])
        is_train = ~train_rows.patient_id.isin(heldout).to_numpy()
        is_valid = samples.fold.eq(fold).to_numpy()
        if not heldout or not is_valid.any():
            raise ValueError(f"Empty validation fold {fold}")
        if set(train_rows.loc[is_train, "patient_id"]) & heldout:
            raise AssertionError("Patient leakage")
        print(f"Fold {fold}: {is_train.sum()} train requests, {is_valid.sum()} held-out requests", flush=True)
        training_rows = train_rows.loc[is_train].reset_index(drop=True)
        model = ArchitectureModel("residual").fit(
            training_rows, train_features[is_train], train_proteins[is_train], epochs=epochs
        )
        x_train = np.column_stack([
            train_features[is_train][:, TREE_INDICES],
            training_rows.target_index.to_numpy(),
        ])
        x_valid = np.column_stack([features[is_valid][:, TREE_INDICES], targets[is_valid]])
        baseline = np.asarray(
            model.predict_features(features[is_valid], targets[is_valid], heads[is_valid],
                                   proteins[is_valid]), dtype=float
        )
        original_tree = tree_classifier_prediction(model.classifier, x_valid)
        predictions = {"v18": baseline}
        for objective in TREE_OBJECTIVES:
            regressor = tree_regressor(objective)
            regressor.fit(x_train, training_rows.rating.to_numpy(float))
            candidate = np.maximum(0, regressor.predict(x_valid))
            if not np.isfinite(candidate).all():
                raise ValueError(f"Nonfinite {objective} prediction")
            for weight in weights:
                name = f"{objective}_w{str(weight).replace('.', 'p')}"
                predictions[name] = blended_prediction(
                    baseline, original_tree, candidate,
                    ALPHA[targets[is_valid]], weight,
                )
        true = samples.loc[is_valid, "rating"].to_numpy(float)
        for name, pred in predictions.items():
            if pred.shape != true.shape or not np.isfinite(pred).all():
                raise ValueError(f"Invalid predictions {name} fold {fold}")
            oof[name][is_valid] = pred
            records.append({
                "fold": fold, "variant": name,
                "smape1": smape_plus_one(true, pred),
                "requests": len(true),
            })
        print("  fold best:", min(records[-len(names):], key=lambda r:r["smape1"]), flush=True)
        del model

    summary = []
    for name in names:
        if not np.isfinite(oof[name]).all():
            raise AssertionError(f"Incomplete predictions {name}")
        checked = evaluate_oof(samples, pd.DataFrame({
            "prediction_id": samples.prediction_id, "rating": oof[name],
        }))
        summary.append({
            "variant": name,
            "mean_fold_smape1": checked["mean_score"],
            "std_fold_smape1": checked["std_score"],
            "pooled_oof_smape1": checked["oof_score"],
            "n_predictions": checked["n_predictions"],
        })
    summary = pd.DataFrame(summary)
    baseline_score = summary.loc[summary.variant.eq("v18"), "mean_fold_smape1"].iloc[0]
    summary["delta_vs_v18"] = summary.mean_fold_smape1 - baseline_score
    summary = summary.sort_values("mean_fold_smape1", ignore_index=True)
    by_fold = pd.DataFrame(records)
    ref_fold = by_fold[by_fold.variant.eq("v18")].set_index("fold").smape1
    by_fold["delta_vs_v18"] = by_fold.smape1 - by_fold.fold.map(ref_fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_fold.to_csv(output_dir / "folds.csv", index=False)
    oof_frame = samples[["prediction_id", "patient_id", "fold", "rating"]].copy()
    for name in names:
        oof_frame["pred_" + name] = oof[name]
    oof_frame.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({
        "experiment": "V18 tree objective regression and blend search",
        "protocol": manifest["protocol"], "split_id": manifest["split_id"],
        "epochs": epochs, "neural_seeds": [4486, 4487, 4488],
        "objectives": list(TREE_OBJECTIVES), "tree_weights": list(weights),
        "limitation": "Hyperparameter comparisons on the same five folds are exploratory; no independent test or Kaggle score.",
        "ai_usage": "Experiment code assisted by ChatGPT; human review required",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Saved results to", output_dir, flush=True)
    return summary, by_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    run(**vars(args))

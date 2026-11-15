"""Fourth-round V18 target/horizon conditional fusion ablation.

Each outer fold uses its frozen held-out patients only for final evaluation.
A deterministic patient-disjoint inner calibration split is made from that
outer fold's training patients. The inner model calibrates fusion weights;
the final model is retrained on the full outer training cohort.

Outputs are exploratory five-fold CV, NOT a Kaggle hidden score.

Run: python C_improved_model/validate_conditional_fusion.py
Smoke: python C_improved_model/validate_conditional_fusion.py --epochs 2
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from validate_residual_scale import (
    ROOT, ArchitectureModel, KEYS, evaluate_oof, load_frozen_validation,
    load_training, protein_matrix, smape_plus_one,
)
from teammate_aligned_model import ALPHA, HORIZONS, NN_INDICES, TARGETS, TREE_INDICES


def components(model, features, targets, heads, proteins):
    """Return V18 constituent predictions with matching head/target indexing."""
    tree_x = np.column_stack([features[:, TREE_INDICES], targets])
    probabilities = model.classifier.predict_proba(tree_x)
    classes = model.classifier.classes_.astype(int)
    choices = np.arange(int(classes.max()) + 1)
    risk = 200 * abs(classes[:, None] - choices) / (classes[:, None] + choices + 2)
    tree = choices[(probabilities @ risk).argmin(axis=1)].astype(float)
    x = torch.tensor(model.neural_features(features[:, NN_INDICES], proteins), dtype=torch.float32)
    with torch.no_grad():
        neural_matrix = np.mean([net(x).numpy() for net in model.networks], axis=0)
    neural = neural_matrix[np.arange(len(features)), heads].astype(float)
    if not (np.isfinite(tree).all() and np.isfinite(neural).all()):
        raise ValueError("Nonfinite component predictions")
    return tree, neural


def blend(tree, neural, targets, heads, weights):
    """weights shape (16,), in horizon-major order."""
    alpha = np.asarray(weights)[heads]
    return np.maximum(0, (1 - alpha) * tree + alpha * neural)


def calibrate_weights(truth, tree, neural, heads, *,
                      grid=(0, .125, .25, .375, .5, .625, .75, .875, 1),
                      shrinkage=100, min_samples=30):
    """Minimize shrunk, head-specific SMAPE on *inner-held-out* patients.

    Regularization is a penalty in average loss units, pulling weights toward
    V18 ALPHA for each target. No outer-held-out labels are ever used here.
    """
    weights = np.tile(np.asarray(ALPHA, dtype=float), len(HORIZONS))
    counts = []
    for h in range(len(weights)):
        mask = heads == h
        n = int(mask.sum())
        counts.append(n)
        if n < min_samples:
            continue
        y, t, z = truth[mask], tree[mask], neural[mask]
        base = weights[h]
        scores = []
        for candidate in grid:
            pred = np.maximum(0, (1 - candidate) * t + candidate * z)
            score = smape_plus_one(y, pred)
            # Shrink more aggressively when the sample count is small.
            score += shrinkage / n * (candidate - base) ** 2
            scores.append(score)
        weights[h] = float(grid[int(np.argmin(scores))])
    return weights, counts


def run(data_dir=None, output_dir=None, epochs=200, calibration_fraction=.25):
    if epochs < 1 or not 0 < calibration_fraction < .5:
        raise ValueError("Invalid epochs or calibration fraction")
    data_dir = Path(data_dir) if data_dir else ROOT / "amp-parkinsons-disease-progression-prediction"
    output_dir = Path(output_dir) if output_dir else ROOT / "C_improved_model" / "conditional_fusion_results"
    samples, folds, manifest, features, proteins, targets, heads = load_frozen_validation(ROOT, data_dir)
    rows, training_features = load_training(data_dir)
    table = pd.read_csv(data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    training_proteins = protein_matrix(rows, table)
    training_proteins[rows.source.eq(1).to_numpy()] = np.nan
    variants = ("v18", "conditional")
    oof = {name: np.full(len(samples), np.nan) for name in variants}
    records, diagnostics, settings = [], [], []
    fixed = np.tile(np.asarray(ALPHA, dtype=float), len(HORIZONS))

    for fold in range(5):
        heldout = set(folds.loc[folds.fold.eq(fold), "patient_id"])
        outer_train = ~rows.patient_id.isin(heldout).to_numpy()
        outer_valid = samples.fold.eq(fold).to_numpy()
        # Select calibration patients from main cohort only. Supplemental
        # clinical rows from those patients are excluded from inner training.
        train_patients = np.sort(rows.loc[outer_train & rows.source.eq(0).to_numpy(),
                                          "patient_id"].unique())
        fit_ids, calibration_ids = train_test_split(
            train_patients, test_size=calibration_fraction,
            random_state=4486 + fold, shuffle=True,
        )
        fit_ids, calibration_ids = set(fit_ids), set(calibration_ids)
        if heldout & (fit_ids | calibration_ids) or fit_ids & calibration_ids:
            raise AssertionError("Patient leakage in nested calibration")
        inner_train = outer_train & ~rows.patient_id.isin(calibration_ids).to_numpy()
        inner_valid = outer_train & rows.patient_id.isin(calibration_ids).to_numpy() & rows.source.eq(0).to_numpy()
        if not inner_train.any() or not inner_valid.any():
            raise ValueError("Empty inner split")
        print(f"Fold {fold}: inner fitting {inner_train.sum()}, calibration {inner_valid.sum()}", flush=True)
        inner = ArchitectureModel("residual").fit(
            rows.loc[inner_train].reset_index(drop=True),
            training_features[inner_train], training_proteins[inner_train], epochs=epochs,
        )
        cal_targets = rows.loc[inner_valid, "target_index"].to_numpy(int)
        cal_heads = rows.loc[inner_valid, "head"].to_numpy(int)
        cal_truth = rows.loc[inner_valid, "rating"].to_numpy(float)
        cal_tree, cal_neural = components(inner, training_features[inner_valid],
                                           cal_targets, cal_heads, training_proteins[inner_valid])
        weights, counts = calibrate_weights(cal_truth, cal_tree, cal_neural, cal_heads)
        del inner
        settings.append({"fold": fold, "calibration_patients": len(calibration_ids),
                         "calibration_requests": len(cal_truth),
                         "weights": weights.tolist(), "head_counts": counts})
        print(f"Fold {fold}: final fitting {outer_train.sum()}", flush=True)
        model = ArchitectureModel("residual").fit(
            rows.loc[outer_train].reset_index(drop=True),
            training_features[outer_train], training_proteins[outer_train], epochs=epochs,
        )
        valid_targets = targets[outer_valid]
        valid_heads = heads[outer_valid]
        tree, neural = components(model, features[outer_valid],
                                  valid_targets, valid_heads, proteins[outer_valid])
        baseline = model.predict_features(features[outer_valid],
                                           valid_targets, valid_heads, proteins[outer_valid])
        check = blend(tree, neural, valid_targets, valid_heads, fixed)
        if not np.allclose(baseline, check, atol=1e-4, rtol=1e-5):
            raise AssertionError("Component reconstruction differs from V18")
        predictions = {"v18": check, "conditional": blend(tree, neural, valid_targets, valid_heads, weights)}
        truth = samples.loc[outer_valid, "rating"].to_numpy(float)
        for name, prediction in predictions.items():
            oof[name][outer_valid] = prediction
            records.append({"fold": fold, "variant": name,
                            "smape1": smape_plus_one(truth, prediction),
                            "requests": len(truth)})
        for head in range(16):
            mask = valid_heads == head
            if mask.any():
                diagnostics.append({
                    "fold": fold, "head": head,
                    "target": TARGETS[head % 4], "horizon": HORIZONS[head // 4],
                    "requests": int(mask.sum()),
                    "tree_smape1": smape_plus_one(truth[mask], tree[mask]),
                    "neural_smape1": smape_plus_one(truth[mask], neural[mask]),
                    "v18_smape1": smape_plus_one(truth[mask], check[mask]),
                    "conditional_smape1": smape_plus_one(truth[mask], predictions["conditional"][mask]),
                    "chosen_alpha": float(weights[head]),
                })
        del model

    summary = []
    for name in variants:
        if not np.isfinite(oof[name]).all():
            raise AssertionError("Incomplete outer OOF predictions")
        result = evaluate_oof(samples, pd.DataFrame({
            "prediction_id": samples.prediction_id, "rating": oof[name],
        }))
        summary.append({"variant": name, "mean_fold_smape1": result["mean_score"],
                        "std_fold_smape1": result["std_score"],
                        "pooled_oof_smape1": result["oof_score"],
                        "n_predictions": result["n_predictions"]})
    summary = pd.DataFrame(summary)
    base_score = summary.loc[summary.variant.eq("v18"), "mean_fold_smape1"].iloc[0]
    summary["delta_vs_v18"] = summary.mean_fold_smape1 - base_score
    by_fold = pd.DataFrame(records)
    base_folds = by_fold.loc[by_fold.variant.eq("v18")].set_index("fold").smape1
    by_fold["delta_vs_v18"] = by_fold.smape1 - by_fold.fold.map(base_folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_fold.to_csv(output_dir / "folds.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "head_diagnostics.csv", index=False)
    output = samples[["prediction_id", "patient_id", "fold", "rating"]].copy()
    for name in variants:
        output["pred_" + name] = oof[name]
    output.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({
        "experiment": "Fourth round: inner patient-held-out conditional target/horizon weights",
        "split_id": manifest["split_id"], "protocol": manifest["protocol"],
        "epochs": epochs, "calibration_fraction": calibration_fraction,
        "calibration_seed": "4486 + outer_fold", "weight_grid": [0, .125, .25, .375, .5, .625, .75, .875, 1],
        "weight_shrinkage": 100, "min_calibration_head_samples": 30,
        "per_fold": settings,
        "limitation": "Five folds reused for model development, exploratory; not a Kaggle score.",
        "ai_usage": "Experiment implementation assisted by ChatGPT; independent review required.",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Saved to", output_dir, flush=True)
    return summary, by_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--calibration-fraction", type=float, default=.25)
    run(**vars(parser.parse_args()))

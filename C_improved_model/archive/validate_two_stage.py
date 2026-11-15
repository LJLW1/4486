"""Compare joint V18 against two-stage residual training on frozen patient folds.

Both variants use the same 200 total epochs, seeds, features, LightGBM and
patient splits. Two-stage uses 100 epochs base-only + 100 correction-only.
This is offline CV, NOT a Kaggle leaderboard score.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from validate_residual_scale import (
    ROOT, load_frozen_validation, ArchitectureModel, load_training,
    protein_matrix, KEYS, evaluate_oof, smape_plus_one,
)

VARIANTS = ("residual", "residual_two_stage")


def run(data_dir=None, output_dir=None, epochs=200, base_epochs=100):
    if epochs < 2 or not 0 < base_epochs < epochs:
        raise ValueError("Require 0 < base_epochs < epochs")
    data_dir = Path(data_dir) if data_dir else ROOT / "amp-parkinsons-disease-progression-prediction"
    output_dir = Path(output_dir) if output_dir else ROOT / "C_improved_model" / "two_stage_results"
    samples, folds, manifest, features, proteins, targets, heads = load_frozen_validation(ROOT, data_dir)
    train_rows, train_features = load_training(data_dir)
    protein_table = pd.read_csv(data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    train_proteins = protein_matrix(train_rows, protein_table)
    train_proteins[train_rows.source.eq(1).to_numpy()] = np.nan
    predictions = {variant: np.full(len(samples), np.nan) for variant in VARIANTS}
    records = []
    for fold in range(5):
        heldout = set(folds.loc[folds.fold.eq(fold), "patient_id"])
        train_mask = ~train_rows.patient_id.isin(heldout).to_numpy()
        valid_mask = samples.fold.eq(fold).to_numpy()
        truth = samples.loc[valid_mask, "rating"].to_numpy(float)
        print(f"Fold {fold}: {train_mask.sum()} train, {valid_mask.sum()} validation requests", flush=True)
        for variant in VARIANTS:
            model = ArchitectureModel(variant).fit(
                train_rows.loc[train_mask].reset_index(drop=True),
                train_features[train_mask],
                train_proteins[train_mask],
                epochs=epochs,
                base_epochs=base_epochs,
            )
            pred = np.asarray(model.predict_features(
                features[valid_mask], targets[valid_mask], heads[valid_mask],
                proteins[valid_mask]), dtype=float)
            if pred.shape != truth.shape or not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError(f"Invalid predictions: fold={fold}, variant={variant}")
            predictions[variant][valid_mask] = pred
            records.append(dict(fold=fold, variant=variant,
                                smape1=smape_plus_one(truth, pred),
                                requests=len(truth)))
            print(f"  {variant}: {records[-1]['smape1']:.6f}", flush=True)
            del model
    summary = []
    for variant in VARIANTS:
        checked = evaluate_oof(samples, pd.DataFrame({
            "prediction_id": samples.prediction_id,
            "rating": predictions[variant],
        }))
        summary.append(dict(variant=variant, mean_fold_smape1=checked["mean_score"],
                            std_fold_smape1=checked["std_score"],
                            pooled_oof_smape1=checked["oof_score"],
                            n_predictions=checked["n_predictions"]))
    summary = pd.DataFrame(summary)
    baseline = summary.loc[summary.variant.eq("residual"), "mean_fold_smape1"].iloc[0]
    summary["delta_vs_joint"] = summary.mean_fold_smape1 - baseline
    by_fold = pd.DataFrame(records)
    base_folds = by_fold[by_fold.variant.eq("residual")].set_index("fold").smape1
    by_fold["delta_vs_joint"] = by_fold.smape1 - by_fold.fold.map(base_folds)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_fold.to_csv(output_dir / "folds.csv", index=False)
    oof = samples[["prediction_id", "patient_id", "fold", "rating"]].copy()
    for variant in VARIANTS:
        oof["pred_" + variant] = predictions[variant]
    oof.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps({
        "protocol": manifest["protocol"], "split_id": manifest["split_id"],
        "epochs": epochs, "base_epochs": base_epochs,
        "seeds": [4486, 4487, 4488],
        "note": "Exploratory frozen patient CV; not Kaggle test performance",
        "ai_usage": "Implementation assisted by ChatGPT; verify and disclose",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print("Saved to", output_dir, flush=True)
    return summary, by_fold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--base-epochs", type=int, default=100)
    args = parser.parse_args()
    run(**vars(args))

"""Inference-time residual-strength ablation on frozen patient-level folds.

This is an offline experiment, not a Kaggle submission. Each fold trains
exactly one V18 model, then evaluates lambda=0, .25, .5, .75, 1 without
retraining. Lambda=1 preserves the original V18 predictions.

Usage:
    python C_improved_model/validate_residual_scale.py
    python C_improved_model/validate_residual_scale.py --epochs 2 --output-dir /tmp/scale-smoke

Use 200 epochs for comparable experiments. Fewer epochs are smoke tests only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.metric import evaluate_oof, smape_plus_one
from architectures import ArchitectureModel, ResidualNetwork
from teammate_aligned_model import History, HORIZONS, KEYS, TARGETS, load_training
from teammate_neural_proteins import protein_matrix
from visit_history_model import read_data

DEFAULT_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)


def load_frozen_validation(root, data_dir):
    """Use the shared request IDs/fold mapping, not a reconstructed split."""
    shared = root / "artifacts" / "shared"
    samples = pd.read_csv(shared / "samples.csv")
    folds = pd.read_csv(shared / "patient_folds.csv")
    manifest = json.loads((shared / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol") != "patient5-omics-smape1-v1" or manifest.get("n_splits") != 5:
        raise ValueError("Unexpected CV protocol; do not silently change the validation split")
    if len(samples) != manifest["n_samples"] or folds.patient_id.nunique() != manifest["n_patients"]:
        raise ValueError("Frozen sample or patient count does not match manifest")
    if samples.prediction_id.isna().any() or samples.prediction_id.duplicated().any():
        raise ValueError("Missing/duplicate frozen prediction IDs")
    if folds.patient_id.duplicated().any() or set(samples.fold.unique()) != set(range(5)):
        raise ValueError("Invalid patient-fold mapping")
    expected = samples.patient_id.map(folds.set_index("patient_id").fold)
    if expected.isna().any() or not np.array_equal(expected.to_numpy(), samples.fold.to_numpy()):
        raise ValueError("Frozen request folds do not match frozen patient assignments")
    if samples[["rating", "patient_id", "visit_month", "target", "horizon"]].isna().any().any():
        raise ValueError("Unexpected missing validation fields")

    clinical, omics = read_data(data_dir)
    protein_table = pd.read_csv(
        Path(data_dir) / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"]
    )
    # History.features uses only observation months <= the prediction origin.
    # No UPDRS labels or future measurements enter these inputs.
    history = History()
    visits = pd.concat([clinical[KEYS], omics[KEYS]], ignore_index=True)
    history.update(visits, protein_table)
    features = np.asarray(
        [history.features(patient, month, horizon)
         for patient, month, horizon in samples[KEYS + ["horizon"]]
         .itertuples(index=False, name=None)],
        dtype=np.float32,
    )
    current_proteins = protein_matrix(samples, protein_table)
    targets = samples.target.map({name: i for i, name in enumerate(TARGETS)})
    heads = samples.horizon.map({h: 4 * i for i, h in enumerate(HORIZONS)}) * 1 + targets
    if targets.isna().any() or heads.isna().any():
        raise ValueError("Unknown target or horizon in frozen samples")
    if not set(samples.patient_id).issubset(set(clinical.patient_id)):
        raise ValueError("Validation includes patients not in the main clinical cohort")
    return samples, folds, manifest, features, current_proteins, targets.to_numpy(int), heads.to_numpy(int)


def evaluate_scales(model, features, targets, heads, proteins, scales):
    """Vary lambda on trained networks while preserving their learned weights."""
    if not model.networks or not all(isinstance(net, ResidualNetwork) for net in model.networks):
        raise TypeError("Expected trained V18 residual networks")
    old = [net.residual_scale for net in model.networks]
    available = np.isfinite(proteins).any(axis=1)
    predictions = {}
    try:
        for value in scales:
            for net in model.networks:
                net.residual_scale = value
            pred = np.asarray(
                model.predict_features(features, targets, heads, proteins), dtype=float
            )
            if pred.shape != (len(features),) or not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError("Invalid predictions")
            predictions[value] = pred
        # The correction is masked when both selected proteins are unavailable.
        if (~available).any():
            reference = predictions[1.0][~available]
            for value in scales:
                if not np.allclose(predictions[value][~available], reference, rtol=0, atol=1e-5):
                    raise AssertionError("Lambda changed predictions on protein-absent requests")
    finally:
        for net, value in zip(model.networks, old):
            net.residual_scale = value
    return predictions


def run(root=ROOT, data_dir=None, output_dir=None, epochs=200, scales=DEFAULT_SCALES):
    root = Path(root)
    data_dir = Path(data_dir) if data_dir is not None else root / "amp-parkinsons-disease-progression-prediction"
    output_dir = Path(output_dir) if output_dir is not None else root / "C_improved_model" / "residual_scale_results"
    scales = tuple(float(value) for value in scales)
    if not scales or len(set(scales)) != len(scales) or 1.0 not in scales:
        raise ValueError("Unique scales including lambda=1 are required")
    if not np.isfinite(scales).all() or min(scales) < 0 or epochs < 1:
        raise ValueError("Scales must be finite/nonnegative and epochs >= 1")
    samples, folds, manifest, features, proteins, targets, heads = load_frozen_validation(root, data_dir)
    train_rows, train_features = load_training(data_dir)
    all_proteins = pd.read_csv(
        data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"]
    )
    train_proteins = protein_matrix(train_rows, all_proteins)
    train_proteins[train_rows.source.eq(1).to_numpy()] = np.nan

    oof = {value: np.full(len(samples), np.nan, dtype=float) for value in scales}
    fold_records = []
    for fold in range(5):
        heldout = set(folds.loc[folds.fold.eq(fold), "patient_id"])
        is_valid = samples.fold.eq(fold).to_numpy()
        is_train = ~train_rows.patient_id.isin(heldout).to_numpy()
        if not heldout or not is_valid.any():
            raise ValueError(f"Empty patient validation fold {fold}")
        if set(train_rows.loc[is_train, "patient_id"]).intersection(heldout):
            raise AssertionError("Patient leakage between train and validation")
        print(f"Fold {fold}: {int(is_train.sum())} train requests, "
              f"{int(is_valid.sum())} held-out requests", flush=True)
        model = ArchitectureModel("residual").fit(
            train_rows.loc[is_train].reset_index(drop=True),
            train_features[is_train],
            train_proteins[is_train],
            epochs=epochs,
        )
        predictions = evaluate_scales(
            model, features[is_valid], targets[is_valid], heads[is_valid],
            proteins[is_valid], scales,
        )
        present = np.isfinite(proteins[is_valid]).any(axis=1)
        truth = samples.loc[is_valid, "rating"].to_numpy(dtype=float)
        for value, pred in predictions.items():
            oof[value][is_valid] = pred
            fold_records.append({
                "fold": fold, "lambda": value,
                "smape1": smape_plus_one(truth, pred),
                "protein_present_smape1": smape_plus_one(truth[present], pred[present]) if present.any() else np.nan,
                "protein_absent_smape1": smape_plus_one(truth[~present], pred[~present]) if (~present).any() else np.nan,
                "requests": len(truth),
                "protein_present_requests": int(present.sum()),
            })
        del model

    summary_records = []
    for value in scales:
        if not np.isfinite(oof[value]).all():
            raise AssertionError(f"Incomplete OOF predictions for lambda={value}")
        checked = evaluate_oof(
            samples, pd.DataFrame({"prediction_id": samples.prediction_id, "rating": oof[value]})
        )
        summary_records.append({
            "lambda": value, "mean_fold_smape1": checked["mean_score"],
            "std_fold_smape1": checked["std_score"],
            "pooled_oof_smape1": checked["oof_score"],
            "n_predictions": checked["n_predictions"],
        })
    summary = pd.DataFrame(summary_records)
    baseline = float(summary.loc[summary["lambda"].eq(1.0), "mean_fold_smape1"].iloc[0])
    summary["delta_vs_v18"] = summary.mean_fold_smape1 - baseline
    summary = summary.sort_values("mean_fold_smape1", ignore_index=True)
    by_fold = pd.DataFrame(fold_records)
    baseline_by_fold = by_fold.loc[by_fold["lambda"].eq(1.0)].set_index("fold")["smape1"]
    by_fold["delta_vs_v18"] = by_fold.smape1 - by_fold.fold.map(baseline_by_fold)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)
    by_fold.to_csv(output_dir / "folds.csv", index=False)
    predictions = samples[["prediction_id", "patient_id", "fold", "rating"]].copy()
    for value in scales:
        predictions[f"pred_lambda_{str(value).replace('.', 'p')}"] = oof[value]
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "model": "V18 residual (LightGBM + 3 seed neural ensemble)",
            "epochs": epochs, "neural_seeds": [4486, 4487, 4488],
            "scales": list(scales), "split_id": manifest["split_id"],
            "n_samples": len(samples), "n_patients": folds.patient_id.nunique(),
            "data_dir": str(data_dir),
            "validation": "frozen patient-level 5-fold SMAPE+1",
            "training": "original clinical-origin requests + supplemental; no held-out patients",
            "interpretation": "Selecting lambda on these folds is exploratory; not an independent test result",
            "ai_usage": "Code structure/debugging assisted by ChatGPT; group should review and document design",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"Saved validation artifacts to {output_dir}", flush=True)
    return summary, by_fold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    run(data_dir=args.data_dir, output_dir=args.output_dir, epochs=args.epochs)


if __name__ == "__main__":
    main()

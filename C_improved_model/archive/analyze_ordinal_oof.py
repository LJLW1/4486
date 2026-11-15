"""Aggregate an ALREADY COMPLETED 5-fold ordinal OOF run; no fitting.

python C_improved_model/analyze_ordinal_oof.py
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "C_improved_model/ordinal_results"

def analyze(input_path, output_dir):
    df = pd.read_csv(input_path)
    keys = ["patient_id", "visit_month", "target", "horizon"]
    names = ["prediction_multiclass", "prediction_ordinal"]
    if any(c not in df.columns for c in keys + names + ["rating", "fold"]):
        raise ValueError("Missing OOF fields")
    if df.duplicated(keys).any():
        raise ValueError("Duplicate prediction keys")
    if set(df.fold) != set(range(5)) or set(df.target) != {f"updrs_{i}" for i in range(1, 5)}:
        raise ValueError("Requires all five folds and four UPDRS targets")
    frozen = pd.read_csv(ROOT / "artifacts/shared/samples.csv")
    original = frozen[keys + ["fold", "rating"]].copy()
    if len(original) != len(df) or original.duplicated(keys).any():
        raise ValueError("OOF request count / uniqueness differs from frozen validation")
    check = original.merge(df[keys + ["fold", "rating"] + names], on=keys,
                           how="outer", validate="one_to_one",
                           suffixes=("_fixed", "_oof"), indicator=True)
    if not check._merge.eq("both").all() or (check.fold_fixed != check.fold_oof).any():
        raise ValueError("OOF keys or folds differ from frozen validation")
    if not np.allclose(check.rating_fixed, check.rating_oof, atol=1e-8, rtol=0):
        raise ValueError("OOF labels differ from frozen validation")
    y = check.rating_fixed.to_numpy(float)
    for name in names:
        p = check[name].to_numpy(float)
        if not np.isfinite(p).all() or (p < 0).any():
            raise ValueError(f"Invalid predictions: {name}")
        check["loss_" + name.removeprefix("prediction_")] = 200 * np.abs(y-p)/(y+p+2)
    check["delta"] = check.loss_ordinal - check.loss_multiclass
    check["flipped"] = check.prediction_multiclass != check.prediction_ordinal
    folds = check.groupby("fold_fixed").agg(
        requests=("delta", "size"), multiclass=("loss_multiclass", "mean"),
        ordinal=("loss_ordinal", "mean"), delta=("delta", "mean"),
        flip_rate=("flipped", "mean")).reset_index().rename(columns={"fold_fixed": "fold"})
    heads = check.groupby(["target", "horizon"]).agg(
        requests=("delta", "size"), multiclass=("loss_multiclass", "mean"),
        ordinal=("loss_ordinal", "mean"), delta=("delta", "mean"),
        flip_rate=("flipped", "mean")).reset_index()
    overall = pd.DataFrame([{
        "mean_fold_multiclass": folds.multiclass.mean(),
        "mean_fold_ordinal": folds.ordinal.mean(),
        "delta_vs_multiclass": folds.delta.mean(),
        "fold_win_rate": (folds.delta < 0).mean(),
        "worst_fold_regression": folds.delta.max(),
        "std_fold_delta": folds.delta.std(ddof=0),
        "pooled_multiclass": check.loss_multiclass.mean(),
        "pooled_ordinal": check.loss_ordinal.mean(),
        "prediction_flip_rate": check.flipped.mean(),
        "n_predictions": len(check)
    }])
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "ordinal_overall_folds.csv", index=False)
    heads.to_csv(output_dir / "ordinal_target_horizon.csv", index=False)
    overall.to_csv(output_dir / "ordinal_overall_summary.csv", index=False)
    print(overall.to_string(index=False))
    print(folds.to_string(index=False))
    return overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT / "ordinal_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()
    analyze(args.input, args.output_dir)

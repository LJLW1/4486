"""Cross-fitted horizon-gated residual strength using EXISTING V18 OOF predictions.

No model retraining. This is a fast, held-out assessment of a novel
horizon-dependent residual gate, not a finished Kaggle submission.

Requirement: run validate_residual_scale.py once previously with its default
lambda grid; uses resulting residual_scale_results/oof_predictions.csv.

Command: python C_improved_model/validate_horizon_gated_residual.py
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "C_improved_model/residual_scale_results/oof_predictions.csv"
OUTPUT = ROOT / "C_improved_model/horizon_gate_results"
CANDIDATES = (0.5, 0.75, 1.0)
REFERENCE = 0.75
# In a small sample, require a meaningful margin before changing the gate.
MARGIN = 0.10


def error(y, p):
    return 200.0 * np.abs(y - p) / (y + p + 2)


def run(oof_path=OOF, output=OUTPUT):
    oof_path, output = Path(oof_path), Path(output)
    predictions = pd.read_csv(oof_path)
    frozen = pd.read_csv(ROOT / "artifacts/shared/samples.csv")
    if "prediction_id" not in predictions.columns:
        raise ValueError("OOF predictions must include prediction_id")
    required = ["pred_lambda_" + str(t).replace(".", "p") for t in CANDIDATES]
    if not set(required).issubset(predictions):
        raise ValueError("Missing lambda columns. Required: " + ", ".join(required))
    if predictions.prediction_id.duplicated().any() or frozen.prediction_id.duplicated().any():
        raise ValueError("Duplicate request IDs")
    if len(frozen) != len(predictions):
        raise ValueError("OOF row count mismatch")
    frame = frozen[["prediction_id", "patient_id", "fold", "horizon", "rating"]].merge(
        predictions[["prediction_id"] + required], on="prediction_id",
        validate="one_to_one", how="left"
    )
    if frame[required].isna().any().any():
        raise ValueError("Missing held-out model predictions")
    if set(frame.fold) != set(range(5)):
        raise ValueError("Expected five patient-level folds")

    y = frame.rating.to_numpy(dtype=float)
    all_predictions = {t: frame["pred_lambda_" + str(t).replace(".", "p")].to_numpy(float)
                       for t in CANDIDATES}
    for t in CANDIDATES:
        if not np.isfinite(all_predictions[t]).all() or (all_predictions[t] < 0).any():
            raise ValueError("Invalid prediction values")
        frame["loss_lambda_" + str(t).replace(".", "p")] = error(y, all_predictions[t])

    gated = np.full(len(frame), np.nan)
    selected = []
    # Fold f is NEVER consulted when choosing its own parameters.
    for fold in range(5):
        train = frame.fold.ne(fold).to_numpy()
        held = frame.fold.eq(fold).to_numpy()
        for horizon in sorted(frame.horizon.unique()):
            history = train & frame.horizon.eq(horizon).to_numpy()
            test = held & frame.horizon.eq(horizon).to_numpy()
            if not test.any():
                continue
            losses = {
                t: float(error(y[history], all_predictions[t][history]).mean())
                for t in CANDIDATES
            }
            best = min(CANDIDATES, key=lambda t: losses[t])
            # Regularization without tuning an extra strength hyperparameter.
            chosen = best if losses[REFERENCE] - losses[best] >= MARGIN else REFERENCE
            gated[test] = all_predictions[chosen][test]
            selected.append({
                "fold": fold, "horizon": int(horizon),
                "chosen_lambda": chosen, "training_reference": losses[REFERENCE],
                "training_best": losses[best], "training_gain": losses[REFERENCE]-losses[best],
                "n_training_requests": int(history.sum()), "n_heldout_requests": int(test.sum())
            })
    if not np.isfinite(gated).all():
        raise AssertionError("Incomplete cross-fitted gated predictions")
    frame["prediction_fixed_075"] = all_predictions[REFERENCE]
    frame["prediction_horizon_gate"] = gated
    frame["loss_fixed_075"] = error(y, all_predictions[REFERENCE])
    frame["loss_horizon_gate"] = error(y, gated)
    frame["delta"] = frame.loss_horizon_gate - frame.loss_fixed_075

    scores = frame.groupby("fold").agg(
        n=("rating", "size"),
        fixed_075=("loss_fixed_075", "mean"),
        horizon_gate=("loss_horizon_gate", "mean"),
        delta=("delta", "mean")
    ).reset_index()
    summary = pd.DataFrame([{
        "fixed_075_mean_fold": scores.fixed_075.mean(),
        "horizon_gate_mean_fold": scores.horizon_gate.mean(),
        "delta_vs_fixed_075": scores.delta.mean(),
        "fold_win_rate": float((scores.delta < 0).mean()),
        "worst_fold_regression": scores.delta.max(),
        "prediction_flip_rate": float((frame.prediction_fixed_075 != frame.prediction_horizon_gate).mean()),
        "n_requests": len(frame),
    }])
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "oof_predictions.csv", index=False)
    scores.to_csv(output / "folds.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    pd.DataFrame(selected).to_csv(output / "selected_gates.csv", index=False)
    (output / "metadata.json").write_text(json.dumps({
        "model": "cross-fitted horizon gate over V18 residual strengths",
        "candidates": list(CANDIDATES), "reference": REFERENCE, "minimum_margin": MARGIN,
        "protocol": "gate trained on other four held-out-patient folds; evaluated on excluded fold",
        "limitations": "Repeated model development on same five folds is exploratory. "
                       "This does not establish leaderboard improvement or inference feature parity.",
        "training_time": "No new tree/network fits; uses stored OOF predictions.",
    }, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(scores.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=OOF)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.oof, args.output)

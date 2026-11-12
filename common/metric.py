"""Competition SMAPE+1, plus strict prediction-ID alignment for local CV."""

import numpy as np
import pandas as pd


def smape_plus_one(y_true, y_pred):
    """200 * mean(abs(y-p) / (abs(y+1)+abs(p+1))); lower is better.

    Missing truth is masked; a prediction for each observed truth must be finite.
    No clipping is performed. Callers own any explicit prediction postprocessing.
    """
    truth, pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape or truth.size == 0:
        raise ValueError("Truth and prediction must have the same nonempty shape")
    if np.isinf(truth).any() or (truth[np.isfinite(truth)] < 0).any():
        raise ValueError("Observed truth must be finite and nonnegative")
    observed = ~np.isnan(truth)
    if not observed.any() or not np.isfinite(pred[observed]).all():
        raise ValueError("No observed truth or nonfinite prediction for observed truth")
    truth, pred = truth[observed], pred[observed]
    return float(200 * np.mean(np.abs(truth - pred) / (np.abs(truth + 1) + np.abs(pred + 1))))


def evaluate_oof(samples, predictions):
    """Require exactly one OOF prediction per labelled request; reject dropped rows."""
    for frame, columns in [(samples, ["prediction_id", "rating", "fold"]),
                           (predictions, ["prediction_id", "rating"])]:
        if not set(columns).issubset(frame.columns):
            raise ValueError(f"Required columns: {columns}")
        if frame.prediction_id.isna().any() or frame.prediction_id.duplicated().any():
            raise ValueError("Prediction IDs must be present and unique")
    if set(samples.prediction_id) != set(predictions.prediction_id):
        raise ValueError("OOF prediction IDs must exactly match labelled sample IDs")
    if samples[["rating", "fold"]].isna().any().any():
        raise ValueError("OOF samples must have observed labels and fold assignments")
    aligned = samples[["prediction_id", "rating", "fold"]].merge(
        predictions[["prediction_id", "rating"]], on="prediction_id", validate="one_to_one",
        suffixes=("_true", "_pred"),
    )
    scores = {int(fold): smape_plus_one(group.rating_true, group.rating_pred)
              for fold, group in aligned.groupby("fold", sort=True)}
    return {
        "fold_scores": scores,
        "mean_score": float(np.mean(list(scores.values()))),
        "std_score": float(np.std(list(scores.values()))),
        "oof_score": smape_plus_one(aligned.rating_true, aligned.rating_pred),
        "n_predictions": len(aligned),
    }

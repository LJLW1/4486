"""Low-cost additive residual calibrator for an EXISTING V7 model.

Requires predictions from V7 that are out-of-fold by patient; do not fit the
residual on V7's in-sample predictions. Fits only a ridge regression, NOT V7.

    python C_improved_model/v7_residual_adapter.py --oof PATH_TO_V7_OOF.csv

OOF CSV: prediction_id, rating, prediction (or pred), with predictions made
by a V7 model that never trained on that patient's labels. For pure training
without evaluation, no fallback to in-sample predictions is permitted.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "artifacts/shared/samples.csv"
DEFAULT_OUT = ROOT / "C_improved_model/v7_residual_results"

TARGETS = ("updrs_1", "updrs_2", "updrs_3", "updrs_4")
HORIZONS = (0, 6, 12, 24)
RIDGE_ALPHA = 100.0
SHRINK = 0.25
MAX_DELTA = 3.0


def feature_matrix(predictions, target, horizon, visit_month):
    """Stable low-dimensional features available in every official prediction_id."""
    p = np.asarray(predictions, float).reshape(-1)
    t = np.asarray(target).astype(str).reshape(-1)
    h = np.asarray(horizon, int).reshape(-1)
    m = np.asarray(visit_month, float).reshape(-1)
    if not (len(p) == len(t) == len(h) == len(m)):
        raise ValueError("Prediction feature lengths differ")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("V7 prediction must be finite and nonnegative")
    if not set(t).issubset(TARGETS) or not set(h).issubset(HORIZONS):
        raise ValueError("Unknown target or horizon")
    # No identifiers, labels, future visits, biomarkers, or hidden test history.
    return np.column_stack([
        np.ones(len(p)), p / 40.0, np.log1p(p) / 4.0,
        m / 84.0, (m + h) / 108.0,
        *[(t == v).astype(float) for v in TARGETS],
        *[(h == v).astype(float) for v in HORIZONS],
        *[((t == v) & (h == k)).astype(float)
          for v in TARGETS for k in HORIZONS],
    ])


def smape(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return 200 * np.abs(y - pred) / (y + pred + 2)


class V7ResidualCorrector:
    """Fits bounded residuals against patient-OOF V7 forecasts only."""

    def __init__(self, ridge_alpha=RIDGE_ALPHA, shrink=SHRINK, max_delta=MAX_DELTA):
        self.ridge_alpha = float(ridge_alpha)
        self.shrink = float(shrink)
        self.max_delta = float(max_delta)
        if self.ridge_alpha <= 0 or not 0 < self.shrink <= 1 or self.max_delta <= 0:
            raise ValueError("Invalid correction settings")
        self.coefficients = None

    def fit(self, frame):
        required = {"rating", "prediction", "target", "horizon", "visit_month"}
        if not required.issubset(frame):
            raise ValueError("Missing fields: " + repr(sorted(required - set(frame))))
        y, p = frame.rating.to_numpy(float), frame.prediction.to_numpy(float)
        x = feature_matrix(p, frame.target, frame.horizon, frame.visit_month)
        raw_residual = y - p
        reg = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
        reg.fit(x, raw_residual)
        self.coefficients = np.asarray(reg.coef_, float)
        return self

    def predict(self, predictions, target, horizon, visit_month):
        if self.coefficients is None:
            raise RuntimeError("Fit/load a corrector first")
        p = np.asarray(predictions, dtype=float).reshape(-1)
        x = feature_matrix(p, target, horizon, visit_month)
        adjustment = np.clip(x @ self.coefficients, -self.max_delta, self.max_delta)
        return np.maximum(0.0, p + self.shrink * adjustment)

    def dump(self, path):
        if self.coefficients is None:
            raise RuntimeError("No fitted coefficients")
        Path(path).write_text(json.dumps({
            "ridge_alpha": self.ridge_alpha,
            "shrink": self.shrink,
            "max_delta": self.max_delta,
            "coefficients": self.coefficients.tolist(),
            "input": "V7 prediction, target, horizon, visit_month",
            "training": "patient-heldout V7 OOF only",
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        cfg = json.loads(Path(path).read_text(encoding="utf-8"))
        item = cls(cfg["ridge_alpha"], cfg["shrink"], cfg["max_delta"])
        item.coefficients = np.asarray(cfg["coefficients"], dtype=float)
        if item.coefficients.shape != (feature_matrix([0], ["updrs_1"], [0], [0]).shape[1],):
            raise ValueError("Unexpected coefficient shape")
        return item

    def correct_api_output(self, output):
        """Apply after V7 predict_batch returns submission-formatted DataFrame."""
        import re
        parsed = output.prediction_id.str.extract(
            r"^(\d+)_(\d+)_(updrs_[1-4])_plus_(0|6|12|24)_months$"
        )
        if parsed.isna().any().any():
            raise ValueError("Unexpected competition prediction_id")
        result = output.copy()
        result["rating"] = self.predict(
            result["rating"].to_numpy(float), parsed[2].to_numpy(),
            parsed[3].astype(int).to_numpy(), parsed[1].astype(int).to_numpy()
        )
        return result


def load_heldout_oof(path):
    raw = pd.read_csv(path)
    if "prediction" not in raw and "pred" in raw:
        raw = raw.rename(columns={"pred": "prediction"})
    if "prediction_id" not in raw or "prediction" not in raw:
        raise ValueError("V7 OOF needs prediction_id and prediction")
    samples = pd.read_csv(FROZEN)
    if samples.prediction_id.duplicated().any() or raw.prediction_id.duplicated().any():
        raise ValueError("Duplicate prediction IDs")
    if len(samples) != len(raw):
        raise ValueError("V7 OOF does not cover all frozen validation requests")
    # Frozen truth and fold assignments are authoritative, not supplied metadata.
    joined = samples[["prediction_id", "patient_id", "visit_month",
                      "target", "horizon", "rating", "fold"]].merge(
        raw[["prediction_id", "prediction"]],
        on="prediction_id", how="left", validate="one_to_one"
    )
    if joined.prediction.isna().any():
        raise ValueError("Missing V7 OOF predictions")
    if "fold" in raw:
        comparison = samples[["prediction_id", "fold"]].merge(
            raw[["prediction_id", "fold"]], on="prediction_id",
            suffixes=("_frozen", "_oof"), validate="one_to_one"
        )
        if not (comparison.fold_frozen == comparison.fold_oof).all():
            raise ValueError("V7 OOF fold assignment differs from frozen folds")
    return joined


def evaluate(oof_path, output_dir):
    df = load_heldout_oof(oof_path)
    corrected = np.full(len(df), np.nan)
    for fold in range(5):
        training = df.loc[df.fold.ne(fold)]
        holdout = df.loc[df.fold.eq(fold)]
        if set(training.patient_id) & set(holdout.patient_id):
            raise AssertionError("Patient leakage")
        estimator = V7ResidualCorrector().fit(training)
        corrected[holdout.index] = estimator.predict(
            holdout.prediction, holdout.target, holdout.horizon, holdout.visit_month
        )
    if not np.isfinite(corrected).all():
        raise ValueError("Missing corrected predictions")
    df["corrected_prediction"] = corrected
    df["baseline_loss"] = smape(df.rating, df.prediction)
    df["corrected_loss"] = smape(df.rating, corrected)
    df["delta_loss"] = df.corrected_loss - df.baseline_loss
    folds = df.groupby("fold").agg(
        baseline=("baseline_loss", "mean"),
        corrected=("corrected_loss", "mean"),
        delta=("delta_loss", "mean"),
    ).reset_index()
    summary = pd.DataFrame([{
        "baseline_fold_mean": folds.baseline.mean(),
        "corrected_fold_mean": folds.corrected.mean(),
        "delta": folds.delta.mean(),
        "fold_win_rate": (folds.delta < 0).mean(),
        "worst_fold_regression": folds.delta.max(),
        "prediction_change_rate": (np.abs(df.prediction-df.corrected_prediction) > 1e-8).mean(),
        "requests": len(df)
    }])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(output_dir / "folds.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    df.to_csv(output_dir / "oof_predictions.csv", index=False)
    # Final fit uses already cross-fitted V7 predictions; safe to deploy only
    # if a separate held-out Kaggle test supports the improvement.
    V7ResidualCorrector().fit(df).dump(output_dir / "residual_corrector.json")
    print(summary.to_string(index=False), flush=True)
    print(folds.to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, required=True,
                        help="PATIENT-HELDOUT V7 OOF, not V7 in-sample outputs")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    evaluate(args.oof, args.out)

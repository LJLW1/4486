"""Offline stage-1 row/weight/time ablation and stage-2 temperature calibration.

Both stages use the same frozen patient-level folds and report mean-fold
SMAPE+1 (primary) plus pooled OOF (secondary). No Kaggle submission.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from visit_history_model import KEYS, TARGETS, read_data, training_rows
from fusion_experiment import PARAMS, decision, spread_prob
from competition_optimized_experiment import make_features

ROOT = Path(__file__).resolve().parents[1]
MONTHS = (0, 6, 12, 18, 24, 36, 48, 60, 72, 84)
SOURCES = ("omics", "winner_origins", "winner_origins_supplemental")
WEIGHTS = ("uniform", "unique_label", "patient_equal")
TIME = ("original", "relative")
TEMPERATURES = (0.70, 0.85, 1.0, 1.2, 1.5)


def loss(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return 200 * np.abs(y - pred) / (y + pred + 2)


def load_data(data_dir):
    clinical, omics = read_data(data_dir)
    supplemental = pd.read_csv(data_dir / "supplemental_clinical_data.csv")
    supplemental.loc[supplemental.visit_month.eq(5), "visit_month"] = 6
    proteins = pd.read_csv(data_dir / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    folds = pd.read_csv(ROOT / "artifacts/shared/patient_folds.csv")
    evaluation = training_rows(clinical, omics).merge(folds, on="patient_id", validate="many_to_one")
    full = training_rows(clinical, clinical[KEYS]).merge(folds, on="patient_id", validate="many_to_one")
    full = full[full.visit_month.isin(MONTHS)].reset_index(drop=True)
    extra = training_rows(supplemental, supplemental[KEYS])
    extra = extra[extra.visit_month.isin(MONTHS)].reset_index(drop=True)
    blank = pd.DataFrame(columns=KEYS)
    datasets = {
        "omics": (evaluation, make_features(evaluation, clinical, omics, proteins).assign(supplemental=0)),
        "winner_origins": (full, make_features(full, clinical, clinical[KEYS], proteins).assign(supplemental=0)),
        "winner_origins_supplemental_extra": (
            extra, make_features(extra, supplemental, blank, blank).assign(supplemental=1)
        ),
    }
    return evaluation, datasets


def time_features(features, strategy):
    x = features.copy().reset_index(drop=True)
    if strategy == "relative":
        x["target_year"] = x.target_month / 12.0
        x["relative_horizon"] = x.horizon / (x.visit_month + 12.0)
    elif strategy != "original":
        raise ValueError(strategy)
    return x


def get_train(datasets, heldout, source):
    if source not in SOURCES:
        raise ValueError(source)
    rows, x = datasets["omics" if source == "omics" else "winner_origins"]
    allowed = ~rows.patient_id.isin(heldout).to_numpy()
    train, features = rows.loc[allowed].reset_index(drop=True), x.loc[allowed].reset_index(drop=True)
    if source == "winner_origins_supplemental":
        extra, z = datasets["winner_origins_supplemental_extra"]
        safe = ~extra.patient_id.isin(heldout).to_numpy()
        train = pd.concat([train, extra.loc[safe]], ignore_index=True)
        features = pd.concat([features, z.loc[safe]], ignore_index=True)
    assert set(train.patient_id).isdisjoint(heldout)
    return train, features


def weights_for(rows, method):
    if method == "uniform":
        weights = np.ones(len(rows))
    elif method == "unique_label":
        keys = ["patient_id", "target_month", "target"]
        weights = 1 / rows.groupby(keys, dropna=False)["rating"].transform("size").to_numpy(float)
    elif method == "patient_equal":
        weights = 1 / rows.groupby("patient_id")["rating"].transform("size").to_numpy(float)
    else:
        raise ValueError(method)
    weights = weights / weights.mean()
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Invalid sample weights")
    return weights


def predict_prob(model, frame):
    return spread_prob(model.predict_proba(frame), model.classes_, int(model.classes_.max()) + 1)


def apply_temperature(prob, temperature):
    p = np.maximum(prob.astype(float), 1e-12) ** (1.0 / temperature)
    return p / p.sum(axis=1, keepdims=True)


def fit_target(train, train_x, target, weighting):
    chosen = train.target.eq(target).to_numpy()
    if not chosen.any():
        raise ValueError("Empty target training set")
    model = LGBMClassifier(**PARAMS)
    w = weights_for(train, weighting)
    model.fit(train_x.loc[chosen], train.loc[chosen, "rating"].astype(int),
              sample_weight=w[chosen])
    return model


def fold_score_table(records):
    if not records:
        raise ValueError("No experiment records to aggregate")
    df = pd.concat(records, ignore_index=True)
    fold = df.groupby(["variant", "fold"], sort=False).loss.mean().reset_index()
    summary = fold.groupby("variant").loss.agg(
        mean_fold_smape1="mean", std_fold_smape1=lambda v: v.std(ddof=0)
    )
    summary["pooled_oof_smape1"] = df.groupby("variant").loss.mean()
    summary["requests"] = df.groupby("variant").size()
    return df, fold, summary.sort_values("mean_fold_smape1")


def stage_rows(evaluation, datasets, output):
    records = []
    evx = datasets["omics"][1]
    for fold in range(5):
        valid = evaluation.loc[evaluation.fold.eq(fold)].reset_index(drop=True)
        valid_x0 = evx.loc[evaluation.fold.eq(fold)].reset_index(drop=True)
        held = set(valid.patient_id)
        for source in SOURCES:
            train, x0 = get_train(datasets, held, source)
            for time in TIME:
                tx, vx = time_features(x0, time), time_features(valid_x0, time)
                for weight in WEIGHTS:
                    predictions = np.full(len(valid), np.nan)
                    for target in TARGETS:
                        m = fit_target(train, tx, target, weight)
                        selected = valid.target.eq(target).to_numpy()
                        predictions[selected] = decision(predict_prob(m, vx.loc[selected]))
                    if not np.isfinite(predictions).all():
                        raise ValueError("Invalid predictions")
                    name = f"{source}__{weight}__{time}"
                    frame = valid[["patient_id", "visit_month", "target", "horizon", "rating", "fold"]].copy()
                    frame["variant"], frame["prediction"] = name, predictions
                    frame["loss"] = loss(frame.rating, frame.prediction)
                    records.append(frame)
                    print(f"fold {fold} {name}: {frame.loss.mean():.4f}", flush=True)
    save(records, output, "rows")


def stage_temperature(evaluation, datasets, output, source, weight, time, inner_fraction):
    records = []
    evx = datasets["omics"][1]
    if source not in SOURCES or weight not in WEIGHTS or time not in TIME:
        raise ValueError("Unsupported configuration")
    for fold in range(5):
        valid = evaluation.loc[evaluation.fold.eq(fold)].reset_index(drop=True)
        valid_x = time_features(evx.loc[evaluation.fold.eq(fold)], time)
        held = set(valid.patient_id)
        train, train_x0 = get_train(datasets, held, source)
        main_patients = np.sort(train.loc[
            train.patient_id.isin(datasets["winner_origins"][0].patient_id), "patient_id"
        ].unique())
        fit_ids, cal_ids = train_test_split(
            main_patients, test_size=inner_fraction, random_state=4486 + fold, shuffle=True
        )
        fit_ids, cal_ids = set(fit_ids), set(cal_ids)
        in_fit = ~train.patient_id.isin(cal_ids).to_numpy()
        in_cal = train.patient_id.isin(cal_ids).to_numpy()
        assert not (held & (fit_ids | cal_ids)) and not (fit_ids & cal_ids)
        x = time_features(train_x0, time)
        predictions = np.full(len(valid), np.nan)
        temperatures = {}
        for target in TARGETS:
            fit = train.loc[in_fit].reset_index(drop=True)
            fit_x = x.loc[in_fit].reset_index(drop=True)
            calibration = train.loc[in_cal & train.target.eq(target).to_numpy()].reset_index(drop=True)
            calibration_x = x.loc[in_cal & train.target.eq(target).to_numpy()].reset_index(drop=True)
            model = fit_target(fit, fit_x, target, weight)
            if len(calibration) < 30:
                best = 1.0
            else:
                p = predict_prob(model, calibration_x)
                scores = [
                    loss(calibration.rating, decision(apply_temperature(p, t))).mean()
                    for t in TEMPERATURES
                ]
                best = TEMPERATURES[int(np.argmin(scores))]
            temperatures[target] = best
            full_model = fit_target(train, x, target, weight)
            selected = valid.target.eq(target).to_numpy()
            if selected.any():
                p_valid = predict_prob(full_model, valid_x.loc[selected])
                predictions[selected] = decision(apply_temperature(p_valid, best))
        if not np.isfinite(predictions).all():
            raise ValueError("Invalid predictions")
        frame = valid[["patient_id", "visit_month", "target", "horizon", "rating", "fold"]].copy()
        frame["variant"], frame["prediction"] = "temperature", predictions
        frame["loss"] = loss(frame.rating, frame.prediction)
        records.append(frame)
        print(f"fold {fold}: temperatures {temperatures}, score {frame.loss.mean():.4f}", flush=True)
        (output / f"temperatures_fold{fold}.json").write_text(json.dumps(temperatures, indent=2) + "\n")
    save(records, output, "temperature")


def stage_temperature_check(evaluation, datasets, output, source, weight, time):
    """Fast paired check: 20 full models, no re-training inner calibration.

    Reuses temperatures_fold*.json from the previous temperature experiment.
    The same fitted classifier predicts T=1 and selected T, isolating only
    probability calibration. Does not tune using outer validation labels.
    """
    records = []
    evx = datasets["omics"][1]
    diagnostics = []
    for fold in range(5):
        temperature_path = output / f"temperatures_fold{fold}.json"
        if not temperature_path.is_file():
            raise FileNotFoundError(
                f"Missing {temperature_path}; first complete the original temperature stage"
            )
        temperatures = json.loads(temperature_path.read_text())
        if set(temperatures) != set(TARGETS):
            raise ValueError(f"Unexpected temperature targets in {temperature_path}")
        valid_mask = evaluation.fold.eq(fold)
        valid = evaluation.loc[valid_mask].reset_index(drop=True)
        valid_x = time_features(evx.loc[valid_mask], time)
        train, train_x0 = get_train(datasets, set(valid.patient_id), source)
        train_x = time_features(train_x0, time)
        baseline = np.full(len(valid), np.nan)
        calibrated = np.full(len(valid), np.nan)
        for target in TARGETS:
            selected = valid.target.eq(target).to_numpy()
            if not selected.any():
                continue
            model = fit_target(train, train_x, target, weight)
            prob = predict_prob(model, valid_x.loc[selected])
            temperature = float(temperatures[target])
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError(f"Invalid temperature for {target}: {temperature}")
            baseline[selected] = decision(prob)
            calibrated[selected] = decision(apply_temperature(prob, temperature))
            del model
        if not np.isfinite(baseline).all() or not np.isfinite(calibrated).all():
            raise AssertionError("Incomplete paired predictions")
        frame = valid[["patient_id", "visit_month", "target", "horizon", "rating", "fold"]].copy()
        frame["pred_t1"] = baseline
        frame["pred_calibrated"] = calibrated
        frame["flipped"] = baseline != calibrated
        frame["loss_t1"] = loss(frame.rating, baseline)
        frame["loss_calibrated"] = loss(frame.rating, calibrated)
        frame["delta_loss"] = frame.loss_calibrated - frame.loss_t1
        diagnostics.append(frame)
        for name, pred in (("t1_same_fit", baseline), ("calibrated_same_fit", calibrated)):
            item = valid[["patient_id", "visit_month", "target", "horizon", "rating", "fold"]].copy()
            item["variant"], item["prediction"] = name, pred
            item["loss"] = loss(item.rating, pred)
            records.append(item)
        # Incremental recovery of completed folds: no lost results on later failures.
        pd.concat(diagnostics, ignore_index=True).to_csv(
            output / "temperature_check_pairs.csv", index=False
        )
        print(
            f"fold {fold}: T=1 {frame.loss_t1.mean():.4f}; "
            f"calibrated {frame.loss_calibrated.mean():.4f}; "
            f"flip rate {frame.flipped.mean():.2%}", flush=True
        )
    save(records, output, "temperature_check")
    pairs = pd.concat(diagnostics, ignore_index=True)
    grouped = pairs.groupby(["target", "horizon"], as_index=False).agg(
        n=("flipped", "size"), flip_rate=("flipped", "mean"),
        mean_delta_loss=("delta_loss", "mean")
    )
    grouped.to_csv(output / "temperature_check_breakdown.csv", index=False)
    print(f"overall prediction flip rate: {pairs.flipped.mean():.2%}", flush=True)


def save(records, output, name):
    df, fold, summary = fold_score_table(records)
    df.to_csv(output / f"{name}_oof.csv", index=False)
    fold.to_csv(output / f"{name}_folds.csv", index=False)
    summary.to_csv(output / f"{name}_summary.csv")
    print(summary.head(20).to_string(), flush=True)
    print("Saved:", output, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("rows", "temperature", "temperature-check"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "amp-parkinsons-disease-progression-prediction")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "C_improved_model" / "row_time_calibration_results")
    parser.add_argument("--source", choices=SOURCES, default="winner_origins")
    parser.add_argument("--weight", choices=WEIGHTS, default="unique_label")
    parser.add_argument("--time", choices=TIME, default="original")
    parser.add_argument("--inner-fraction", type=float, default=.25)
    args = parser.parse_args()
    if not 0 < args.inner_fraction < .5:
        parser.error("--inner-fraction must be between 0 and 0.5")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluation, datasets = load_data(args.data_dir)
    if args.stage == "rows":
        stage_rows(evaluation, datasets, args.output_dir)
    elif args.stage == "temperature":
        stage_temperature(evaluation, datasets, args.output_dir, args.source, args.weight,
                          args.time, args.inner_fraction)
    else:
        stage_temperature_check(evaluation, datasets, args.output_dir, args.source,
                                args.weight, args.time)


if __name__ == "__main__":
    main()

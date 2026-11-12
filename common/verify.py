"""Real-data integration checks, without fitting a model or creating model scores."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import N_SPLITS
from .data import KEYS, load_raw, prediction_requests, visit_features
from .interface import model_folds
from .metric import evaluate_oof
from .prepare import prepare


def verify(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    manifest = prepare(data_dir, output_dir)
    raw = load_raw(data_dir)
    samples = pd.read_csv(output_dir / "samples.csv")
    patients = pd.read_csv(output_dir / "patient_folds.csv")
    if not np.array_equal(np.sort(samples.fold.unique()), np.arange(N_SPLITS)):
        raise ValueError("Expected all five folds")
    # Identity predictions test plumbing only; they are never logged as an experiment.
    identity = samples[["prediction_id", "rating"]].sample(frac=1, random_state=1)
    result = evaluate_oof(samples, identity)
    if result["oof_score"] != 0:
        raise ValueError("Identity/alignment check failed")
    folds = []
    for data in model_folds(raw, samples, patients):
        if set(data.train_samples.patient_id) & set(data.valid_samples.patient_id):
            raise ValueError("Patient leakage")
        if set(data.supplemental.patient_id) & set(data.valid_samples.patient_id):
            raise ValueError("Supplemental leakage")
        if list(data.train_features) != list(data.valid_features):
            raise ValueError("Fold feature schemas differ")
        for labels, features in [(data.train_samples, data.train_features),
                                 (data.valid_samples, data.valid_features)]:
            if not labels.prediction_id.reset_index(drop=True).equals(features.prediction_id):
                raise ValueError("Features and labels misaligned")
            if "rating" in features or any(c.startswith("updrs_") for c in features):
                raise ValueError("Clinical labels in features")
        folds.append(dict(fold=data.fold, train_rows=len(data.train_samples),
                          valid_rows=len(data.valid_samples), feature_columns=len(data.train_features.columns)))
    test = pd.read_csv(data_dir / "example_test_files/test.csv")
    requests = prediction_requests(test[KEYS])
    submission = pd.read_csv(data_dir / "example_test_files/sample_submission.csv")
    if set(requests.prediction_id) != set(submission.prediction_id):
        raise ValueError("Request IDs differ from Kaggle example")
    test_proteins = pd.read_csv(data_dir / "example_test_files/test_proteins.csv")
    test_peptides = pd.read_csv(data_dir / "example_test_files/test_peptides.csv")
    from .data import RawData, validate_table
    test_raw = RawData(raw.clinical.iloc[:0], validate_table(test_proteins, "proteins"),
                       validate_table(test_peptides, "peptides"), raw.supplemental.iloc[:0])
    inference_features = visit_features(test[KEYS], test_raw)
    if len(inference_features) != test.visit_id.nunique():
        raise ValueError("Missing-omics visits were lost")
    report = dict(status="passed", split_id=manifest["split_id"], n_patients=len(patients),
                  n_labelled_predictions=len(samples), folds=folds,
                  example_request_count=len(requests),
                  example_visits_without_omics=int(inference_features.filter(regex="::").isna().all(axis=1).sum()),
                  checks=["input/output hashes", "patient isolation", "training-only vocabulary",
                          "no clinical labels in features", "feature/label alignment",
                          "identity score with shuffled IDs", "Kaggle example request IDs", "missing omics preserved"],
                  note="Infrastructure checks only; no model was trained and no CV performance is claimed.")
    (output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="amp-parkinsons-disease-progression-prediction")
    parser.add_argument("--output-dir", default="artifacts/shared")
    args = parser.parse_args()
    verify(args.data_dir, args.output_dir)

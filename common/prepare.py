"""Run once to freeze samples and folds; subsequent runs verify identical inputs."""

import argparse
import csv
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from . import HORIZONS, N_SPLITS, PROTOCOL, SEED, TARGETS
from .data import FILES, load_raw, make_samples
from .experiments import FIELDS
from .validation import attach_folds, make_patient_folds


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_dir, output_dir):
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    specification = dict(protocol=PROTOCOL, seed=SEED, n_splits=N_SPLITS,
                         targets=TARGETS, horizons=HORIZONS,
                         raw_sha256={name: sha256(data_dir / name) for name in FILES.values()})
    split_id = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["split_id"] != split_id:
            raise ValueError("Frozen inputs/protocol differ; use a new output directory")
        for name, digest in manifest["output_sha256"].items():
            if sha256(output_dir / name) != digest:
                raise ValueError(f"Frozen output was modified: {name}")
        print(f"Verified frozen split {split_id}")
        return manifest
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["samples.csv", "patient_folds.csv", "fold_summary.csv"]:
        if (output_dir / name).exists():
            raise ValueError(f"Unmanifested output exists: {name}; use a clean directory")
    raw = load_raw(data_dir)
    patient_folds = make_patient_folds(raw)
    samples = attach_folds(make_samples(raw), patient_folds)
    summary = samples.groupby("fold").agg(n_samples=("prediction_id", "size"),
                                           n_patients=("patient_id", "nunique"))
    patient_folds.to_csv(output_dir / "patient_folds.csv", index=False)
    samples.to_csv(output_dir / "samples.csv", index=False)
    summary.to_csv(output_dir / "fold_summary.csv")
    manifest = dict(specification, split_id=split_id, n_samples=len(samples),
                    n_patients=len(patient_folds),
                    environment=dict(python=platform.python_version(),
                                     **{p: version(p) for p in ["numpy", "pandas", "scikit-learn", "filelock"]}),
                    output_sha256={name: sha256(output_dir / name)
                                   for name in ["samples.csv", "patient_folds.csv", "fold_summary.csv"]})
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log_path = output_dir / "experiment_results.csv"
    if not log_path.exists():
        with log_path.open("x", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(FIELDS)
    print(f"Frozen {len(patient_folds)} patients / {len(samples)} labelled predictions: {split_id}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="amp-parkinsons-disease-progression-prediction")
    parser.add_argument("--output-dir", default="artifacts/shared")
    args = parser.parse_args()
    prepare(args.data_dir, args.output_dir)

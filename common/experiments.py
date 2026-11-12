"""Append-only, locked experiment records; every record references a frozen split."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock

FIELDS = ["experiment_id", "created_utc", "model", "features", "fold_scores", "mean_score",
          "std_score", "oof_score", "seed", "runtime_seconds", "split_id", "config", "notes"]


def log_experiment(path, *, experiment_id, model, features, evaluation, seed,
                   runtime_seconds, split_id, config, notes=""):
    scores = evaluation["fold_scores"]
    if sorted(map(int, scores)) != list(range(5)):
        raise ValueError("A comparable experiment must contain all five folds")
    values = list(scores.values())
    if not np.isfinite(values).all() or not np.isfinite(runtime_seconds) or runtime_seconds < 0:
        raise ValueError("Invalid scores or runtime")
    if not np.isclose(evaluation["mean_score"], np.mean(values)) or not np.isclose(evaluation["std_score"], np.std(values)):
        raise ValueError("Summary disagrees with fold scores")
    if not np.isfinite(evaluation["oof_score"]):
        raise ValueError("Invalid OOF score")
    if not all(isinstance(v, str) and v.strip() for v in [experiment_id, model, features, split_id]):
        raise ValueError("Experiment ID, model, features, and split ID are required")
    row = dict(experiment_id=experiment_id, created_utc=datetime.now(timezone.utc).isoformat(),
               model=model, features=features, fold_scores=json.dumps(scores, sort_keys=True),
               mean_score=evaluation["mean_score"], std_score=evaluation["std_score"],
               oof_score=evaluation["oof_score"], seed=seed, runtime_seconds=runtime_seconds,
               split_id=split_id, config=json.dumps(config, sort_keys=True, allow_nan=False), notes=notes)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        existing = []
        if path.exists() and path.stat().st_size:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != FIELDS:
                    raise ValueError("Experiment log schema differs")
                existing = list(reader)
        if any(r["experiment_id"] == experiment_id for r in existing):
            raise ValueError("Experiment ID already exists")
        if any(r["split_id"] != split_id for r in existing):
            raise ValueError("Use a separate experiment log for a different split")
        with path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

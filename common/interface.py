"""The common entry point used by all three model branches."""

from dataclasses import dataclass, fields

import pandas as pd

from . import N_SPLITS
from .data import KEYS, RawData, visit_features
from .validation import iter_folds, supplemental_for_fold


@dataclass
class FoldData:
    fold: int
    train_samples: pd.DataFrame
    valid_samples: pd.DataFrame
    train_features: pd.DataFrame
    valid_features: pd.DataFrame
    supplemental: pd.DataFrame


def model_folds(raw, samples, patient_folds):
    """Keep labels separate; learn omics vocabulary using training patients only.

    Feature rows align one-to-one and in order with the corresponding sample rows.
    Retained patient_id/visit_id are join keys, not model features.
    """
    assignments = samples[["patient_id", "fold"]].drop_duplicates().merge(
        patient_folds, on="patient_id", how="left", validate="many_to_one", suffixes=("", "_frozen")
    )
    if not assignments.fold.eq(assignments.fold_frozen).all():
        raise ValueError("Sample folds disagree with frozen patient assignments")
    if set(samples.fold) != set(range(N_SPLITS)):
        raise ValueError("Model interface requires all five frozen folds")
    for fold, train, valid in iter_folds(samples):
        train_ids = set(patient_folds.loc[patient_folds.fold != fold, "patient_id"])
        train_raw = RawData(**{
            field.name: getattr(raw, field.name).loc[getattr(raw, field.name).patient_id.isin(train_ids)]
            for field in fields(RawData)
        })
        train_visits = visit_features(train[KEYS], train_raw)
        valid_visits = visit_features(
            valid[KEYS], raw,
            protein_columns=[c for c in train_visits if c.startswith("protein::")],
            peptide_columns=[c for c in train_visits if c.startswith("peptide::")],
        )
        def expand(requests, visits):
            # Only explicit forecast metadata and contemporaneous measurements.
            return requests.drop(columns=["rating", "fold"]).merge(
                visits, on=KEYS, how="left", validate="many_to_one", sort=False
            )
        yield FoldData(fold, train, valid, expand(train, train_visits),
                       expand(valid, valid_visits), supplemental_for_fold(raw, patient_folds, fold))

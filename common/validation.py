"""Deterministic patient-only folds, independent of source row order."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from . import N_SPLITS, SEED


def make_patient_folds(raw, n_splits=N_SPLITS, seed=SEED):
    # Main-cohort patients define balanced validation groups. Supplemental patients
    # never enter validation and are selected separately with a patient exclusion.
    patients = np.sort(np.unique(np.concatenate([
        raw.clinical.patient_id, raw.proteins.patient_id, raw.peptides.patient_id,
    ])))
    folds = np.full(len(patients), -1, dtype=int)
    for fold, (_, val) in enumerate(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(patients)):
        folds[val] = fold
    return pd.DataFrame({"patient_id": patients, "fold": folds})


def attach_folds(samples, patient_folds):
    if patient_folds.patient_id.duplicated().any() or patient_folds.fold.isna().any():
        raise ValueError("Each patient needs exactly one fold")
    result = samples.merge(patient_folds, on="patient_id", how="left", validate="many_to_one")
    if result.fold.isna().any():
        raise ValueError("Some patients have no frozen fold")
    return result


def iter_folds(samples):
    """Yield (fold, training rows, validation rows), retaining original indices."""
    if samples.fold.isna().any() or samples.groupby("patient_id").fold.nunique().gt(1).any():
        raise ValueError("Invalid patient folds")
    for fold in sorted(samples.fold.unique()):
        train, valid = samples.loc[samples.fold != fold].copy(), samples.loc[samples.fold == fold].copy()
        if set(train.patient_id) & set(valid.patient_id):
            raise ValueError("Patient leakage")
        yield int(fold), train, valid


def supplemental_for_fold(raw, patient_folds, fold):
    """Optional training-only data; never use held-out patients' supplemental labels."""
    held_out = patient_folds.loc[patient_folds.fold == fold, "patient_id"]
    return raw.supplemental.loc[~raw.supplemental.patient_id.isin(held_out)].copy()

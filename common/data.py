"""Validated raw tables, forecast labels, and label-free current-visit features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import HORIZONS, TARGETS

KEYS = ["visit_id", "patient_id", "visit_month"]
MEDICATION = "upd23b_clinical_state_on_medication"
FILES = {
    "clinical": "train_clinical_data.csv",
    "proteins": "train_proteins.csv",
    "peptides": "train_peptides.csv",
    "supplemental": "supplemental_clinical_data.csv",
}


@dataclass
class RawData:
    clinical: pd.DataFrame
    proteins: pd.DataFrame
    peptides: pd.DataFrame
    supplemental: pd.DataFrame


def validate_table(frame, kind):
    """Fail on ambiguous keys/invalid values; preserve genuine missing measurements."""
    extra = {
        "clinical": [*TARGETS, MEDICATION],
        "supplemental": [*TARGETS, MEDICATION],
        "proteins": ["UniProt", "NPX"],
        "peptides": ["UniProt", "Peptide", "PeptideAbundance"],
    }[kind]
    missing = set(KEYS + extra) - set(frame.columns)
    if missing:
        raise ValueError(f"{kind}: missing columns {sorted(missing)}")
    frame = frame[KEYS + extra].copy()
    for col in ("patient_id", "visit_month"):
        values = pd.to_numeric(frame[col], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError(f"{kind}: invalid {col}")
        frame[col] = values.astype("int64")
    expected = frame.patient_id.astype(str) + "_" + frame.visit_month.astype(str)
    if not frame.visit_id.eq(expected).all():
        raise ValueError(f"{kind}: visit_id disagrees with patient/month")
    identifiers = ["UniProt", "Peptide"] if kind == "peptides" else (["UniProt"] if kind == "proteins" else [])
    if frame[KEYS + identifiers].isna().any().any():
        raise ValueError(f"{kind}: missing key")
    if frame.duplicated(["visit_id", *identifiers]).any():
        raise ValueError(f"{kind}: duplicate observation key")
    numeric = ["NPX"] if kind == "proteins" else (["PeptideAbundance"] if kind == "peptides" else list(TARGETS))
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
        observed = frame[col].dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError(f"{kind}: invalid {col}")
    return frame.sort_values(KEYS + identifiers).reset_index(drop=True)


def load_raw(data_dir):
    path = Path(data_dir)
    return RawData(**{
        kind: validate_table(pd.read_csv(path / filename), kind)
        for kind, filename in FILES.items()
    })


def prediction_requests(visits):
    """One row per (origin visit, horizon, target), with Kaggle prediction IDs."""
    visits = visits[KEYS].drop_duplicates().copy()
    if visits.visit_id.duplicated().any():
        raise ValueError("Conflicting visit metadata")
    grid = pd.MultiIndex.from_product([HORIZONS, TARGETS], names=["horizon", "target"]).to_frame(index=False)
    requests = visits.merge(grid, how="cross")
    requests["target_month"] = requests.visit_month + requests.horizon
    requests["prediction_id"] = (
        requests.visit_id + "_" + requests.target + "_plus_" + requests.horizon.astype(str) + "_months"
    )
    return requests.sort_values("prediction_id").reset_index(drop=True)


def make_samples(raw):
    """Omics-visit origins; exact future-month matching; no interpolation of labels."""
    origins = pd.concat([raw.proteins[KEYS], raw.peptides[KEYS]]).drop_duplicates()
    labels = raw.clinical.melt(id_vars=["patient_id", "visit_month"], value_vars=list(TARGETS),
                               var_name="target", value_name="rating")
    labels = labels.rename(columns={"visit_month": "target_month"}).dropna(subset=["rating"])
    return prediction_requests(origins).merge(
        labels, on=["patient_id", "target_month", "target"], how="inner", validate="many_to_one"
    ).sort_values("prediction_id").reset_index(drop=True)


def visit_features(visits, raw, *, protein_columns=None, peptide_columns=None):
    """Only contemporaneous omics. Pass training-fold columns when transforming validation.

    None selects the vocabulary in the provided raw tables; for a fold, supply raw
    tables filtered to training patients to learn columns, then freeze those columns.
    No clinical scores, medication state, imputation, scaling, or future aggregation.
    """
    features = visits[KEYS].drop_duplicates().copy()
    if features.visit_id.duplicated().any():
        raise ValueError("Conflicting visit metadata")
    for frame, keys, value, prefix, columns in [
        (raw.proteins, ["UniProt"], "NPX", "protein::", protein_columns),
        (raw.peptides, ["UniProt", "Peptide"], "PeptideAbundance", "peptide::", peptide_columns),
    ]:
        wide = frame.pivot(index="visit_id", columns=keys, values=value)
        wide.columns = [prefix + ("::".join(c) if isinstance(c, tuple) else c) for c in wide.columns]
        if columns is not None:
            wide = wide.reindex(columns=columns)
        features = features.merge(wide, on="visit_id", how="left", validate="one_to_one")
    return features.sort_values("visit_id").reset_index(drop=True)

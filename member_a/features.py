"""Label-free features with training-fold vocabulary and explicit time boundaries."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

TIME_COLUMNS = ["visit_month", "horizon", "target_month"]


def molecular_summaries(raw, kind, vocabulary):
    """Summarize only recognized analytes within each observed sampling visit."""
    if kind == "protein":
        allowed = {c.removeprefix("protein::") for c in vocabulary}
        frame = raw.proteins.loc[raw.proteins.UniProt.isin(allowed)]
        abundance = "NPX"
    else:
        allowed = {c.removeprefix("peptide::") for c in vocabulary}
        ids = raw.peptides.UniProt + "::" + raw.peptides.Peptide
        frame = raw.peptides.loc[ids.isin(allowed)]
        abundance = "PeptideAbundance"
    value = np.log1p(frame[abundance].to_numpy(float))
    z = frame[["patient_id", "visit_month"]].assign(value=value)
    grouped = z.groupby(["patient_id", "visit_month"]).value
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"])
    summary["iqr"] = grouped.quantile(.75) - grouped.quantile(.25)
    # A single measurement has zero spread. All-missing summaries remain missing.
    summary.loc[summary['count'] == 1, 'std'] = 0
    summary = summary.sort_index().reset_index()
    summary = summary.rename(columns={c: kind + "_" + c for c in summary if c not in ["patient_id", "visit_month"]})
    for key in ["mean", "median", "count"]:
        column = kind + "_" + key
        group = summary.groupby("patient_id", sort=False)[column]
        summary[f"hist_{kind}_{key}_delta"] = group.diff()
        summary[f"hist_{kind}_{key}_change_first"] = summary[column] - group.transform("first")
        summary[f"hist_{kind}_{key}_running_mean"] = group.expanding().mean().reset_index(level=0, drop=True).sort_index()
    summary[f"hist_{kind}_sample_count"] = summary.groupby("patient_id").cumcount() + 1
    summary[f"hist_{kind}_span"] = summary.visit_month - summary.groupby("patient_id").visit_month.transform("first")
    return summary


def feature_frame(requests, raw, protein_columns, peptide_columns):
    """Return 39 fixed columns, in request order. Never read clinical labels/schedule."""
    requests = requests.reset_index(drop=True)
    result = requests[TIME_COLUMNS].astype(float).copy()
    for kind, vocabulary in [("protein", protein_columns), ("peptide", peptide_columns)]:
        summary = molecular_summaries(raw, kind, vocabulary)
        current_columns = [c for c in summary if c.startswith(kind + "_")]
        history_columns = [c for c in summary if c.startswith("hist_")]
        # Contemporaneous values follow common.interface: no implicit last-value fill.
        current = requests[["patient_id", "visit_month"]].merge(
            summary[["patient_id", "visit_month", *current_columns]],
            on=["patient_id", "visit_month"], how="left", sort=False, validate="many_to_one")
        for c in current_columns:
            result[c] = current[c].to_numpy(float)
        history = np.full((len(requests), len(history_columns)), np.nan)
        for patient, rows in requests.groupby("patient_id", sort=False):
            seen = summary.loc[summary.patient_id == patient].sort_values("visit_month")
            positions = np.searchsorted(seen.visit_month.to_numpy(), rows.visit_month.to_numpy(), side="right") - 1
            valid = positions >= 0
            if valid.any():
                history[rows.index.to_numpy()[valid]] = seen[history_columns].to_numpy(float)[positions[valid]]
        for j, c in enumerate(history_columns):
            result[c] = history[:, j]
    if np.isinf(result.to_numpy(float)).any():
        raise ValueError("Infinite molecular feature")
    return result


def selected_columns(frame, experiment):
    columns = list(TIME_COLUMNS)
    if experiment != "A0":
        columns += [c for c in frame if c not in TIME_COLUMNS and not c.startswith("hist_")]
    if experiment in ["A2", "A3"]:
        columns += [c for c in frame if c.startswith("hist_")]
    return columns


def fit_protein_pca(train_features):
    """Fit imputation, scaling and 16 PCs on distinct training visits only."""
    unique = train_features.drop_duplicates("visit_id")
    columns = [c for c in unique if c.startswith("protein::") and unique[c].notna().any()]
    imputer, scaler = SimpleImputer(strategy="median"), StandardScaler()
    z = np.log1p(unique[columns].to_numpy(float))
    z = scaler.fit_transform(imputer.fit_transform(z))
    n = min(16, z.shape[0] - 1, z.shape[1])
    if n < 1:
        raise ValueError("Insufficient training protein observations for PCA")
    pca = PCA(n_components=n, svd_solver="full", random_state=4486).fit(z)
    return dict(columns=columns, imputer=imputer, scaler=scaler, pca=pca,
                fit_visit_ids=unique.visit_id.tolist())


def transform_protein_pca(features, bundle):
    z = np.log1p(features[bundle['columns']].to_numpy(float))
    z = bundle['scaler'].transform(bundle['imputer'].transform(z)) - bundle['pca'].mean_
    z = np.einsum('ij,kj->ik', z, bundle['pca'].components_, optimize=False)
    if not np.isfinite(z).all():
        raise ValueError("Non-finite PCA projection")
    return z


def predict_bundle(bundle, feature_values, contemporaneous_features):
    x = feature_values[bundle['columns']].to_numpy(float)
    if bundle['pca'] is not None:
        x = np.column_stack([x, transform_protein_pca(contemporaneous_features, bundle['pca'])])
    predictions = np.full(len(x), np.nan)
    for target, model in bundle['models'].items():
        mask = contemporaneous_features.target.eq(target).to_numpy()
        predictions[mask] = np.maximum(0, model.predict(x[mask]))
    if not np.isfinite(predictions).all():
        raise ValueError("Every request must have a finite prediction")
    return predictions

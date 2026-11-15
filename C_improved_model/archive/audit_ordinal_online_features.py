"""Audit whether historical features require visits unavailable from competition API.

Offline check, NO training and NO Kaggle submission:
    python C_improved_model/audit_ordinal_online_features.py

The API gives test requests and batch omics, not the hidden patients'
full clinical visit table. The omics-only replay is a conservative proxy,
NOT an exact reconstruction of the test stream.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from visit_history_model import KEYS, read_data, training_rows
from competition_optimized_experiment import make_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "amp-parkinsons-disease-progression-prediction"
OUT = ROOT / "C_improved_model/ordinal_results"

def main():
    clinical, omics = read_data(DATA)
    proteins = pd.read_csv(DATA / "train_proteins.csv", usecols=KEYS + ["UniProt", "NPX"])
    rows = training_rows(clinical, omics)
    observed = make_features(rows, clinical, omics, proteins).reset_index(drop=True)
    # Replay only omics-origin visits as an under-approximation of API-visible
    # test visits; actual API test may reveal additional visits.
    empty = pd.DataFrame(columns=KEYS)
    replay = make_features(rows, empty, omics, proteins).reset_index(drop=True)
    cols = [c for c in observed if c in replay.columns]
    differences = observed[cols].ne(replay[cols])
    result = []
    for col in cols:
        if col in ("visit_month", "horizon", "target_month"):
            continue
        result.append({
            "feature": col,
            "different_requests": int(differences[col].sum()),
            "different_fraction": float(differences[col].mean()),
            "uses_clinical_visits_offline": col in ("seen_6", "seen_18", "seen_48", "seen_6_18_count")
        })
    output = pd.DataFrame(result).sort_values("different_fraction", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUT / "ordinal_online_visibility_audit.csv", index=False)
    print(output.to_string(index=False))
    suspect = output.loc[output.uses_clinical_visits_offline &
                         output.different_requests.gt(0)]
    if len(suspect):
        print("AUDIT RESULT: exact offline feature parity NOT established. "
              "Clinical visit history affects the model features but is "
              "not delivered as labels/visits by the competition API. "
              "Do not treat the offline CV score as an online guarantee.")
    else:
        print("No differences under this conservative proxy; this alone "
              "does NOT establish exact API feature parity.")
    print("Required before Kaggle: implement streaming state from each "
          "iter_test batch and verify against the original API demo. "
          "Never prepopulate hidden patient history from full clinical data.")

if __name__ == "__main__":
    main()

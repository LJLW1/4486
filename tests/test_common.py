import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from common import TARGETS
from common.data import FILES, KEYS, MEDICATION, RawData, make_samples, prediction_requests, validate_table, visit_features
from common.experiments import log_experiment
from common.interface import model_folds
from common.metric import evaluate_oof, smape_plus_one
from common.prepare import prepare
from common.validation import attach_folds, iter_folds, make_patient_folds, supplemental_for_fold


def fixture():
    clinical, proteins, peptides = [], [], []
    for patient in range(10):
        for month in [0, 6, 12, 24, 30]:
            key = dict(visit_id=f"{patient}_{month}", patient_id=patient, visit_month=month)
            clinical.append(dict(key, **{t: float(patient + month) for t in TARGETS}, **{MEDICATION: "On"}))
            proteins.append(dict(key, UniProt="P1", NPX=float(month + 1)))
            peptides.append(dict(key, UniProt="P1", Peptide="SEQ", PeptideAbundance=float(month + 2)))
    clinical = pd.DataFrame(clinical)
    supplemental = clinical.loc[clinical.patient_id == 0].copy()  # Deliberate overlap exercises guard.
    return RawData(clinical, pd.DataFrame(proteins), pd.DataFrame(peptides), supplemental)


class MetricTests(unittest.TestCase):
    def test_shift_and_percent_scale(self):
        self.assertAlmostEqual(smape_plus_one([0, 1], [1, 3]), 200 * (1 / 3 + 2 / 6) / 2)
        self.assertEqual(smape_plus_one([0, 4], [0, 4]), 0)
        self.assertEqual(smape_plus_one([0], [-1]), 200)  # No silent clipping.

    def test_missing_labels_and_invalid_predictions(self):
        self.assertEqual(smape_plus_one([np.nan, 2], [np.nan, 2]), 0)
        for truth, pred in [([np.nan], [1]), ([1], [np.nan]), ([1], [np.inf]),
                            ([1], [1, 2]), ([], []), ([-1], [0]), ([np.inf], [1])]:
            with self.assertRaises(ValueError):
                smape_plus_one(truth, pred)

    def test_oof_alignment_and_weighting(self):
        samples = pd.DataFrame({"prediction_id": ["a", "b", "c"], "rating": [0, 0, 1], "fold": [0, 0, 1]})
        predictions = pd.DataFrame({"prediction_id": ["c", "b", "a"], "rating": [1, 1, 1]})
        score = evaluate_oof(samples, predictions)
        self.assertAlmostEqual(score["mean_score"], 100 / 3)
        self.assertAlmostEqual(score["oof_score"], 400 / 9)
        for bad in [predictions.iloc[:2], pd.concat([predictions, predictions.iloc[:1]]),
                    predictions.assign(prediction_id=["d", "b", "a"])]:
            with self.assertRaises(ValueError):
                evaluate_oof(samples, bad)


class DataTests(unittest.TestCase):
    def test_exact_horizon_and_missing_label(self):
        raw = fixture()
        raw.clinical.loc[(raw.clinical.visit_id == "0_6"), "updrs_4"] = np.nan
        samples = make_samples(raw)
        self.assertFalse(samples.prediction_id.eq("0_0_updrs_4_plus_6_months").any())
        row = samples.set_index("prediction_id").loc["0_6_updrs_1_plus_24_months"]
        self.assertEqual(row.rating, 30)
        self.assertEqual(row.target_month, 30)
        self.assertFalse(samples.prediction_id.str.contains("0_12_updrs_1_plus_6_months").any())
        self.assertFalse(samples.prediction_id.duplicated().any())

    def test_omics_origin_can_have_no_current_label(self):
        raw = fixture()
        raw.clinical = raw.clinical.loc[raw.clinical.visit_month != 0]
        samples = make_samples(raw)
        self.assertTrue(samples.prediction_id.eq("0_0_updrs_1_plus_6_months").any())
        self.assertFalse(samples.prediction_id.eq("0_0_updrs_1_plus_0_months").any())

    def test_no_future_or_label_features(self):
        raw = fixture()
        before = visit_features(raw.clinical.loc[raw.clinical.visit_month == 0], raw)
        raw.proteins.loc[raw.proteins.visit_month > 0, "NPX"] = 999999
        raw.clinical.loc[:, list(TARGETS)] = 999
        after = visit_features(raw.clinical.loc[raw.clinical.visit_month == 0], raw)
        pd.testing.assert_frame_equal(before, after)
        self.assertFalse(set(TARGETS + (MEDICATION,)) & set(before))

    def test_schema_rejects_duplicates_and_bad_keys(self):
        raw = fixture()
        for bad in [pd.concat([raw.clinical, raw.clinical.iloc[:1]]),
                    raw.clinical.assign(visit_id="bad"),
                    raw.clinical.assign(updrs_1=-1),
                    raw.clinical.drop(columns="updrs_1")]:
            with self.assertRaises(ValueError):
                validate_table(bad, "clinical")

    def test_requests_without_omics(self):
        raw = fixture()
        visits = pd.DataFrame([dict(visit_id="99_0", patient_id=99, visit_month=0)])
        self.assertEqual(len(prediction_requests(visits)), 16)
        features = visit_features(visits, raw)
        self.assertTrue(features.filter(regex="::").isna().all().all())


class FoldTests(unittest.TestCase):
    def test_deterministic_patient_partition(self):
        raw = fixture()
        folds = make_patient_folds(raw)
        raw.clinical = raw.clinical.sample(frac=1, random_state=5)
        raw.proteins = raw.proteins.sample(frac=1, random_state=6)
        pd.testing.assert_frame_equal(folds, make_patient_folds(raw))
        samples = attach_folds(make_samples(raw), folds)
        seen = []
        for fold, train, valid in iter_folds(samples):
            self.assertFalse(set(train.patient_id) & set(valid.patient_id))
            self.assertFalse(set(valid.patient_id) & set(supplemental_for_fold(raw, folds, fold).patient_id))
            seen.extend(valid.prediction_id)
        self.assertEqual(sorted(seen), sorted(samples.prediction_id))

    def test_missing_and_duplicate_assignments_fail(self):
        raw = fixture()
        folds = make_patient_folds(raw)
        samples = make_samples(raw)
        for bad in [folds.iloc[1:], pd.concat([folds, folds.iloc[:1]])]:
            with self.assertRaises(ValueError):
                attach_folds(samples, bad)

    def test_fold_vocabulary_ignores_validation_only_protein(self):
        raw = fixture()
        folds = make_patient_folds(raw)
        patient = folds.loc[folds.fold == 0, "patient_id"].iloc[0]
        extra = raw.proteins.loc[raw.proteins.patient_id == patient].assign(UniProt="VALID_ONLY")
        raw.proteins = pd.concat([raw.proteins, extra])
        samples = attach_folds(make_samples(raw), folds)
        fold = next(model_folds(raw, samples, folds))
        self.assertNotIn("protein::VALID_ONLY", fold.valid_features)
        self.assertEqual(list(fold.train_features), list(fold.valid_features))
        self.assertEqual(list(fold.valid_features.prediction_id), list(fold.valid_samples.prediction_id))
        self.assertNotIn("rating", fold.valid_features)

    def test_interface_rejects_different_patient_assignment(self):
        raw = fixture()
        folds = make_patient_folds(raw)
        samples = attach_folds(make_samples(raw), folds)
        changed = folds.assign(fold=(folds.fold + 1) % 5)
        with self.assertRaises(ValueError):
            next(model_folds(raw, samples, changed))


class PersistenceTests(unittest.TestCase):
    def test_frozen_inputs_and_outputs_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = fixture()
            for kind, filename in FILES.items():
                getattr(raw, kind).to_csv(root / filename, index=False)
            output = root / "shared"
            first = prepare(root, output)
            self.assertEqual(first["split_id"], prepare(root, output)["split_id"])
            with (output / "samples.csv").open("a") as stream:
                stream.write("corruption\n")
            with self.assertRaises(ValueError):
                prepare(root, output)
            raw.clinical.assign(updrs_1=0).to_csv(root / FILES["clinical"], index=False)
            with self.assertRaises(ValueError):
                prepare(root, output)

    def test_log_completeness_and_no_mixed_splits(self):
        evaluation = dict(fold_scores={i: float(i) for i in range(5)}, mean_score=2.0,
                          std_score=float(np.std(range(5))), oof_score=2.1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.csv"
            kwargs = dict(model="test", features="test", evaluation=evaluation, seed=4486,
                          runtime_seconds=0.1, split_id="split_a", config={"x": 1}, notes="测试, notes")
            log_experiment(path, experiment_id="one", **kwargs)
            with self.assertRaises(ValueError):
                log_experiment(path, experiment_id="one", **kwargs)
            with self.assertRaises(ValueError):
                log_experiment(path, experiment_id="two", **dict(kwargs, split_id="split_b"))
            self.assertEqual(json.loads(pd.read_csv(path).iloc[0].config), {"x": 1})


if __name__ == "__main__":
    unittest.main()

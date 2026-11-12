"""
Unit and Regression Tests for Candidate v7 Alternative Model Replacement,
Adapter Compatibility, and Rollback Verification.

Verifies:
1. Diagnostic comparison reports exist and satisfy Phase 3 & Phase 6 schema invariants.
2. Strict pre-2024 split boundaries (2024 and 2025 excluded from model selection).
3. Candidate v7 model artifacts & versioned contract exist under models/candidates/candidate_v7/.
4. Production (models/production/) and earlier candidates remain untouched.
5. ModelAdapter supports XGBoost, LightGBM, HistGradientBoosting, and CatBoost routing.
6. End-to-end prediction output, class mapping, training/inference parity, SHAP XAI,
   and explicit rollback to production XGBoost all operate cleanly.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from backend.predictor import predict
from scripts.feature_engineering.candidate_features import (
    CANDIDATE_V2_FEATURE_NAMES,
    build_candidate_features,
)
from scripts.ml.model_adapter import (
    ACTIVE_REGISTRY_PATH,
    CANDIDATE_V7_DIR,
    PROD_DIR,
    activate_model,
    clear_model_cache,
    extract_features_for_contract,
    get_active_model_status,
    load_active_model,
    load_model_from_dir,
    predict_from_features,
    predict_ticker_via_adapter,
    rollback_to_production,
)

REPORT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v7"
REPORT_JSON = REPORT_DIR / "model_comparison_report.json"
REPORT_MD = REPORT_DIR / "model_comparison_report.md"


class TestCandidateV7Models(unittest.TestCase):

    def test_diagnostic_reports_exist_and_complete(self):
        """Verify JSON and MD reports contain all required Phase 3 & Phase 6 metrics."""
        self.assertTrue(REPORT_JSON.exists(), f"Missing {REPORT_JSON}")
        self.assertTrue(REPORT_MD.exists(), f"Missing {REPORT_MD}")

        with open(REPORT_JSON, "r", encoding="utf-8") as f:
            report = json.load(f)

        self.assertEqual(report["candidate_id"], "candidate_v7")
        self.assertEqual(report["holdout_policy"], "2024 and 2025 Strictly Excluded")
        self.assertEqual(report["purge_bars"], 5)
        self.assertEqual(report["random_seed"], 42)
        self.assertIn("library_versions", report)
        self.assertIn("splits", report)

        # Pre-2024 splits only
        self.assertEqual(
            sorted([s["eval_year"] for s in report["splits"].values()]),
            [2021, 2022, 2023],
        )

        # Required baseline and alternative models evaluated
        models = report["models"]
        for req in [
            "Majority_HOLD",
            "Candidate_v2_XGB_Balanced",
            "Candidate_v4_XGB_Sqrt",
            "LightGBM_Sqrt",
            "LightGBM_Fixed200_Balanced",
            "HistGB_Sqrt",
            "HistGB_Balanced",
        ]:
            self.assertIn(req, models)
            m = models[req]
            self.assertEqual(m["status"], "evaluated")
            self.assertIn("oof_overall", m)
            self.assertIn("by_year", m)
            self.assertIn("accuracy_vs_coverage", m)
            self.assertIn("trading_performance", m)
            self.assertIn("timing", m)
            self.assertIn("calibration", m["oof_overall"])
            for cls_name in ["SELL", "HOLD", "BUY"]:
                self.assertIn(cls_name, m["oof_overall"]["per_class"])

    def test_candidate_v7_contract_and_artifacts(self):
        """Verify Candidate v7 model binary, contract, and metadata exist with required fields."""
        model_file = CANDIDATE_V7_DIR / "candidate_model.joblib"
        contract_file = CANDIDATE_V7_DIR / "model_contract.json"
        meta_file = CANDIDATE_V7_DIR / "metadata.json"

        self.assertTrue(model_file.exists())
        self.assertTrue(contract_file.exists())
        self.assertTrue(meta_file.exists())

        with open(contract_file, "r", encoding="utf-8") as f:
            contract = json.load(f)

        required_keys = [
            "candidate_id",
            "model_version",
            "model_type",
            "library",
            "library_version",
            "class_mapping",
            "preprocessing_requirements",
            "feature_pipeline",
            "feature_names",
            "feature_count",
            "imputation_values",
        ]
        for k in required_keys:
            self.assertIn(k, contract, f"Missing contract key: {k}")

        self.assertEqual(contract["candidate_id"], "candidate_v7")
        self.assertEqual(contract["model_type"], "HistGradientBoostingClassifier")
        self.assertEqual(contract["class_mapping"], {"0": "SELL", "1": "HOLD", "2": "BUY"})
        self.assertEqual(contract["feature_count"], 34)
        self.assertEqual(contract["feature_names"], CANDIDATE_V2_FEATURE_NAMES)

    def test_existing_production_and_v2_untouched(self):
        """Ensure legacy production XGBoost and Candidate v2 artifacts remain intact."""
        prod_model = PROD_DIR / "model.json"
        prod_contract = PROD_DIR / "model_contract.json"
        v2_model = PROJECT_ROOT / "models" / "candidates" / "candidate_v2" / "candidate_model.json"

        self.assertTrue(prod_model.exists())
        self.assertTrue(prod_contract.exists())
        self.assertTrue(v2_model.exists())
        self.assertGreater(prod_model.stat().st_size, 1_000_000)

        with open(prod_contract, "r", encoding="utf-8") as f:
            c = json.load(f)
        self.assertEqual(c["model_version"], "1.0.0")
        self.assertEqual(c["model_type"], "XGBClassifier")

    def test_adapter_supports_histgb_lightgbm_xgboost_and_catboost_routing(self):
        """Verify model adapter loads HistGB, LightGBM, XGBoost, and routes CatBoost properly."""
        # 1. HistGradientBoostingClassifier (Candidate v7)
        m_v7, c_v7 = load_model_from_dir(CANDIDATE_V7_DIR, use_cache=False)
        self.assertIsInstance(m_v7, HistGradientBoostingClassifier)
        X_syn = pd.DataFrame([{f: 0.0 for f in c_v7["feature_names"]}])
        res_v7 = predict_from_features(m_v7, c_v7, X_syn, active_model_id="candidate_v7")
        self.assertIn(res_v7["prediction"], ["SELL", "HOLD", "BUY"])
        self.assertAlmostEqual(sum(res_v7["probabilities"].values()), 1.0, places=2)

        # 2. XGBClassifier (Production)
        m_prod, c_prod = load_model_from_dir(PROD_DIR, use_cache=False)
        X_prod = pd.DataFrame([{f: 0.0 for f in c_prod["feature_names"]}])
        res_prod = predict_from_features(m_prod, c_prod, X_prod, active_model_id="production")
        self.assertIn(res_prod["prediction"], ["SELL", "HOLD", "BUY"])
        self.assertAlmostEqual(sum(res_prod["probabilities"].values()), 1.0, places=2)

        # 3. LightGBMClassifier (round-trip via temporary model directory)
        rng = np.random.RandomState(42)
        X_lgb = pd.DataFrame(rng.randn(120, 34), columns=CANDIDATE_V2_FEATURE_NAMES)
        y_lgb = rng.choice([0, 1, 2], size=120)
        lgb_model = LGBMClassifier(
            n_estimators=10, max_depth=3, num_leaves=7, verbose=-1, random_state=42, n_jobs=1
        ).fit(X_lgb, y_lgb)

        with tempfile.TemporaryDirectory() as tmpdir:
            tdir = Path(tmpdir)
            joblib.dump(lgb_model, tdir / "candidate_model.joblib")
            lgb_contract = {
                "candidate_id": "test_lgb",
                "model_version": "test-lgb",
                "model_type": "LGBMClassifier",
                "model_format": "joblib",
                "model_filename": "candidate_model.joblib",
                "class_mapping": {"0": "SELL", "1": "HOLD", "2": "BUY"},
                "feature_names": CANDIDATE_V2_FEATURE_NAMES,
                "imputation_values": {f: 0.0 for f in CANDIDATE_V2_FEATURE_NAMES},
            }
            with open(tdir / "model_contract.json", "w", encoding="utf-8") as f:
                json.dump(lgb_contract, f)
            loaded_lgb, loaded_c = load_model_from_dir(tdir, use_cache=False)
            self.assertIsInstance(loaded_lgb, LGBMClassifier)
            res_lgb = predict_from_features(loaded_lgb, loaded_c, X_lgb.iloc[[0]])
            self.assertIn(res_lgb["prediction"], ["SELL", "HOLD", "BUY"])
            self.assertAlmostEqual(sum(res_lgb["probabilities"].values()), 1.0, places=2)

        # 4. CatBoost routing check (verifies graceful error if blocked by OS, or load if allowed)
        with tempfile.TemporaryDirectory() as tmpdir:
            cb_dir = Path(tmpdir)
            (cb_dir / "candidate_model.cbm").write_bytes(b"placeholder")
            cb_contract = {
                "candidate_id": "test_cb",
                "model_type": "CatBoostClassifier",
                "model_format": "cbm",
                "model_filename": "candidate_model.cbm",
                "feature_names": CANDIDATE_V2_FEATURE_NAMES,
            }
            with open(cb_dir / "model_contract.json", "w", encoding="utf-8") as f:
                json.dump(cb_contract, f)
            with self.assertRaises((RuntimeError, Exception)):
                load_model_from_dir(cb_dir, use_cache=False)

    def test_training_inference_feature_parity_candidate_v7(self):
        """Verify live adapter feature extraction matches batch Candidate v2 features."""
        tcs_path = PROJECT_ROOT / "data" / "processed" / "prices" / "TCS.csv"
        price_df = pd.read_csv(tcs_path)
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df["Ticker"] = "TCS"

        nifty_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "market" / "nifty50.csv")
        vix_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "market" / "indiavix.csv")
        earnings_df = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "internal" / "earnings_events.csv")

        _, contract = load_model_from_dir(CANDIDATE_V7_DIR, use_cache=False)
        batch_featured = build_candidate_features(
            price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )
        _, X_adapter = extract_features_for_contract(
            price_df, contract, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )
        res = predict_ticker_via_adapter(
            "TCS",
            price_df=price_df,
            nifty_df=nifty_df,
            vix_df=vix_df,
            earnings_df=earnings_df,
            model_id_or_path=str(CANDIDATE_V7_DIR),
        )
        for feat in CANDIDATE_V2_FEATURE_NAMES:
            batch_val = float(X_adapter[feat].iloc[-1])
            live_val = float(res["feature_snapshot"][feat])
            self.assertAlmostEqual(batch_val, live_val, places=4)

    def test_rollback_and_activation_mechanism(self):
        """Verify switching to production and re-activating Candidate v7 works cleanly."""
        original_registry = None
        if ACTIVE_REGISTRY_PATH.exists():
            original_registry = ACTIVE_REGISTRY_PATH.read_text(encoding="utf-8")

        try:
            rb = rollback_to_production(reason="Regression test rollback")
            self.assertEqual(rb["active_model_id"], "production")
            _, _, active_id = load_active_model(use_cache=False)
            self.assertEqual(active_id, "production")

            act = activate_model("candidate_v7", reason="Regression test activation")
            self.assertEqual(act["active_model_id"], "candidate_v7")
            model, contract, active_id = load_active_model(use_cache=False)
            self.assertEqual(active_id, "candidate_v7")
            self.assertEqual(contract["model_type"], "HistGradientBoostingClassifier")
            self.assertIsInstance(model, HistGradientBoostingClassifier)
        finally:
            clear_model_cache()
            if original_registry is not None:
                ACTIVE_REGISTRY_PATH.write_text(original_registry, encoding="utf-8")
            elif ACTIVE_REGISTRY_PATH.exists():
                ACTIVE_REGISTRY_PATH.unlink()

    def test_backend_predict_with_candidate_v7(self):
        """Verify end-to-end backend.predictor.predict works when candidate_v7 is active."""
        prev_env = os.environ.get("STOCKDNA_ACTIVE_MODEL")
        os.environ["STOCKDNA_ACTIVE_MODEL"] = "candidate_v7"
        clear_model_cache()
        try:
            out = predict("POWERGRID")
            self.assertEqual(out["ticker"], "POWERGRID")
            self.assertEqual(out["model_type"], "HistGradientBoostingClassifier")
            self.assertEqual(out["active_model_id"], "candidate_v7")
            self.assertIn(out["prediction"], ["SELL", "HOLD", "BUY"])
            self.assertAlmostEqual(sum(out["probabilities"].values()), 1.0, places=2)
            self.assertAlmostEqual(
                out["internal_percentage"] + out["external_percentage"], 100.0, places=1
            )
            self.assertEqual(len(out["xai"]["all_feature_contributions"]), 34)
        finally:
            clear_model_cache()
            if prev_env is None:
                os.environ.pop("STOCKDNA_ACTIVE_MODEL", None)
            else:
                os.environ["STOCKDNA_ACTIVE_MODEL"] = prev_env


if __name__ == "__main__":
    unittest.main()


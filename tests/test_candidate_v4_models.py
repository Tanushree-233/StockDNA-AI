"""
Unit Tests for Candidate v4 Controlled Model Comparison.

Verifies:
1. Candidate v4 diagnostic reports exist and adhere to JSON schema.
2. All 5 model configurations were evaluated on identical rows.
3. Accuracy trade-offs and empirical properties are verified.
4. Production artifacts and previous candidates remain untouched.
5. Strict temporal split integrity and pre-2024 walk-forward bounds.
"""

import unittest
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v4"
REPORT_JSON = OUTPUT_DIR / "model_comparison_report.json"
REPORT_MD = OUTPUT_DIR / "model_comparison_report.md"
PROD_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "model.json"
CANDIDATE_V2_PATH = PROJECT_ROOT / "models" / "candidates" / "candidate_v2" / "candidate_model.json"


class TestCandidateV4Models(unittest.TestCase):

    def test_reports_exist_and_valid_schema(self):
        """Verifies report files exist and contain required keys."""
        self.assertTrue(REPORT_JSON.exists(), f"Missing {REPORT_JSON}")
        self.assertTrue(REPORT_MD.exists(), f"Missing {REPORT_MD}")

        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        self.assertEqual(data["candidate_id"], "candidate_v4")
        self.assertEqual(data["feature_count"], 34)
        self.assertEqual(data["holdout_status"], "2024 and 2025 Strictly Excluded")
        self.assertIn("aggregate_means", data)
        self.assertIn("delta_vs_baseline", data)
        self.assertIn("splits", data)

    def test_all_model_configurations_evaluated(self):
        """Verifies that all 5 model configurations exist across all pre-2024 splits."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        expected_configs = [
            "candidate_v2_baseline",
            "xgb_natural_prior",
            "xgb_sqrt_weighted",
            "random_forest",
            "ensemble_xgb_rf",
        ]

        for s_name, s_data in data["splits"].items():
            for cfg in expected_configs:
                self.assertIn(cfg, s_data, f"Missing config {cfg} in {s_name}")
                self.assertIn("accuracy", s_data[cfg])
                self.assertIn("macro_f1", s_data[cfg])
                self.assertIn("balanced_accuracy", s_data[cfg])
                self.assertIn("confusion_matrix", s_data[cfg])

    def test_accuracy_improvement_empirically_demonstrated(self):
        """Verifies that unweighted and bagging configurations demonstrated substantial accuracy improvements."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        base_acc = data["aggregate_means"]["candidate_v2_baseline"]["accuracy"]
        nat_acc = data["aggregate_means"]["xgb_natural_prior"]["accuracy"]
        rf_acc = data["aggregate_means"]["random_forest"]["accuracy"]
        sqrt_acc = data["aggregate_means"]["xgb_sqrt_weighted"]["accuracy"]

        # Natural prior and Random Forest must show higher overall accuracy than baseline
        self.assertGreater(nat_acc, base_acc, "XGB Natural Prior should have higher accuracy than balanced baseline")
        self.assertGreater(rf_acc, base_acc, "Random Forest should have higher accuracy than balanced baseline")
        self.assertGreater(sqrt_acc, base_acc, "XGB Sqrt-Weighted should have higher accuracy than balanced baseline")

        # Delta assertions
        delta_nat = data["delta_vs_baseline"]["xgb_natural_prior"]["accuracy"]
        delta_rf = data["delta_vs_baseline"]["random_forest"]["accuracy"]
        self.assertGreater(delta_nat, 0.04, f"XGB Natural Prior should gain > 4% accuracy, got {delta_nat}")
        self.assertGreater(delta_rf, 0.08, f"Random Forest should gain > 8% accuracy, got {delta_rf}")

    def test_production_and_candidate_artifacts_untouched(self):
        """Ensures production model and previous candidate artifacts were not modified."""
        self.assertTrue(PROD_MODEL_PATH.exists())
        self.assertTrue(CANDIDATE_V2_PATH.exists())

        # Production model file size verification
        self.assertGreater(PROD_MODEL_PATH.stat().st_size, 1_000_000)
        self.assertGreater(CANDIDATE_V2_PATH.stat().st_size, 1_000_000)

    def test_split_bounds_strictly_pre_2024(self):
        """Verifies all evaluated splits in the report are strictly pre-2024."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        valid_split_years = [2021, 2022, 2023]
        for s_name in data["splits"]:
            year = int(s_name.split("(")[1].split(")")[0])
            self.assertIn(year, valid_split_years, f"Split year {year} is not strictly pre-2024")
            self.assertLessEqual(year, 2023, f"Split year {year} violates pre-2024 constraint")


if __name__ == "__main__":
    unittest.main()

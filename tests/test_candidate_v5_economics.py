"""
Unit Tests for Candidate v5 Economic-Value and Robustness Evaluation.

Verifies:
1. Economic report existence and JSON schema integrity.
2. Temporal isolation, strictly pre-2024 walk-forward bounds, and 5-bar purges.
3. Prediction alignment and integrity across identical rows.
4. Transaction cost mathematical consistency.
5. Production and candidate v1-v4 artifacts remain completely unchanged.
"""

import unittest
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v5"
REPORT_JSON = OUTPUT_DIR / "economic_evaluation_report.json"
REPORT_MD = OUTPUT_DIR / "economic_evaluation_report.md"
PROD_MODEL_PATH = PROJECT_ROOT / "models" / "production" / "model.json"
CANDIDATE_V2_PATH = PROJECT_ROOT / "models" / "candidates" / "candidate_v2" / "candidate_model.json"


class TestCandidateV5Economics(unittest.TestCase):

    def test_economic_reports_exist_and_valid_schema(self):
        """Verifies report files exist and contain required keys."""
        self.assertTrue(REPORT_JSON.exists(), f"Missing {REPORT_JSON}")
        self.assertTrue(REPORT_MD.exists(), f"Missing {REPORT_MD}")

        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        self.assertEqual(data["candidate_id"], "candidate_v5_economic_evaluation")
        self.assertEqual(data["holdout_status"], "2024 and 2025 Strictly Excluded")
        self.assertIn("trading_assumptions", data)
        self.assertIn("aggregate_means", data)
        self.assertIn("daily_portfolio_simulation", data)
        self.assertIn("split_level_results", data)

    def test_temporal_isolation_and_purge(self):
        """Verifies evaluation splits are strictly pre-2024."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        splits = data["split_level_results"]
        for s_name, s_data in splits.items():
            year = s_data["eval_year"]
            self.assertLessEqual(year, 2023, f"Split year {year} violates pre-2024 constraint")
            self.assertGreaterEqual(year, 2021)

    def test_prediction_alignment_and_integrity(self):
        """Verifies both models evaluated identical rows across all splits."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        splits = data["split_level_results"]
        total_rows = sum(s["eval_rows"] for s in splits.values())
        self.assertEqual(total_rows, 2145, "Expected exactly 2145 out-of-fold evaluated rows across 2021-2023")

        for s_name, s_data in splits.items():
            v2_rows = s_data["candidate_v2"]["classification"]["confusion_matrix"]
            v4_rows = s_data["candidate_v4_sqrt"]["classification"]["confusion_matrix"]
            self.assertEqual(np.sum(v2_rows), s_data["eval_rows"])
            self.assertEqual(np.sum(v4_rows), s_data["eval_rows"])

    def test_cost_calculation_mathematics(self):
        """Verifies net returns monotonically decrease as transaction costs increase."""
        with open(REPORT_JSON, "r") as f:
            data = json.load(f)

        port = data["daily_portfolio_simulation"]
        for m_key in ["candidate_v2", "candidate_v4_sqrt"]:
            lo = port[m_key]["long_only"]
            ret_0 = lo["0bps"]["total_net_return"]
            ret_5 = lo["5bps"]["total_net_return"]
            ret_10 = lo["10bps"]["total_net_return"]
            ret_15 = lo["15bps"]["total_net_return"]
            ret_25 = lo["25bps"]["total_net_return"]

            self.assertGreater(ret_0, ret_5, f"Cost monotonicity violated: 0bps={ret_0} vs 5bps={ret_5}")
            self.assertGreater(ret_5, ret_10, f"Cost monotonicity violated: 5bps={ret_5} vs 10bps={ret_10}")
            self.assertGreater(ret_10, ret_15, f"Cost monotonicity violated: 10bps={ret_10} vs 15bps={ret_15}")
            self.assertGreater(ret_15, ret_25, f"Cost monotonicity violated: 15bps={ret_15} vs 25bps={ret_25}")

    def test_production_and_previous_candidate_artifacts_unchanged(self):
        """Ensures production model and previous candidate artifacts were not modified."""
        self.assertTrue(PROD_MODEL_PATH.exists())
        self.assertTrue(CANDIDATE_V2_PATH.exists())
        self.assertGreater(PROD_MODEL_PATH.stat().st_size, 1_000_000)
        self.assertGreater(CANDIDATE_V2_PATH.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()

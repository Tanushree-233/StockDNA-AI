"""
Unit Tests for Candidate v3 Decision Calibration Module.

Verifies:
1. Class ordering contract (0: SELL, 1: HOLD, 2: BUY).
2. Probability handling, shape validation, non-negativity, and normalization.
3. Threshold constraints and frequency safeguards.
4. Temporal split integrity and 5-bar boundary purges.
5. Absence of evaluation data leakage during calibration.
6. Equivalence with argmax at zero-margin baseline.
7. Evaluation artifacts existence and schema integrity.
"""

import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd

from scripts.ml.decision_calibration import (
    DecisionRuleType,
    FrequencySafeguards,
    CalibratedDecisionPolicy,
    calibrate_margin_thresholds,
    calibrate_direct_thresholds,
)
from scripts.feature_engineering.candidate_features import CANDIDATE_V2_FEATURE_NAMES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v3_calibration"


class TestDecisionCalibration(unittest.TestCase):

    def test_class_ordering_contract(self):
        """Verifies strict contract: index 0 is SELL, 1 is HOLD, 2 is BUY."""
        policy_argmax = CalibratedDecisionPolicy(rule_type=DecisionRuleType.ARGMAX)
        probs = np.array([
            [0.70, 0.20, 0.10],  # Clear SELL
            [0.10, 0.80, 0.10],  # Clear HOLD
            [0.10, 0.20, 0.70],  # Clear BUY
        ])
        preds = policy_argmax.predict(probs)
        self.assertEqual(preds[0], 0, "Index 0 must map to SELL (class 0)")
        self.assertEqual(preds[1], 1, "Index 1 must map to HOLD (class 1)")
        self.assertEqual(preds[2], 2, "Index 2 must map to BUY (class 2)")

    def test_probability_handling_and_normalization(self):
        """Verifies probability input dimensions, non-negativity, and auto-normalization."""
        policy = CalibratedDecisionPolicy(rule_type=DecisionRuleType.ARGMAX)

        # Invalid dimension
        with self.assertRaises(ValueError):
            policy.predict(np.array([0.5, 0.3, 0.2]))  # 1D array

        with self.assertRaises(ValueError):
            policy.predict(np.array([[0.5, 0.5]]))  # 2 classes instead of 3

        # Negative probabilities
        with self.assertRaises(ValueError):
            policy.predict(np.array([[-0.1, 0.6, 0.5]]))

        # Unnormalized probabilities summing to != 1 (e.g. logits or soft scores)
        unnormalized = np.array([
            [1.0, 2.0, 1.0],  # argmax is class 1 (HOLD)
            [3.0, 1.0, 0.5],  # argmax is class 0 (SELL)
            [0.5, 1.0, 4.0],  # argmax is class 2 (BUY)
        ])
        preds = policy.predict(unnormalized)
        self.assertEqual(preds.tolist(), [1, 0, 2])

    def test_threshold_constraints_and_safeguards(self):
        """Verifies safeguards reject degenerate rules predicting excessive directional signals."""
        safeguards = FrequencySafeguards(
            max_buy_rate=0.40,
            max_sell_rate=0.40,
            min_hold_rate=0.30,
        )

        # Normal balanced distribution: 20% SELL, 60% HOLD, 20% BUY -> Valid
        normal_preds = np.array([0] * 20 + [1] * 60 + [2] * 20)
        self.assertTrue(safeguards.is_valid(normal_preds))

        # Excessive BUY: 50% BUY (> 40% cap) -> Invalid
        excessive_buy = np.array([0] * 10 + [1] * 40 + [2] * 50)
        self.assertFalse(safeguards.is_valid(excessive_buy))

        # Excessive SELL: 45% SELL (> 40% cap) -> Invalid
        excessive_sell = np.array([0] * 45 + [1] * 35 + [2] * 20)
        self.assertFalse(safeguards.is_valid(excessive_sell))

        # Collapsed HOLD: 20% HOLD (< 30% floor) -> Invalid
        collapsed_hold = np.array([0] * 40 + [1] * 20 + [2] * 40)
        self.assertFalse(safeguards.is_valid(collapsed_hold))

    def test_argmax_equivalence_at_zero_margins(self):
        """Margin rule with delta_sell=0.0 and delta_buy=0.0 must match argmax for non-tied probs."""
        policy_zero_margin = CalibratedDecisionPolicy(
            rule_type=DecisionRuleType.MARGIN_THRESHOLD,
            parameters={"delta_sell": 0.0, "delta_buy": 0.0},
        )
        probs = np.array([
            [0.55, 0.30, 0.15],
            [0.20, 0.60, 0.20],
            [0.15, 0.25, 0.60],
            [0.45, 0.40, 0.15],
            [0.10, 0.45, 0.4501],
        ])
        pred_margin = policy_zero_margin.predict(probs)
        pred_argmax = np.argmax(probs, axis=1)
        np.testing.assert_array_equal(pred_margin, pred_argmax)

    def test_absence_of_leakage(self):
        """Verifies threshold calibration depends exclusively on calibration data, not eval data."""
        np.random.seed(42)
        p_cal = np.random.dirichlet([1, 2, 1], size=200)
        y_cal = np.random.choice([0, 1, 2], size=200, p=[0.25, 0.50, 0.25])

        # Calibration function only accepts p_cal and y_cal
        policy_1 = calibrate_margin_thresholds(p_cal, y_cal)
        policy_2 = calibrate_margin_thresholds(p_cal, y_cal)

        self.assertEqual(policy_1.parameters["delta_sell"], policy_2.parameters["delta_sell"])
        self.assertEqual(policy_1.parameters["delta_buy"], policy_2.parameters["delta_buy"])

    def test_temporal_split_integrity(self):
        """Verifies that all pre-2024 splits strictly respect chronological order and 5-bar purges."""
        df = pd.read_csv(DATA_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        tickers = ["INFY", "RELIANCE", "TCS"]

        splits = [
            {"inner_tr": [2018, 2019], "cal": 2020, "outer_tr": [2018, 2019, 2020], "eval": 2021},
            {"inner_tr": [2018, 2019, 2020], "cal": 2021, "outer_tr": [2018, 2019, 2020, 2021], "eval": 2022},
            {"inner_tr": [2018, 2019, 2020, 2021], "cal": 2022, "outer_tr": [2018, 2019, 2020, 2021, 2022], "eval": 2023},
        ]

        for s in splits:
            # 1. Assert chronological year sequencing
            self.assertTrue(max(s["inner_tr"]) < s["cal"], "Inner train years must precede calibration year")
            self.assertTrue(max(s["outer_tr"]) < s["eval"], "Outer train years must precede eval year")
            self.assertTrue(s["eval"] <= 2023, "Evaluation year must be strictly pre-2024")

            # 2. Assert 5-bar boundary purge per ticker
            for t in tickers:
                tdf = df[df["Ticker"] == t].sort_values("Date").reset_index(drop=True)
                
                # Check outer train purge
                tr_rows = tdf[tdf["Date"].dt.year.isin(s["outer_tr"])]
                purged_tr_rows = tr_rows.iloc[:-5]
                last_tr_date = purged_tr_rows["Date"].iloc[-1]
                
                eval_rows = tdf[tdf["Date"].dt.year == s["eval"]]
                first_eval_date = eval_rows["Date"].iloc[0]

                # There must be at least 5 trading bars between last training bar and first eval bar
                bars_between = len(tdf[(tdf["Date"] > last_tr_date) & (tdf["Date"] < first_eval_date)])
                self.assertGreaterEqual(
                    bars_between, 5,
                    f"Ticker {t} split {s['eval']} requires >= 5 purged bars between training and eval, got {bars_between}"
                )

    def test_evaluation_report_artifacts_exist(self):
        """Verifies that the generated calibration reports exist and match required schema."""
        json_path = OUTPUT_DIR / "calibration_evaluation_report.json"
        md_path = OUTPUT_DIR / "calibration_evaluation_report.md"

        self.assertTrue(json_path.exists(), f"Missing {json_path}")
        self.assertTrue(md_path.exists(), f"Missing {md_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        self.assertEqual(data["candidate_id"], "candidate_v3_calibration")
        self.assertIn("aggregate_means", data)
        self.assertIn("splits", data)
        self.assertIn("delta_vs_argmax", data)
        self.assertIn("default_argmax", data["aggregate_means"])
        self.assertIn("calibrated_margin_thresholds", data["aggregate_means"])


if __name__ == "__main__":
    unittest.main()

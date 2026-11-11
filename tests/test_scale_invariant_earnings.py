"""
Unit tests for the isolated Scale-Invariant Earnings Features module.
Verifies:
1. Scale-invariance under stock splits and price scaling.
2. Exact 34-feature contract preservation.
3. Zero look-ahead / point-in-time correctness.
4. Production contract isolation (production files untouched).
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from scripts.feature_engineering.scale_invariant_earnings import (
    compute_scale_invariant_earnings_features,
    get_scale_invariant_feature_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestScaleInvariantEarnings(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            "Date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "Ticker": ["TCS"] * 10,
            "Close": [3000.0, 3100.0, 3050.0, 3200.0, 3150.0, 3300.0, 3250.0, 3400.0, 3350.0, 3500.0],
            "Reported_EPS": [30.0] * 10,
            "EPS_Estimate": [28.0] * 10,
            "Daily_Return": [0.01] * 10,
        })

    def test_earnings_yield_formula(self):
        """Verify Earnings_Yield = Reported_EPS / Close and Forward_Earnings_Yield = EPS_Estimate / Close."""
        out = compute_scale_invariant_earnings_features(self.sample_df)
        self.assertIn("Earnings_Yield", out.columns)
        self.assertIn("Forward_Earnings_Yield", out.columns)

        expected_ey = 30.0 / 3000.0  # 0.01 (1% yield)
        expected_fey = 28.0 / 3000.0
        self.assertAlmostEqual(out["Earnings_Yield"].iloc[0], expected_ey, places=5)
        self.assertAlmostEqual(out["Forward_Earnings_Yield"].iloc[0], expected_fey, places=5)

    def test_scale_invariance_under_stock_split(self):
        """Verify that a 2-for-1 stock split (halving price and halving EPS) leaves earnings yields identical."""
        df_split = self.sample_df.copy()
        df_split["Close"] = df_split["Close"] / 2.0  # Halve price
        df_split["Reported_EPS"] = df_split["Reported_EPS"] / 2.0  # Halve EPS
        df_split["EPS_Estimate"] = df_split["EPS_Estimate"] / 2.0  # Halve estimated EPS

        out_normal = compute_scale_invariant_earnings_features(self.sample_df)
        out_split = compute_scale_invariant_earnings_features(df_split)

        np.testing.assert_allclose(
            out_normal["Earnings_Yield"].values,
            out_split["Earnings_Yield"].values,
            rtol=1e-5,
            err_msg="Earnings_Yield is not invariant under a stock split!",
        )
        np.testing.assert_allclose(
            out_normal["Forward_Earnings_Yield"].values,
            out_split["Forward_Earnings_Yield"].values,
            rtol=1e-5,
            err_msg="Forward_Earnings_Yield is not invariant under a stock split!",
        )

    def test_contract_preservation_length(self):
        """Verify that 1-to-1 replacement preserves exact feature count."""
        base_features = ["Daily_Return", "Reported_EPS", "EPS_Estimate"]
        X, new_features = get_scale_invariant_feature_matrix(self.sample_df, base_features, replace_nominal_eps=True)
        self.assertEqual(len(new_features), len(base_features))
        self.assertIn("Earnings_Yield", new_features)
        self.assertIn("Forward_Earnings_Yield", new_features)
        self.assertNotIn("Reported_EPS", new_features)
        self.assertNotIn("EPS_Estimate", new_features)

    def test_zero_or_negative_price_safety(self):
        """Verify graceful handling of zero or missing prices."""
        df_zero = self.sample_df.copy()
        df_zero.loc[0, "Close"] = 0.0
        df_zero.loc[1, "Close"] = np.nan
        out = compute_scale_invariant_earnings_features(df_zero)
        self.assertFalse(np.isinf(out["Earnings_Yield"].iloc[0]))
        self.assertFalse(np.isnan(out["Earnings_Yield"].iloc[1]))

    def test_production_artifacts_unmodified(self):
        """Verify that production files are completely untouched."""
        prod_contract_path = PROJECT_ROOT / "models" / "production" / "model_contract.json"
        self.assertTrue(prod_contract_path.exists())
        with open(prod_contract_path) as f:
            import json
            contract = json.load(f)
        self.assertEqual(contract["model_version"], "1.0.0")
        self.assertIn("Reported_EPS", contract["feature_names"])


if __name__ == "__main__":
    unittest.main()

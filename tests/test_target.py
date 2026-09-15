"""
Unit tests for Target Generation in StockDNA-AI.

Verifies:
1. Target formula calculation: Close[t+5] / Close[t] - 1
2. BUY (2), HOLD (1), SELL (0) class mapping
3. Final incomplete rows (last `horizon` rows) dropped from supervised training
4. Grouping by Ticker prevents cross-ticker target leakage
"""

import unittest
import numpy as np
import pandas as pd
from scripts.ml.target_design import generate_target, LABEL_MAP, DEFAULT_BUY_THRESHOLD, DEFAULT_SELL_THRESHOLD


class TestTargetDesign(unittest.TestCase):

    def setUp(self):
        # Create a synthetic 10-bar price series
        # Prices: [100, 100, 100, 100, 100, 105, 95, 100, 100, 100]
        # Bar 0 -> Bar 5: 105 / 100 - 1 = +5% (BUY, 2)
        # Bar 1 -> Bar 6: 95 / 100 - 1 = -5% (SELL, 0)
        # Bar 2 -> Bar 7: 100 / 100 - 1 = 0% (HOLD, 1)
        self.prices = [100.0, 100.0, 100.0, 100.0, 100.0, 105.0, 95.0, 100.0, 100.0, 100.0]
        self.df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "Close": self.prices,
            "Ticker": "TEST",
        })

    def test_target_formula_and_labels(self):
        res = generate_target(self.df, horizon=5, buy_threshold=0.02, sell_threshold=-0.02, drop_unlabeled=True)

        # Since horizon=5 and input has 10 rows, exactly 5 rows should remain
        self.assertEqual(len(res), 5)

        # Check return calculation
        self.assertAlmostEqual(res["Forward_Return_5d"].iloc[0], 0.05, places=4)
        self.assertAlmostEqual(res["Forward_Return_5d"].iloc[1], -0.05, places=4)
        self.assertAlmostEqual(res["Forward_Return_5d"].iloc[2], 0.00, places=4)

        # Check label mapping
        self.assertEqual(res["Target"].iloc[0], 2)  # BUY
        self.assertEqual(res["Target"].iloc[1], 0)  # SELL
        self.assertEqual(res["Target"].iloc[2], 1)  # HOLD

    def test_final_incomplete_rows_removed(self):
        # Without dropping unlabeled
        res_all = generate_target(self.df, horizon=5, drop_unlabeled=False)
        self.assertEqual(len(res_all), 10)
        self.assertTrue(res_all["Forward_Return_5d"].iloc[5:].isna().all())

        # With drop_unlabeled
        res_clean = generate_target(self.df, horizon=5, drop_unlabeled=True)
        self.assertEqual(len(res_clean), 5)
        self.assertFalse(res_clean["Target"].isna().any())

    def test_panel_grouping_prevents_cross_ticker_leakage(self):
        # Two tickers of 7 rows each (horizon=5)
        df_a = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=7, freq="D"),
            "Close": [100.0] * 7,
            "Ticker": "TICKER_A",
        })
        df_b = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=7, freq="D"),
            "Close": [200.0] * 7,
            "Ticker": "TICKER_B",
        })
        panel = pd.concat([df_a, df_b], ignore_index=True)

        res = generate_target(panel, horizon=5, drop_unlabeled=True)

        # Each ticker has 7 rows, horizon=5 -> exactly 2 rows per ticker = 4 rows total
        self.assertEqual(len(res), 4)
        self.assertEqual(list(res["Ticker"].value_counts().values), [2, 2])

        # If grouping worked, returns should be 0.0, NOT cross-ticker 200/100-1 = +100%
        self.assertTrue((res["Forward_Return_5d"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()

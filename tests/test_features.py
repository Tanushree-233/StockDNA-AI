"""
Unit tests for Scale-Invariant Feature Generation and Registry in StockDNA-AI.

Verifies:
1. No raw nominal price columns in final X (Close, Open, High, Low, raw SMAs/EMAs/BBs)
2. Every feature has an explicit INTERNAL or EXTERNAL group
3. Scale-invariance: Multiplying price by 100x yields identical scale-invariant ratios
"""

import unittest
import numpy as np
import pandas as pd

from config.feature_registry import (
    FEATURE_REGISTRY,
    get_feature_names,
    get_internal_features,
    get_external_features,
    get_feature_group,
)
from scripts.feature_engineering.canonical_features import (
    compute_technical_features,
    extract_feature_matrix,
    build_canonical_features,
)


class TestFeatures(unittest.TestCase):

    def test_feature_registry_integrity(self):
        names = get_feature_names()
        internal = get_internal_features()
        external = get_external_features()

        self.assertEqual(len(names), 34)
        self.assertEqual(len(internal), 10)
        self.assertEqual(len(external), 24)
        self.assertEqual(len(names), len(internal) + len(external))

        # Every feature must have group, source, dtype, definition, point_in_time_rule, imputation_rule
        for name, meta in FEATURE_REGISTRY.items():
            self.assertIn(meta["group"], ["INTERNAL", "EXTERNAL"])
            self.assertTrue(len(meta["definition"]) > 0)
            self.assertTrue(len(meta["point_in_time_rule"]) > 0)
            self.assertTrue(len(meta["imputation_rule"]) > 0)

    def test_no_raw_nominal_prices_in_feature_names(self):
        names = get_feature_names()
        forbidden_raw = [
            "Close", "Open", "High", "Low", "Adj Close",
            "SMA20", "SMA50", "SMA100", "SMA200",
            "EMA20", "EMA50", "EMA100",
            "BB_High", "BB_Low", "BB_Middle",
            "Lag_Close_1", "Lag_Close_3", "Lag_Close_5",
            "NIFTY_Close",
        ]
        for forbidden in forbidden_raw:
            self.assertNotIn(forbidden, names)

    def test_scale_invariance_under_price_multiplication(self):
        # Generate 60 bars of random price data
        np.random.seed(42)
        base_prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 60)))
        df_normal = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "Close": base_prices,
            "High": base_prices * 1.01,
            "Low": base_prices * 0.99,
            "Open": base_prices * 1.002,
            "Volume": [100000] * 60,
            "Ticker": "STOCK_A",
        })

        # Same stock scaled by 100x (e.g. MRF vs penny stock)
        df_scaled = df_normal.copy()
        df_scaled["Close"] = df_normal["Close"] * 100.0
        df_scaled["High"] = df_normal["High"] * 100.0
        df_scaled["Low"] = df_normal["Low"] * 100.0
        df_scaled["Open"] = df_normal["Open"] * 100.0

        feat_normal = compute_technical_features(df_normal)
        feat_scaled = compute_technical_features(df_scaled)

        scale_invariant_cols = [
            "Daily_Return", "Return_5d", "Close_to_SMA20", "Close_to_SMA50",
            "Normalized_ATR", "High_Low_Spread_Pct", "Open_Close_Spread_Pct",
            "Bollinger_PctB", "RSI_14", "Normalized_MACD", "Normalized_MACD_Hist"
        ]

        for col in scale_invariant_cols:
            diff = np.abs(feat_normal[col].iloc[30:] - feat_scaled[col].iloc[30:])
            max_diff = float(diff.max())
            self.assertLess(max_diff, 1e-4, f"Feature {col} is not scale-invariant! Max diff: {max_diff}")


if __name__ == "__main__":
    unittest.main()

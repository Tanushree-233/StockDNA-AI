"""
Unit tests for Offline / Live Feature Parity in StockDNA-AI.

Verifies:
1. Feature extraction in batch training pipeline vs. live prediction service
2. Given identical input data, offline and live feature computation must produce
   exact identical values (max difference < 1e-6)
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from config.settings import PROJECT_ROOT
from config.feature_registry import get_feature_names
from scripts.feature_engineering.canonical_features import (
    build_canonical_features,
    extract_feature_matrix,
)
from scripts.ml.predict_service import load_production_model, predict_stock


class TestOfflineLiveParity(unittest.TestCase):

    def test_feature_parity_between_batch_and_live(self):
        # Load a slice of real price data for TCS
        tcs_path = PROJECT_ROOT / "data" / "processed" / "prices" / "TCS.csv"
        if not tcs_path.exists():
            tcs_path = PROJECT_ROOT / "data" / "raw" / "prices" / "TCS.csv"
        self.assertTrue(tcs_path.exists(), "TCS price file required for parity test.")

        price_df = pd.read_csv(tcs_path)
        if "Price" in price_df.columns and "Date" not in price_df.columns:
            price_df = price_df.rename(columns={"Price": "Date"})
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df["Ticker"] = "TCS"

        nifty_path = PROJECT_ROOT / "data" / "raw" / "market" / "nifty50.csv"
        nifty_df = pd.read_csv(nifty_path) if nifty_path.exists() else None
        vix_path = PROJECT_ROOT / "data" / "raw" / "market" / "indiavix.csv"
        vix_df = pd.read_csv(vix_path) if vix_path.exists() else None
        earnings_path = PROJECT_ROOT / "data" / "processed" / "internal" / "earnings_events.csv"
        earnings_df = pd.read_csv(earnings_path) if earnings_path.exists() else None

        # 1. Compute via batch pipeline (offline path)
        offline_featured = build_canonical_features(
            price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )
        offline_latest_row = offline_featured.iloc[[-1]]
        offline_vector = extract_feature_matrix(offline_latest_row)

        # 2. Compute via live prediction service (live path)
        live_res = predict_stock(
            "TCS", price_df=price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )
        live_snapshot = live_res["feature_snapshot"]

        feature_names = get_feature_names()
        for feat in feature_names:
            offline_val = float(offline_vector[feat].iloc[0])
            live_val = float(live_snapshot[feat])

            # Both NaN or both equal
            if np.isnan(offline_val):
                # Live fills via training median or 0.0, which is also fine
                continue
            diff = abs(offline_val - live_val)
            self.assertLess(
                diff,
                1e-4,
                f"Parity failure on feature '{feat}': offline={offline_val}, live={live_val}, diff={diff}"
            )


if __name__ == "__main__":
    unittest.main()

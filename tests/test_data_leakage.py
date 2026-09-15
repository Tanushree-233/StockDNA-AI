"""
Unit tests for Point-in-Time Data Leakage Prevention in StockDNA-AI.

Verifies:
1. No future price or target in X feature matrix
2. No current snapshot fundamentals (MarketCap, PE_Ratio, Beta, BookValue, etc.) in model inputs
3. Point-in-time earnings alignment: An announcement on date T is NOT visible on date T-1
4. Train/Val/Test embargo prevents 5-day forward target overlap across split boundaries
"""

import unittest
import numpy as np
import pandas as pd

from config.feature_registry import get_feature_names
from scripts.feature_engineering.canonical_features import compute_internal_features
from scripts.ml.train_production_model import load_and_split_data


class TestDataLeakage(unittest.TestCase):

    def test_no_snapshot_fundamentals_in_feature_set(self):
        """Verify no current static snapshot metrics from Ticker.info are in feature set."""
        names = get_feature_names()
        leaky_snapshots = [
            "MarketCap", "PE_Ratio", "ForwardPE", "BookValue",
            "DividendYield", "Beta", "ForwardEPS", "TrailingEPS",
            "EarningsQuarterlyGrowth"
        ]
        for leaked in leaky_snapshots:
            self.assertNotIn(leaked, names, f"Snapshot fundamental {leaked} found in feature set!")

    def test_no_target_or_future_return_in_feature_set(self):
        names = get_feature_names()
        forbidden_targets = [
            "Target", "target", "Target_Return", "Forward_Return_5d",
            "future_price", "future_return"
        ]
        for f in forbidden_targets:
            self.assertNotIn(f, names)

    def test_point_in_time_earnings_event_alignment(self):
        """Earnings event announced on Jan 15 must NOT appear on Jan 14."""
        market_dates = pd.date_range("2024-01-10", "2024-01-20", freq="D")
        stock_df = pd.DataFrame({
            "Date": market_dates,
            "Close": [100.0] * len(market_dates),
            "Ticker": "TCS",
        })

        # Earnings announcement on Jan 15
        earnings_df = pd.DataFrame({
            "Ticker": ["TCS"],
            "Date": [pd.Timestamp("2024-01-15")],
            "EPS_Estimate": [10.0],
            "Reported_EPS": [12.0],
            "EPS_Surprise": [2.0],
        })

        merged = compute_internal_features(stock_df, earnings_df)

        # Before Jan 15: Reported_EPS should be 0.0 (unseen), Earnings_Event should be 0
        before_event = merged[merged["Date"] < "2024-01-15"]
        self.assertTrue((before_event["Earnings_Event"] == 0).all())
        self.assertTrue((before_event["Reported_EPS"] == 0.0).all())

        # On Jan 15: Earnings_Event must be 1, Reported_EPS must be 12.0
        on_event = merged[merged["Date"] == "2024-01-15"]
        self.assertEqual(on_event["Earnings_Event"].iloc[0], 1)
        self.assertEqual(on_event["Reported_EPS"].iloc[0], 12.0)

        # After Jan 15: Earnings_Event is 0, but Reported_EPS is retained as prior quarterly result
        after_event = merged[merged["Date"] > "2024-01-15"]
        self.assertTrue((after_event["Earnings_Event"] == 0).all())
        self.assertTrue((after_event["Reported_EPS"] == 12.0).all())

    def test_chronological_splits_and_embargo(self):
        train_df, val_df, test_df = load_and_split_data()

        train_max = train_df["Date"].max()
        val_min = val_df["Date"].min()
        val_max = val_df["Date"].max()
        test_min = test_df["Date"].min()

        # Strict ordering
        self.assertLess(train_max, val_min)
        self.assertLess(val_max, test_min)

        # 5-day embargo verification:
        # Distance between train_max and val_min must be >= 5 calendar days
        embargo_train_val = (val_min - train_max).days
        self.assertGreaterEqual(embargo_train_val, 5, "Train-to-Val embargo is less than 5 days!")

        embargo_val_test = (test_min - val_max).days
        self.assertGreaterEqual(embargo_val_test, 5, "Val-to-Test embargo is less than 5 days!")

    def test_trading_day_label_embargo_zero_overlap(self):
        """Verify that the 5-day forward return of the last observation in each split
        is fully realized BEFORE the subsequent split begins in trading days."""
        df = pd.read_csv("data/processed/production_dataset.csv")
        df["Date"] = pd.to_datetime(df["Date"])

        train_df, val_df, test_df = load_and_split_data()
        val_start_date = val_df["Date"].min()
        test_start_date = test_df["Date"].min()

        for ticker in df["Ticker"].unique():
            ticker_data = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
            
            # Find the index of the last train observation for this ticker
            last_train_idx = ticker_data[ticker_data["Date"] <= train_df["Date"].max()].index[-1]
            # Realized return date is t + 5 trading days
            train_realized_date = ticker_data.loc[last_train_idx + 5, "Date"]
            self.assertLess(train_realized_date, val_start_date,
                            f"{ticker}: Train label realized on {train_realized_date} overlaps with val start {val_start_date}")

            # Find the index of the last val observation for this ticker
            last_val_idx = ticker_data[ticker_data["Date"] <= val_df["Date"].max()].index[-1]
            val_realized_date = ticker_data.loc[last_val_idx + 5, "Date"]
            self.assertLess(val_realized_date, test_start_date,
                            f"{ticker}: Val label realized on {val_realized_date} overlaps with test start {test_start_date}")

    def test_no_infinite_or_nan_in_imputed_features(self):
        """Verify that apply_imputation produces strictly finite values without inf or NaN."""
        from scripts.ml.train_production_model import compute_training_imputation, apply_imputation
        train_df, val_df, test_df = load_and_split_data()
        feature_names = get_feature_names()
        imputation_dict = compute_training_imputation(train_df, feature_names)

        for split_df, name in [(train_df, "Train"), (val_df, "Val"), (test_df, "Test")]:
            X = apply_imputation(split_df, feature_names, imputation_dict)
            self.assertFalse(np.isnan(X.values).any(), f"NaNs found in {name} features!")
            self.assertFalse(np.isinf(X.values).any(), f"Infs found in {name} features!")


if __name__ == "__main__":
    unittest.main()


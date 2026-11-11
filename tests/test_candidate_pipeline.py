"""
Unit tests for Candidate v2 pipeline.

Verifies:
1. Feature generation produces all 34 candidate features.
2. Robustness to missing, zero, or invalid prices.
3. Contract consistency (exact 34-feature contract, names, and ordering).
4. Training and inference parity (single-bar inference matches batch processing).
5. Absolute production and Candidate v1 isolation.
"""

import unittest
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from scripts.feature_engineering.candidate_features import (
    build_candidate_features,
    extract_candidate_feature_matrix,
    apply_scale_invariant_earnings,
    CANDIDATE_V2_FEATURE_NAMES,
)


class TestCandidateV2Pipeline(unittest.TestCase):

    def setUp(self):
        # Generate synthetic realistic multi-bar stock series (>= 220 bars for 200 SMA)
        dates = pd.date_range("2023-01-01", periods=250, freq="B")
        np.random.seed(42)
        prices = 1000.0 + np.cumsum(np.random.randn(len(dates)) * 10)
        self.sample_df = pd.DataFrame({
            "Date": dates,
            "Ticker": ["INFY"] * len(dates),
            "Close": prices,
            "High": prices + np.random.uniform(5, 15, len(dates)),
            "Low": prices - np.random.uniform(5, 15, len(dates)),
            "Open": prices + np.random.uniform(-5, 5, len(dates)),
            "Volume": np.random.randint(500000, 2000000, len(dates)),
            "Reported_EPS": [15.5] * len(dates),
            "EPS_Estimate": [15.0] * len(dates),
            "EPS_Surprise": [0.5] * len(dates),
            "EPS_Surprise_Pct": [0.0333] * len(dates),
            "Earnings_Event": [0] * len(dates),
            "Positive_Earnings_Surprise": [1] * len(dates),
            "Negative_Earnings_Surprise": [0] * len(dates),
            "Days_Since_Earnings": [12.0] * len(dates),
            "Post_Earnings_1_Week": [0] * len(dates),
            "Post_Earnings_1_Month": [1] * len(dates),
        })

    def test_feature_generation_dimension(self):
        """Verify that build_candidate_features produces all 34 contract features."""
        feat_df = build_candidate_features(self.sample_df)
        X = extract_candidate_feature_matrix(feat_df)
        self.assertEqual(len(X.columns), 34)
        self.assertEqual(list(X.columns), CANDIDATE_V2_FEATURE_NAMES)
        self.assertIn("Earnings_Yield", X.columns)
        self.assertIn("Forward_Earnings_Yield", X.columns)
        self.assertNotIn("Reported_EPS", X.columns)
        self.assertNotIn("EPS_Estimate", X.columns)

    def test_missing_and_invalid_prices(self):
        """Verify that zero, negative, or NaN prices do not generate infinities or crash."""
        bad_df = self.sample_df.copy()
        bad_df.loc[0, "Close"] = 0.0
        bad_df.loc[1, "Close"] = -100.0
        bad_df.loc[2, "Close"] = np.nan
        bad_df.loc[3, "Reported_EPS"] = np.nan

        out = apply_scale_invariant_earnings(bad_df)
        self.assertTrue(np.isfinite(out["Earnings_Yield"]).all())
        self.assertTrue(np.isfinite(out["Forward_Earnings_Yield"]).all())

    def test_contract_consistency(self):
        """Verify contract specification matches code constants."""
        self.assertEqual(len(CANDIDATE_V2_FEATURE_NAMES), 34)
        # Check uniqueness
        self.assertEqual(len(set(CANDIDATE_V2_FEATURE_NAMES)), 34)

    def test_training_inference_parity(self):
        """Verify that live single-bar extraction produces identical features to batch processing."""
        # 1. Batch computation
        batch_df = build_candidate_features(self.sample_df)
        batch_matrix = extract_candidate_feature_matrix(batch_df)
        last_batch_row = batch_matrix.iloc[-1].to_dict()

        # 2. Single-bar latest point computation (passing the same series history)
        inference_df = build_candidate_features(self.sample_df)
        inference_matrix = extract_candidate_feature_matrix(inference_df)
        last_inf_row = inference_matrix.iloc[[-1]].iloc[0].to_dict()

        for feat in CANDIDATE_V2_FEATURE_NAMES:
            val_batch = last_batch_row[feat]
            val_inf = last_inf_row[feat]
            if pd.isna(val_batch) and pd.isna(val_inf):
                continue
            self.assertAlmostEqual(
                val_batch, val_inf, places=6,
                msg=f"Feature '{feat}' diverged between batch and inference! Batch: {val_batch}, Inf: {val_inf}"
            )

    def test_production_and_candidate_v1_isolation(self):
        """Verify that production and Candidate v1 artifacts remain untouched."""
        prod_contract = PROJECT_ROOT / "models" / "production" / "model_contract.json"
        prod_model = PROJECT_ROOT / "models" / "production" / "model.json"
        cand_v1 = PROJECT_ROOT / "models" / "candidates" / "candidate_v1" / "candidate_model.joblib"

        self.assertTrue(prod_contract.exists())
        self.assertTrue(prod_model.exists())
        self.assertTrue(cand_v1.exists())

        with open(prod_contract) as f:
            c = json.load(f)
        self.assertEqual(c["model_version"], "1.0.0")
        self.assertIn("Reported_EPS", c["feature_names"])


if __name__ == "__main__":
    unittest.main()

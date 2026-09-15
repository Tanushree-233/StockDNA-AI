"""
Unit tests for End-to-End Prediction Service and Backend Predictor in StockDNA-AI.

Verifies:
1. `predict_stock` returns canonical structured prediction dictionary
2. Works across tickers in company universe (TCS, INFY, RELIANCE)
3. Backend `predict()` integrates SHAP decomposition with INTERNAL/EXTERNAL split
"""

import unittest
from scripts.ml.predict_service import predict_stock
from backend.predictor import predict


class TestPrediction(unittest.TestCase):

    def test_predict_stock_contract(self):
        res = predict_stock("TCS")

        required_keys = [
            "ticker", "timestamp", "prediction", "confidence",
            "probabilities", "model_version", "feature_snapshot"
        ]
        for k in required_keys:
            self.assertIn(k, res)

        self.assertEqual(res["ticker"], "TCS")
        self.assertIn(res["prediction"], ["SELL", "HOLD", "BUY"])
        self.assertGreaterEqual(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)

        probs = res["probabilities"]
        self.assertIn("SELL", probs)
        self.assertIn("HOLD", probs)
        self.assertIn("BUY", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

        # Feature snapshot must contain all 34 features
        self.assertEqual(len(res["feature_snapshot"]), 34)

    def test_predict_all_universe_tickers(self):
        for ticker in ["TCS", "INFY", "RELIANCE"]:
            res = predict_stock(ticker)
            self.assertEqual(res["ticker"], ticker)
            self.assertIn(res["prediction"], ["SELL", "HOLD", "BUY"])

    def test_backend_predict_with_xai_decomposition(self):
        out = predict("INFY")

        self.assertEqual(out["ticker"], "INFY")
        self.assertIn(out["prediction"], ["SELL", "HOLD", "BUY"])
        self.assertIn(out["primary_driver"], ["INTERNAL", "EXTERNAL"])

        # Percentages must sum to ~100%
        int_pct = out["internal_percentage"]
        ext_pct = out["external_percentage"]
        self.assertAlmostEqual(int_pct + ext_pct, 100.0, places=1)

        # SHAP XAI structure present
        self.assertIn("xai", out)
        self.assertIn("top_internal_factors", out["xai"])
        self.assertIn("top_external_factors", out["xai"])
        self.assertIn("all_feature_contributions", out["xai"])


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for Market Trends, AI Chatbot, and Layman Explainability in StockDNA-AI.
"""

import unittest
from fastapi.testclient import TestClient
from backend.app import app
from backend.services.shap_service import FEATURE_LAYMAN_METADATA, generate_layman_explanation, generate_layman_narrative


class TestTrendsAndChatbot(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_trends_summary_endpoint(self):
        """Verify summary endpoint returns all 22 stocks with price, change, and 52-week stats."""
        res = self.client.get("/market/trends/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 22)

        # Check fields on first stock
        stock = data[0]
        for field in ["ticker", "company", "sector", "latest_price", "change_1d", "change_pct_1d", "high_52w", "low_52w", "trend"]:
            self.assertIn(field, stock)
            self.assertIsNotNone(stock[field])

    def test_trends_ticker_series_ranges(self):
        """Verify ticker trends series for different time horizons."""
        for ticker in ["TCS", "INFY", "RELIANCE"]:
            for r in ["1M", "3M", "6M", "1Y"]:
                res = self.client.get(f"/market/trends/{ticker}?range={r}")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                self.assertEqual(data["ticker"], ticker)
                self.assertEqual(data["range"], r)
                self.assertIn("series", data)
                self.assertGreater(len(data["series"]), 0)

                first_pt = data["series"][0]
                for k in ["date", "open", "high", "low", "close", "volume"]:
                    self.assertIn(k, first_pt)

    def test_chatbot_company_question(self):
        """Verify chatbot answers questions about specific tracked companies."""
        res = self.client.post("/chatbot/message", json={"message": "Tell me about Reliance Industries"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reply", data)
        self.assertEqual(data["detected_ticker"], "RELIANCE")
        self.assertIn("Reliance", data["reply"])
        self.assertIn("suggested_questions", data)
        self.assertGreater(len(data["suggested_questions"]), 0)

    def test_chatbot_comparison(self):
        """Verify chatbot side-by-side comparison between two companies."""
        res = self.client.post("/chatbot/message", json={"message": "Compare TCS and Infosys"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["detected_intent"], "comparison")
        self.assertIn("TCS", data["reply"])
        self.assertIn("Infosys", data["reply"])

    def test_chatbot_layman_concept(self):
        """Verify chatbot plain-English concept explanations."""
        res = self.client.post("/chatbot/message", json={"message": "What is India VIX in simple words?"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("reply", data)
        self.assertIn("Fear", data["reply"])

    def test_layman_metadata_covers_all_34_features(self):
        """Verify that all 34 production features have complete layman metadata and translations."""
        self.assertEqual(len(FEATURE_LAYMAN_METADATA), 34)

        for feat, meta in FEATURE_LAYMAN_METADATA.items():
            self.assertIn("name", meta, f"Feature {feat} missing layman name")
            self.assertIn("meaning", meta, f"Feature {feat} missing layman meaning")
            self.assertIn("pos_hold", meta, f"Feature {feat} missing pos_hold")
            self.assertIn("neg_hold", meta, f"Feature {feat} missing neg_hold")

            # Test layman explanation generator
            res_hold_pos = generate_layman_explanation(feat, 0.05, "HOLD")
            self.assertTrue(len(res_hold_pos["layman_name"]) > 0)
            self.assertTrue(len(res_hold_pos["layman_explanation"]) > 0)
            self.assertTrue(len(res_hold_pos["layman_definition"]) > 0)

            res_buy_pos = generate_layman_explanation(feat, 0.05, "BUY")
            self.assertTrue(len(res_buy_pos["layman_explanation"]) > 0)

            res_sell_pos = generate_layman_explanation(feat, 0.05, "SELL")
            self.assertTrue(len(res_sell_pos["layman_explanation"]) > 0)


if __name__ == "__main__":
    unittest.main()


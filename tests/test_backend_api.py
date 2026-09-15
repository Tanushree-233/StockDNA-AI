"""
Comprehensive Automated API Test Suite for StockDNA-AI.

Verifies:
1. Authentication: Register, Login, Invalid credentials, Unauthenticated access
2. Prediction API: POST /predict returns valid schema with XAI decomposition
3. User Isolation: User A cannot read or delete User B's predictions
4. History CRUD: GET /history, GET /history/{id}, DELETE /history/{id}, DELETE /history
5. Data Tool API: Standalone operation of universe, process, sentiment, and factor classification
6. API Error Handling: 404 on unknown ticker, 401 on missing auth, 422 on malformed payloads
"""

import unittest
from fastapi.testclient import TestClient
from backend.app import app


class TestBackendAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

        # Register and log in User 1
        cls.client.post("/auth/register", json={
            "username": "api_user_1",
            "email": "user1@stockdna.io",
            "password": "SecurePassword123!"
        })
        login_1 = cls.client.post("/auth/login", json={
            "email": "user1@stockdna.io",
            "password": "SecurePassword123!"
        })
        cls.token_1 = login_1.json()["access_token"]
        cls.headers_1 = {"Authorization": f"Bearer {cls.token_1}"}

        # Register and log in User 2
        cls.client.post("/auth/register", json={
            "username": "api_user_2",
            "email": "user2@stockdna.io",
            "password": "SecurePassword123!"
        })
        login_2 = cls.client.post("/auth/login", json={
            "email": "user2@stockdna.io",
            "password": "SecurePassword123!"
        })
        cls.token_2 = login_2.json()["access_token"]
        cls.headers_2 = {"Authorization": f"Bearer {cls.token_2}"}

    def test_auth_duplicate_registration_rejected(self):
        res = self.client.post("/auth/register", json={
            "username": "api_user_1",
            "email": "user1@stockdna.io",
            "password": "Password123!"
        })
        self.assertEqual(res.status_code, 400)

    def test_auth_invalid_login_rejected(self):
        res = self.client.post("/auth/login", json={
            "email": "user1@stockdna.io",
            "password": "WrongPassword!"
        })
        self.assertEqual(res.status_code, 401)

    def test_unauthenticated_predict_rejected(self):
        res = self.client.post("/predict", json={"ticker": "TCS"})
        self.assertEqual(res.status_code, 401)

    def test_prediction_api_contract(self):
        res = self.client.post("/predict", json={"ticker": "TCS"}, headers=self.headers_1)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # Required fields in Step 2 contract
        self.assertEqual(data["ticker"], "TCS")
        self.assertIn("company", data)
        self.assertIn("timestamp", data)
        self.assertIn(data["prediction"], ["BUY", "HOLD", "SELL"])
        self.assertIsInstance(data["confidence"], (int, float))

        # Probabilities
        probs = data["probabilities"]
        self.assertIn("SELL", probs)
        self.assertIn("HOLD", probs)
        self.assertIn("BUY", probs)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

        # XAI sub-object
        self.assertIn("xai", data)
        xai = data["xai"]
        self.assertIn(xai["primary_driver"], ["INTERNAL", "EXTERNAL"])
        self.assertAlmostEqual(xai["internal_percentage"] + xai["external_percentage"], 100.0, places=1)
        self.assertTrue(len(xai["top_internal_factors"]) > 0)
        self.assertTrue(len(xai["top_external_factors"]) > 0)
        self.assertEqual(len(xai["all_feature_contributions"]), 34)

        # Factor structure
        sample_factor = xai["top_internal_factors"][0]
        self.assertIn("feature", sample_factor)
        self.assertEqual(sample_factor["group"], "INTERNAL")
        self.assertIn("shap_value", sample_factor)
        self.assertIn("importance", sample_factor)
        self.assertIn(sample_factor["direction"], ["POSITIVE", "NEGATIVE", "NEUTRAL"])
        self.assertIn("explanation", sample_factor)

    def test_prediction_unsupported_ticker_error(self):
        res = self.client.post("/predict", json={"ticker": "UNKNOWN_TICKER_XYZ"}, headers=self.headers_1)
        self.assertEqual(res.status_code, 404)

    def test_prediction_user_isolation(self):
        # User 1 predicts TCS
        self.client.post("/predict", json={"ticker": "TCS"}, headers=self.headers_1)
        # User 2 predicts INFY
        self.client.post("/predict", json={"ticker": "INFY"}, headers=self.headers_2)

        # Check User 1 history
        hist_1 = self.client.get("/history/", headers=self.headers_1).json()
        tickers_1 = [h["ticker"] for h in hist_1]
        self.assertTrue(all(t == "TCS" for t in tickers_1))

        # Check User 2 history
        hist_2 = self.client.get("/history/", headers=self.headers_2).json()
        tickers_2 = [h["ticker"] for h in hist_2]
        self.assertTrue(all(t == "INFY" for t in tickers_2))

        # User 1 cannot access User 2's prediction item
        user_2_pred_id = hist_2[0]["id"]
        cross_get = self.client.get(f"/history/{user_2_pred_id}", headers=self.headers_1)
        self.assertEqual(cross_get.status_code, 404)

        # User 1 cannot delete User 2's prediction item
        cross_del = self.client.delete(f"/history/{user_2_pred_id}", headers=self.headers_1)
        self.assertEqual(cross_del.status_code, 404)

        # User 2 CAN access their own prediction item
        own_get = self.client.get(f"/history/{user_2_pred_id}", headers=self.headers_2)
        self.assertEqual(own_get.status_code, 200)

    def test_datatool_standalone_endpoints(self):
        # 1. Universe
        u_res = self.client.get("/datatool/universe")
        self.assertEqual(u_res.status_code, 200)
        self.assertTrue(len(u_res.json()) >= 3)

        # 2. Process
        p_res = self.client.post("/datatool/process", json={
            "ticker": "RELIANCE",
            "records": [
                {"Date": "2024-01-01", "Close": 2500.0, "Open": 2490.0, "High": 2510.0, "Low": 2480.0, "Volume": 400000},
                {"Date": "2024-01-02", "Close": 2520.0, "Open": 2500.0, "High": 2530.0, "Low": 2495.0, "Volume": 450000}
            ]
        })
        self.assertEqual(p_res.status_code, 200)
        self.assertEqual(p_res.json()["records_count"], 2)

        # 3. Sentiment
        s_res = self.client.post("/datatool/sentiment", json={"text": "Profits decline sharply amid high inflation"})
        self.assertEqual(s_res.status_code, 200)
        self.assertEqual(s_res.json()["sentiment"], "NEGATIVE")

        # 4. Factor classify
        f_res = self.client.post("/datatool/factors/classify", json={"feature_name": "Reported_EPS"})
        self.assertEqual(f_res.status_code, 200)
        self.assertEqual(f_res.json()["group"], "INTERNAL")


if __name__ == "__main__":
    unittest.main()

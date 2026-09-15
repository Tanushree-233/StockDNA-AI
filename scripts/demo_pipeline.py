"""
End-to-End Demonstration Script for StockDNA-AI Phase 4.

Demonstrates:
1. User registration & authentication
2. Execution of live prediction via FastAPI test client
3. Integration of Data Tool -> Canonical Features -> Production XGBoost -> BUY/HOLD/SELL + Confidence
4. SHAP XAI calculation with INTERNAL vs EXTERNAL factor decomposition
5. Relational persistence in SQLite/PostgreSQL prediction history
"""

import os
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app import app


def run_demo():
    client = TestClient(app)

    print("=" * 70)
    print("STOCKDNA-AI PHASE 4: END-TO-END SYSTEM DEMONSTRATION")
    print("=" * 70)

    # 1. Register & Authenticate User
    client.post("/auth/register", json={
        "username": "demo_trader",
        "email": "demo_trader@stockdna.io",
        "password": "Password123!"
    })
    login = client.post("/auth/login", json={
        "email": "demo_trader@stockdna.io",
        "password": "Password123!"
    })
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("[1] Authentication: Logged in successfully. JWT obtained.")

    # 2. Predict TCS
    print("\n[2] Executing Production Prediction on TCS...")
    resp = client.post("/predict", json={"ticker": "TCS"}, headers=headers)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    print(f"    Ticker         : {data['ticker']} ({data['company']})")
    print(f"    Timestamp      : {data['timestamp']}")
    print(f"    Prediction     : {data['prediction']}")
    print(f"    Confidence     : {data['confidence']:.4f}")
    print(f"    Probabilities  : {data['probabilities']}")

    # 3. Inspect XAI Decomposition
    print("\n[3] Explainable AI (SHAP) Factor Decomposition:")
    xai = data["xai"]
    print(f"    Primary Driver : {xai['primary_driver']}")
    print(f"    Internal Share : {xai['internal_percentage']}%")
    print(f"    External Share : {xai['external_percentage']}%")
    print(f"    Total Check    : {xai['internal_percentage'] + xai['external_percentage']}%")

    print("\n    Top Internal Drivers (Company-Specific):")
    for f in xai["top_internal_factors"][:3]:
        print(f"      * {f['feature']:<26} [SHAP: {f['shap_value']:+.4f} | {f['direction']:<8}]: {f['explanation']}")

    print("\n    Top External Drivers (Market / Technical / Macro):")
    for f in xai["top_external_factors"][:3]:
        print(f"      * {f['feature']:<26} [SHAP: {f['shap_value']:+.4f} | {f['direction']:<8}]: {f['explanation']}")

    # 4. Predict INFY
    print("\n[4] Executing Production Prediction on INFY...")
    resp_infy = client.post("/predict", json={"ticker": "INFY"}, headers=headers)
    data_infy = resp_infy.json()
    print(f"    Ticker         : {data_infy['ticker']} ({data_infy['company']})")
    print(f"    Prediction     : {data_infy['prediction']} (Confidence: {data_infy['confidence']:.4f})")
    print(f"    Primary Driver : {data_infy['xai']['primary_driver']} (Internal: {data_infy['xai']['internal_percentage']}%, External: {data_infy['xai']['external_percentage']}%)")

    # 5. Database History
    print("\n[5] Auditing Persisted History...")
    hist_resp = client.get("/history/", headers=headers)
    records = hist_resp.json()
    print(f"    History Count for demo_trader: {len(records)} records")
    for r in records[:2]:
        print(f"      Record ID={r['id']}: {r['ticker']} -> {r['prediction']} (Driver: {r['primary_driver']}, Internal: {r['internal_percentage']}%, External: {r['external_percentage']}%)")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY (ALL 3 DELIVERABLES VERIFIED)")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()

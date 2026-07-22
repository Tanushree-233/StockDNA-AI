import joblib
import numpy as np
import pandas as pd

from backend.services.feature_service import get_price_features
from backend.services.fundamentals_service import get_fundamentals
from backend.services.market_service import get_market_features


# Load trained model
model = joblib.load("models/xgboost_model.pkl")

# Load training feature columns
X_train = joblib.load("models/X_train.pkl")
FEATURE_COLUMNS = list(X_train.columns)

LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY"
}


def predict(ticker: str):
    """
    Predict BUY / HOLD / SELL for a stock ticker.
    """

    # -----------------------------
    # Get all feature groups
    # -----------------------------
    technical = get_price_features(ticker)

    fundamentals = get_fundamentals(ticker)

    market = get_market_features()

    # -----------------------------
    # Merge features
    # -----------------------------
    features = {}

    features.update(technical)
    features.update(fundamentals)
    features.update(market)

    # -----------------------------
    # Keep only model features
    # -----------------------------
    model_input = {}

    for col in FEATURE_COLUMNS:
        model_input[col] = features.get(col, 0)

    df = pd.DataFrame([model_input])

    # Ensure exact training order
    df = df[FEATURE_COLUMNS]

    # -----------------------------
    # Predict
    # -----------------------------
    probabilities = model.predict_proba(df)[0]

    prediction = int(np.argmax(probabilities))

    return {

        "ticker": ticker.upper(),

        "prediction": LABELS[prediction],

        "confidence": round(
            float(probabilities[prediction]) * 100,
            2
        ),

        "probabilities": {

            "SELL": round(float(probabilities[0]) * 100, 2),

            "HOLD": round(float(probabilities[1]) * 100, 2),

            "BUY": round(float(probabilities[2]) * 100, 2)

        }

    }
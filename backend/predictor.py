import joblib
import numpy as np
import pandas as pd

from backend.services.feature_service import get_price_features
from backend.services.fundamentals_service import get_fundamentals
from backend.services.market_service import get_market_features
from backend.services.shap_service import get_shap_explanation
from backend.services.history_service import save_prediction

from fastapi import HTTPException

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
    try:
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
        top_features = get_shap_explanation(model, df)

        prediction = int(np.argmax(probabilities))

        result = {

            "ticker": ticker.upper(),

            "company": fundamentals["CompanyName"],
            "sector": fundamentals["Sector"],
            "industry": fundamentals["Industry"],
            "exchange": fundamentals["Exchange"],
            "currency": fundamentals["Currency"],
            "country": fundamentals["Country"],
            "website": fundamentals["Website"],

            "current_price": fundamentals["CurrentPrice"],
            "market_cap": fundamentals["MarketCap"],

            "prediction": LABELS[prediction],

            "confidence": round(
                float(probabilities[prediction]) * 100,
                2
            ),

            "probabilities": {

                "SELL": round(float(probabilities[0]) * 100, 2),

                "HOLD": round(float(probabilities[1]) * 100, 2),

                "BUY": round(float(probabilities[2]) * 100, 2)

            },

            "top_features": top_features

        }

        save_prediction(result)

        return result

    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction: " + str(e)
        )

"""
Backend Predictor Module for StockDNA-AI.

Integrates:
1. Canonical production model (`models/production/model.json` + `model_contract.json`)
2. Explainable AI SHAP decomposition (INTERNAL vs EXTERNAL factor attribution)
3. Relational persistence of predictions with user ownership isolation
4. Data Tool consumption through BaseDataProvider abstraction
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from fastapi import HTTPException, status

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml.predict_service import load_production_model, predict_stock
from backend.services.shap_service import get_shap_explanation
from backend.services.history_service import save_prediction
from backend.services.news_service import get_latest_news
from config.settings import get_company_universe, get_tickers
from data_tool.provider import BaseDataProvider, DataToolProvider


def predict(
    ticker: str,
    user_id: Optional[int] = None,
    data_provider: Optional[BaseDataProvider] = None,
) -> Dict[str, Any]:
    """
    Executes live inference and XAI attribution for a stock ticker.

    Parameters
    ----------
    ticker : str
        Stock symbol (e.g. "TCS", "INFY", "RELIANCE")
    user_id : Optional[int]
        Authenticated user ID to associate history record
    data_provider : Optional[BaseDataProvider]
        Provider implementation (default: DataToolProvider)

    Returns
    -------
    Dict adhering to the authoritative PredictionResponse contract.
    """
    clean_ticker = ticker.replace(".NS", "").strip().upper()

    # Validate ticker against universe if universe is defined
    known_clean = [t.replace(".NS", "").strip().upper() for t in get_tickers()]
    if known_clean and clean_ticker not in known_clean:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{clean_ticker}' is not in the supported universe. Supported tickers: {known_clean}",
        )

    try:
        # 1. Canonical prediction with offline/live feature parity via DataToolProvider
        provider = data_provider or DataToolProvider()
        pred_res = predict_stock(clean_ticker, data_provider=provider)
        model, contract = load_production_model()

        # 2. Extract feature matrix for SHAP
        feature_names = contract["feature_names"]
        feature_snapshot = pred_res["feature_snapshot"]
        X_df = pd.DataFrame([feature_snapshot])[feature_names]

        # 3. Compute SHAP explanations and factor decomposition
        # Ensure predicted class index aligns with class label
        inv_map = {v: int(k) for k, v in contract["class_mapping"].items()}
        pred_class_idx = inv_map.get(pred_res["prediction"], 1)

        shap_explanation = get_shap_explanation(
            model, X_df, predicted_class_idx=pred_class_idx
        )

        # 4. Resolve company metadata
        universe = get_company_universe()
        match = universe[universe["clean_ticker"] == clean_ticker]
        company_name = match["company"].iloc[0] if not match.empty else clean_ticker
        sector = match["sector"].iloc[0] if not match.empty else "N/A"

        # 5. Fetch latest news headlines
        try:
            news = get_latest_news(clean_ticker)
        except Exception:
            news = []

        # 6. Build authoritative XAI sub-object
        xai_payload = {
            "primary_driver": shap_explanation["primary_driver"],
            "internal_percentage": shap_explanation["internal_percentage"],
            "external_percentage": shap_explanation["external_percentage"],
            "top_internal_factors": shap_explanation["top_internal_factors"],
            "top_external_factors": shap_explanation["top_external_factors"],
            "all_feature_contributions": shap_explanation["all_feature_contributions"],
        }

        # 7. Build Authoritative Prediction Response Contract
        # Float probabilities in [0.0, 1.0] as specified in Step 2 contract
        prob_dict = {k: float(v) for k, v in pred_res["probabilities"].items()}
        conf_float = float(pred_res["confidence"])

        result = {
            "ticker": clean_ticker,
            "company": company_name,
            "timestamp": pred_res["timestamp"],
            "prediction": pred_res["prediction"],
            "confidence": conf_float,
            "probabilities": prob_dict,
            "xai": xai_payload,
            "model_version": pred_res["model_version"],
            "sector": sector,
            "latest_news": news,
            "feature_snapshot": feature_snapshot,
            # Backward-compatibility convenience top-level fields:
            "primary_driver": shap_explanation["primary_driver"],
            "internal_percentage": shap_explanation["internal_percentage"],
            "external_percentage": shap_explanation["external_percentage"],
            "top_features": shap_explanation["top_features"],
            "factor_attribution": shap_explanation["factor_attribution"],
        }

        # 8. Relational persistence (associated with user_id)
        try:
            save_prediction(result, user_id=user_id)
        except Exception as e:
            print(f"[Predictor] History save warning: {e}")

        return result

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Data file not found for ticker '{clean_ticker}': {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data for ticker '{clean_ticker}': {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during prediction: {str(e)}",
        )

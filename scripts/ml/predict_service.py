"""
Authoritative Production Prediction Service for StockDNA-AI.

Guarantees:
1. Loads native `models/production/model.json` and `models/production/model_contract.json`.
2. Generates canonical scale-invariant features with strict offline/live parity.
3. Imputes missing values using strictly training-fitted imputation rules stored in the contract.
4. Returns prediction label, confidence score (max probability), probability distribution,
   model version, and feature snapshot.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_ticker_mapping, get_tickers
from config.feature_registry import get_feature_names, get_feature_group
from scripts.feature_engineering.canonical_features import (
    build_canonical_features,
    extract_feature_matrix,
)

PROD_DIR = PROJECT_ROOT / "models" / "production"
MODEL_PATH = PROD_DIR / "model.json"
CONTRACT_PATH = PROD_DIR / "model_contract.json"

_MODEL_CACHE: Optional[XGBClassifier] = None
_CONTRACT_CACHE: Optional[Dict[str, Any]] = None


def load_production_model() -> Tuple[XGBClassifier, Dict[str, Any]]:
    """
    Loads and caches the production XGBoost model and its contract.
    """
    global _MODEL_CACHE, _CONTRACT_CACHE

    if _MODEL_CACHE is not None and _CONTRACT_CACHE is not None:
        return _MODEL_CACHE, _CONTRACT_CACHE

    if not MODEL_PATH.exists() or not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Production model artifacts not found at {PROD_DIR}. Please run training first."
        )

    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    _MODEL_CACHE = model
    _CONTRACT_CACHE = contract

    return _MODEL_CACHE, _CONTRACT_CACHE


from data_tool.provider import BaseDataProvider, DataToolProvider

def predict_stock(
    ticker: str,
    price_df: Optional[pd.DataFrame] = None,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
    data_provider: Optional[BaseDataProvider] = None,
) -> Dict[str, Any]:
    """
    Canonical prediction entrypoint for a single stock ticker.

    Parameters
    ----------
    ticker : str
        Clean ticker symbol (e.g. "TCS", "INFY", "RELIANCE") or Yahoo symbol ("TCS.NS").
    price_df : Optional[pd.DataFrame]
        Historical OHLCV data for the stock. If None, loads through BaseDataProvider.
    nifty_df : Optional[pd.DataFrame]
        NIFTY 50 data. If None, loads from data storage.
    vix_df : Optional[pd.DataFrame]
        India VIX data. If None, loads from data storage.
    earnings_df : Optional[pd.DataFrame]
        Historical earnings data. If None, loads from data storage.
    data_provider : Optional[BaseDataProvider]
        Provider implementation (default: DataToolProvider).

    Returns
    -------
    Dict with ticker, timestamp, prediction, confidence, probabilities, model_version, feature_snapshot.
    """
    model, contract = load_production_model()
    clean_ticker = ticker.replace(".NS", "").strip().upper()

    # Load price data through provider abstraction if not provided
    if price_df is None:
        provider = data_provider or DataToolProvider()
        price_df = provider.get_market_data(clean_ticker)

    # Ensure required price columns are present and numeric
    price_df = price_df.copy()
    price_df["Ticker"] = clean_ticker
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    price_df = price_df.dropna(subset=["Close", "High", "Low", "Open"]).sort_values("Date").reset_index(drop=True)

    # Load benchmark data if missing
    if nifty_df is None:
        nifty_path = PROJECT_ROOT / "data" / "raw" / "market" / "nifty50.csv"
        if nifty_path.exists():
            nifty_df = pd.read_csv(nifty_path)
    if vix_df is None:
        vix_path = PROJECT_ROOT / "data" / "raw" / "market" / "indiavix.csv"
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path)
    if earnings_df is None:
        earnings_path = PROJECT_ROOT / "data" / "processed" / "internal" / "earnings_events.csv"
        if earnings_path.exists():
            earnings_df = pd.read_csv(earnings_path)

    # Compute canonical features
    featured = build_canonical_features(
        price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
    )

    if featured.empty:
        raise ValueError(f"Feature computation resulted in empty dataset for {clean_ticker}")

    # Latest bar
    latest_row = featured.iloc[[-1]].copy()
    timestamp_str = latest_row["Date"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")

    # Extract model feature columns in exact order
    feature_names = contract["feature_names"]
    imputation_dict = contract["imputation_values"]

    X = latest_row[feature_names].copy()
    for feat in feature_names:
        fill_val = imputation_dict.get(feat, 0.0)
        X[feat] = X[feat].fillna(fill_val)

    # Model inference
    probs = model.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    class_mapping = {int(k): v for k, v in contract["class_mapping"].items()}
    pred_label = class_mapping.get(pred_idx, "HOLD")

    probabilities_dict = {
        class_mapping.get(i, str(i)): round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    feature_snapshot = {
        col: round(float(X[col].iloc[0]), 5) if pd.notnull(X[col].iloc[0]) else 0.0
        for col in feature_names
    }

    return {
        "ticker": clean_ticker,
        "timestamp": timestamp_str,
        "prediction": pred_label,
        "confidence": round(conf, 4),
        "probabilities": probabilities_dict,
        "model_version": contract.get("model_version", "1.0.0"),
        "feature_snapshot": feature_snapshot,
    }


if __name__ == "__main__":
    res = predict_stock("TCS")
    print("\n--- SAMPLE PREDICTION (TCS) ---")
    print(json.dumps({k: v for k, v in res.items() if k != "feature_snapshot"}, indent=2))
    print(f"Feature snapshot count: {len(res['feature_snapshot'])}")

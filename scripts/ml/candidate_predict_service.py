"""
Candidate v2 Inference Service for StockDNA-AI.

Loads the Candidate v2 model and contract from `models/candidates/candidate_v2/`.
Computes candidate features with strict parity to batch training data.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_engineering.candidate_features import (
    build_candidate_features,
    extract_candidate_feature_matrix,
    CANDIDATE_V2_FEATURE_NAMES,
)
from data_tool.provider import BaseDataProvider, DataToolProvider

CANDIDATE_DIR = PROJECT_ROOT / "models" / "candidates" / "candidate_v2"
MODEL_PATH = CANDIDATE_DIR / "candidate_model.json"
CONTRACT_PATH = CANDIDATE_DIR / "model_contract.json"

_CANDIDATE_MODEL_CACHE: Optional[XGBClassifier] = None
_CANDIDATE_CONTRACT_CACHE: Optional[Dict[str, Any]] = None


def load_candidate_model() -> Tuple[XGBClassifier, Dict[str, Any]]:
    """Loads and caches the Candidate v2 model and contract."""
    global _CANDIDATE_MODEL_CACHE, _CANDIDATE_CONTRACT_CACHE

    if _CANDIDATE_MODEL_CACHE is not None and _CANDIDATE_CONTRACT_CACHE is not None:
        return _CANDIDATE_MODEL_CACHE, _CANDIDATE_CONTRACT_CACHE

    if not MODEL_PATH.exists() or not CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Candidate v2 artifacts not found at {CANDIDATE_DIR}. Please run training first."
        )

    with open(CONTRACT_PATH, "r", encoding="utf-8") as f:
        contract = json.load(f)

    model = XGBClassifier()
    model.load_model(str(MODEL_PATH))

    _CANDIDATE_MODEL_CACHE = model
    _CANDIDATE_CONTRACT_CACHE = contract
    return _CANDIDATE_MODEL_CACHE, _CANDIDATE_CONTRACT_CACHE


def predict_stock_candidate(
    ticker: str,
    price_df: Optional[pd.DataFrame] = None,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
    data_provider: Optional[BaseDataProvider] = None,
) -> Dict[str, Any]:
    """Candidate v2 prediction entrypoint for a single stock ticker."""
    model, contract = load_candidate_model()
    clean_ticker = ticker.replace(".NS", "").strip().upper()

    if price_df is None:
        provider = data_provider or DataToolProvider()
        price_df = provider.get_market_data(clean_ticker)

    price_df = price_df.copy()
    price_df["Ticker"] = clean_ticker
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    price_df = price_df.dropna(subset=["Close", "High", "Low", "Open"]).sort_values("Date").reset_index(drop=True)

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

    # Compute Candidate v2 features with scale-invariant earnings yields
    featured = build_candidate_features(
        price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
    )

    if featured.empty:
        raise ValueError(f"Feature computation resulted in empty dataset for {clean_ticker}")

    latest_row = featured.iloc[[-1]].copy()
    timestamp_str = latest_row["Date"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")

    feature_names = contract["feature_names"]
    imputation_dict = contract["imputation_values"]

    X = latest_row[feature_names].copy()
    for feat in feature_names:
        fill_val = imputation_dict.get(feat, 0.0)
        X[feat] = X[feat].replace([np.inf, -np.inf], np.nan).fillna(fill_val)

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
        "model_version": contract.get("model_version", "2.0.0-candidate"),
        "candidate_id": "candidate_v2",
        "feature_snapshot": feature_snapshot,
    }

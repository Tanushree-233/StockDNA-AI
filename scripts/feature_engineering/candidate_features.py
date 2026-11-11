"""
Candidate Feature Engineering Engine for StockDNA-AI (Candidate v2).

Integrates Scale-Invariant Earnings Valuation:
- Replaces nominal `Reported_EPS` with `Earnings_Yield` = Reported_EPS / Close
- Replaces nominal `EPS_Estimate` with `Forward_Earnings_Yield` = EPS_Estimate / Close

Guarantees:
1. Exact 34-feature contract dimension maintained.
2. Identical transformation applied to batch training data and single-bar live inference.
3. Strict point-in-time correctness: all features backward-looking.
4. Isolated: Does not modify production contract or canonical features.
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd

from scripts.feature_engineering.canonical_features import (
    build_canonical_features,
)

# Explicit 34-feature contract for Candidate v2
CANDIDATE_V2_FEATURE_NAMES: List[str] = [
    "Daily_Return",
    "Return_5d",
    "Return_10d",
    "Return_21d",
    "ROC_10",
    "Close_to_SMA20",
    "Close_to_SMA50",
    "Close_to_SMA200",
    "SMA20_to_SMA50",
    "Normalized_ATR",
    "High_Low_Spread_Pct",
    "Open_Close_Spread_Pct",
    "Rolling_Volatility_20d",
    "Bollinger_PctB",
    "RSI_14",
    "Normalized_MACD",
    "Normalized_MACD_Hist",
    "Volume_Change",
    "Volume_to_MA20",
    "NIFTY_Return_1d",
    "NIFTY_Return_5d",
    "Relative_Strength_5d",
    "VIX_Level",
    "VIX_Change_5d",
    "Earnings_Yield",
    "Forward_Earnings_Yield",
    "EPS_Surprise",
    "EPS_Surprise_Pct",
    "Earnings_Event",
    "Positive_Earnings_Surprise",
    "Negative_Earnings_Surprise",
    "Days_Since_Earnings",
    "Post_Earnings_1_Week",
    "Post_Earnings_1_Month",
]


def apply_scale_invariant_earnings(
    df: pd.DataFrame,
    price_col: str = "Close",
    reported_eps_col: str = "Reported_EPS",
    eps_estimate_col: str = "EPS_Estimate",
) -> pd.DataFrame:
    """
    Computes scale-independent earnings yields from Close price and nominal EPS.
    Robust against zero, negative, or missing prices.
    """
    data = df.copy()

    close_series = data[price_col] if price_col in data.columns else pd.Series(1.0, index=data.index)
    close_safe = close_series.replace(0.0, np.nan).abs()
    close_safe = close_safe.fillna(1.0)
    # Add epsilon to prevent division by zero
    close_denom = np.where(close_safe < 1e-4, 1.0, close_safe)

    # 1. Earnings_Yield = Reported_EPS / Close
    if reported_eps_col in data.columns:
        reported_eps = data[reported_eps_col].fillna(0.0)
        data["Earnings_Yield"] = np.where(
            np.isfinite(reported_eps) & np.isfinite(close_denom),
            reported_eps / close_denom,
            0.0,
        )
    else:
        data["Earnings_Yield"] = 0.0

    # 2. Forward_Earnings_Yield = EPS_Estimate / Close
    if eps_estimate_col in data.columns:
        eps_est = data[eps_estimate_col].fillna(0.0)
        data["Forward_Earnings_Yield"] = np.where(
            np.isfinite(eps_est) & np.isfinite(close_denom),
            eps_est / close_denom,
            0.0,
        )
    else:
        data["Forward_Earnings_Yield"] = 0.0

    return data


def build_candidate_features(
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Generates the complete 34 candidate feature set for Candidate v2.
    Guarantees strict parity between batch processing and live inference.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # If canonical features already present
    if "Close_to_SMA20" in df.columns and "Reported_EPS" in df.columns:
        transformed = apply_scale_invariant_earnings(df)
        return transformed

    # Otherwise build from raw prices
    canonical = build_canonical_features(
        df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
    )
    transformed = apply_scale_invariant_earnings(canonical)
    return transformed


def extract_candidate_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly the 34 Candidate v2 model features in exact contract order.
    """
    missing = [c for c in CANDIDATE_V2_FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate feature matrix missing required columns: {missing}")
    return df[CANDIDATE_V2_FEATURE_NAMES].copy()

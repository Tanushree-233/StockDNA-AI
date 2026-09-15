"""
Canonical Scale-Invariant Feature Engineering Engine for StockDNA-AI.

Guarantees:
1. Zero nominal price contamination in model features (no Close, Open, High, Low, raw SMAs, raw EMAs).
2. Exactly generates the 34 features registered in `config.feature_registry`.
3. Strict point-in-time correctness: all indicators strictly backward-looking.
4. Parity: The exact same logic is executed for batch training data and live inference.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands, AverageTrueRange

from config.feature_registry import FEATURE_NAMES, get_feature_names


def compute_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes technical momentum, trend, and volatility indicators for a single stock series.
    Expects columns: ['Date', 'Close', 'High', 'Low', 'Open', 'Volume'].
    Assumes `df` is sorted chronologically by Date.
    """
    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]
    open_p = data["Open"]
    volume = data["Volume"]

    # --- Daily Return & Rolling Returns ---
    data["Daily_Return"] = close.pct_change(fill_method=None)
    data["Return_5d"] = close.pct_change(5, fill_method=None)
    data["Return_10d"] = close.pct_change(10, fill_method=None)
    data["Return_21d"] = close.pct_change(21, fill_method=None)
    data["ROC_10"] = ((close - close.shift(10)) / (close.shift(10) + 1e-8)) * 100.0

    # --- Moving Average Distance Ratios (Scale-Invariant) ---
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=200).mean()

    data["Close_to_SMA20"] = (close / (sma20 + 1e-8)) - 1.0
    data["Close_to_SMA50"] = (close / (sma50 + 1e-8)) - 1.0
    data["Close_to_SMA200"] = (close / (sma200 + 1e-8)) - 1.0
    data["SMA20_to_SMA50"] = (sma20 / (sma50 + 1e-8)) - 1.0

    # --- Volatility & Spreads ---
    atr_ind = AverageTrueRange(high=high, low=low, close=close, window=14)
    data["Normalized_ATR"] = atr_ind.average_true_range() / (close + 1e-8)
    data["High_Low_Spread_Pct"] = (high - low) / (close + 1e-8)
    data["Open_Close_Spread_Pct"] = (close - open_p) / (open_p + 1e-8)
    data["Rolling_Volatility_20d"] = data["Daily_Return"].rolling(window=20).std()

    # --- Bollinger Bands (%B) ---
    bb = BollingerBands(close=close, window=20, window_dev=2)
    bb_high = bb.bollinger_hband()
    bb_low = bb.bollinger_lband()
    bb_width = bb_high - bb_low
    data["Bollinger_PctB"] = np.where(
        bb_width > 1e-8,
        (close - bb_low) / bb_width,
        0.5
    )

    # --- RSI ---
    rsi = RSIIndicator(close=close, window=14)
    data["RSI_14"] = rsi.rsi()

    # --- MACD (Normalized by Close) ---
    macd_ind = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    data["Normalized_MACD"] = macd_ind.macd() / (close + 1e-8)
    data["Normalized_MACD_Hist"] = macd_ind.macd_diff() / (close + 1e-8)

    # --- Volume Features ---
    vol_ma20 = volume.rolling(window=20).mean()
    vol_shift = volume.shift(1)
    data["Volume_Change"] = np.where(
        vol_shift > 0,
        (volume - vol_shift) / vol_shift,
        0.0
    )
    data["Volume_to_MA20"] = volume / (vol_ma20 + 1e-8)

    return data


def compute_macro_features(
    stock_df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Merges benchmark (NIFTY 50) and volatility (India VIX) features onto the stock DataFrame.
    If NIFTY_Close / VIX_Close are already in stock_df, computes returns directly.
    """
    data = stock_df.copy()

    # Merge external NIFTY if provided and not already present
    if "NIFTY_Close" not in data.columns and nifty_df is not None:
        nifty = nifty_df[["Date", "Close"]].rename(columns={"Close": "NIFTY_Close"}).copy()
        nifty["Date"] = pd.to_datetime(nifty["Date"])
        data["Date"] = pd.to_datetime(data["Date"])
        data = pd.merge(data, nifty, on="Date", how="left")

    # Merge external VIX if provided and not already present
    if "VIX_Close" not in data.columns and vix_df is not None:
        vix = vix_df[["Date", "Close"]].rename(columns={"Close": "VIX_Close"}).copy()
        vix["Date"] = pd.to_datetime(vix["Date"])
        data["Date"] = pd.to_datetime(data["Date"])
        data = pd.merge(data, vix, on="Date", how="left")

    if "NIFTY_Close" in data.columns:
        data["NIFTY_Return_1d"] = data["NIFTY_Close"].pct_change(fill_method=None)
        data["NIFTY_Return_5d"] = data["NIFTY_Close"].pct_change(5, fill_method=None)
    else:
        data["NIFTY_Return_1d"] = 0.0
        data["NIFTY_Return_5d"] = 0.0

    if "Return_5d" in data.columns and "NIFTY_Return_5d" in data.columns:
        data["Relative_Strength_5d"] = data["Return_5d"] - data["NIFTY_Return_5d"]
    else:
        data["Relative_Strength_5d"] = 0.0

    if "VIX_Close" in data.columns:
        data["VIX_Level"] = data["VIX_Close"]
        data["VIX_Change_5d"] = data["VIX_Close"].pct_change(5, fill_method=None)
    else:
        data["VIX_Level"] = 15.0
        data["VIX_Change_5d"] = 0.0

    return data


def compute_internal_features(
    stock_df: pd.DataFrame,
    earnings_events_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aligns historical earnings announcement events point-in-time via backward merge.
    Guarantees zero future earnings information leaks into historical bars.
    """
    data = stock_df.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    # If already merged in source (e.g. from internal pipeline)
    if "Reported_EPS" in data.columns and "EPS_Estimate" in data.columns:
        if "EPS_Surprise_Pct" not in data.columns:
            data["EPS_Surprise_Pct"] = data["EPS_Surprise"] / (data["EPS_Estimate"].abs() + 1e-5)
        for col in ["Earnings_Event", "Positive_Earnings_Surprise", "Negative_Earnings_Surprise",
                    "Post_Earnings_1_Week", "Post_Earnings_1_Month"]:
            if col in data.columns:
                data[col] = data[col].fillna(0).astype(int)
        return data

    if earnings_events_df is not None and not earnings_events_df.empty:
        events = earnings_events_df.copy()
        events["Date"] = pd.to_datetime(events["Date"])
        
        # Standardize ticker matching
        ticker_val = str(data["Ticker"].iloc[0]).replace(".NS", "") if "Ticker" in data.columns else None
        if ticker_val:
            events["Ticker"] = events["Ticker"].astype(str).str.replace(".NS", "")
            events = events[events["Ticker"] == ticker_val].sort_values("Date")
        
        events["Last_Earnings_Date"] = events["Date"]
        
        # Point-in-time merge_asof backward
        data = data.sort_values("Date").reset_index(drop=True)
        merged = pd.merge_asof(
            data,
            events[["Date", "Last_Earnings_Date", "EPS_Estimate", "Reported_EPS", "EPS_Surprise"]],
            on="Date",
            direction="backward",
        )
        
        merged["Earnings_Event"] = (merged["Date"] == merged["Last_Earnings_Date"]).astype(int)
        merged["EPS_Estimate"] = merged["EPS_Estimate"].fillna(0.0)
        merged["Reported_EPS"] = merged["Reported_EPS"].fillna(0.0)
        merged["EPS_Surprise"] = merged["EPS_Surprise"].fillna(0.0)
        merged["EPS_Surprise_Pct"] = merged["EPS_Surprise"] / (merged["EPS_Estimate"].abs() + 1e-5)
        merged["Positive_Earnings_Surprise"] = (merged["EPS_Surprise"] > 0).astype(int)
        merged["Negative_Earnings_Surprise"] = (merged["EPS_Surprise"] < 0).astype(int)
        
        days_since = (merged["Date"] - merged["Last_Earnings_Date"]).dt.days
        merged["Days_Since_Earnings"] = days_since.fillna(999.0)
        merged["Post_Earnings_1_Week"] = merged["Days_Since_Earnings"].between(0, 5).astype(int)
        merged["Post_Earnings_1_Month"] = merged["Days_Since_Earnings"].between(0, 21).astype(int)
        
        return merged

    # Fallback if no earnings events provided
    data["Reported_EPS"] = 0.0
    data["EPS_Estimate"] = 0.0
    data["EPS_Surprise"] = 0.0
    data["EPS_Surprise_Pct"] = 0.0
    data["Earnings_Event"] = 0
    data["Positive_Earnings_Surprise"] = 0
    data["Negative_Earnings_Surprise"] = 0
    data["Days_Since_Earnings"] = 999.0
    data["Post_Earnings_1_Week"] = 0
    data["Post_Earnings_1_Month"] = 0

    return data


def build_canonical_features(
    df: pd.DataFrame,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Main entry point for generating the complete canonical feature set.
    Handles single-ticker and multi-ticker panels.
    Returns DataFrame containing metadata columns (Date, Ticker) plus all 34 model features.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    # If multi-ticker panel
    if "Ticker" in data.columns and data["Ticker"].nunique() > 1:
        processed_chunks = []
        for ticker, group in data.groupby("Ticker", sort=False):
            sorted_group = group.sort_values("Date").reset_index(drop=True)
            tech = compute_technical_features(sorted_group)
            macro = compute_macro_features(tech, nifty_df, vix_df)
            full = compute_internal_features(macro, earnings_df)
            processed_chunks.append(full)
        res = pd.concat(processed_chunks, ignore_index=True)
    else:
        sorted_group = data.sort_values("Date").reset_index(drop=True)
        tech = compute_technical_features(sorted_group)
        macro = compute_macro_features(tech, nifty_df, vix_df)
        res = compute_internal_features(macro, earnings_df)

    return res


def extract_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts strictly the registered 34 model features in the exact canonical order.
    Eliminates all nominal price columns, metadata, and future labels.
    """
    expected_cols = get_feature_names()
    missing = [col for col in expected_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required canonical features: {missing}")
    return df[expected_cols].copy()

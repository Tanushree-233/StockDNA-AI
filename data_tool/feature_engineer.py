from typing import List, Optional
import pandas as pd
import numpy as np


def compute_scale_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes purely scale-invariant technical indicators and financial ratios.
    Zero nominal price levels (e.g. raw ₹3,500 vs ₹1,500) are emitted as model features,
    guaranteeing mathematical generalization across different stocks and market regimes.
    """
    if df is None or df.empty or "Close" not in df.columns:
        raise ValueError("DataFrame must contain 'Close' price column.")

    data = df.copy()
    close = data["Close"]

    # 1. Return Horizons & Momentum
    data["Daily_Return"] = close.pct_change()
    data["Return_5d"] = close.pct_change(5)
    data["Return_10d"] = close.pct_change(10)
    data["Return_21d"] = close.pct_change(21)

    # 2. Moving Average Distance Ratios (Scale-Invariant)
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    sma200 = close.rolling(window=min(200, max(20, len(close)))).mean()

    data["Close_to_SMA20"] = (close / (sma20 + 1e-8)) - 1.0
    data["Close_to_SMA50"] = (close / (sma50 + 1e-8)) - 1.0
    data["Close_to_SMA200"] = (close / (sma200 + 1e-8)) - 1.0
    data["SMA20_to_SMA50"] = (sma20 / (sma50 + 1e-8)) - 1.0

    # 3. RSI (Naturally bounded [0, 100])
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=5).mean()
    avg_loss = loss.rolling(window=14, min_periods=5).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    data["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 4. MACD Normalized by Price
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_raw = ema12 - ema26
    macd_sig = macd_raw.ewm(span=9, adjust=False).mean()
    macd_hist = macd_raw - macd_sig

    data["Normalized_MACD"] = macd_raw / (close + 1e-8)
    data["Normalized_MACD_Signal"] = macd_sig / (close + 1e-8)
    data["Normalized_MACD_Hist"] = macd_hist / (close + 1e-8)

    # 5. Bollinger Bands (%B oscillator)
    rolling20 = close.rolling(20)
    bb_mid = rolling20.mean()
    bb_std = rolling20.std()
    bb_high = bb_mid + 2.0 * bb_std
    bb_low = bb_mid - 2.0 * bb_std
    data["Bollinger_PctB"] = (close - bb_low) / (bb_high - bb_low + 1e-8)

    # 6. Normalized Volatility & ATR
    if "High" in data.columns and "Low" in data.columns:
        high = data["High"]
        low = data["Low"]
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr14 = tr.rolling(window=14, min_periods=5).mean()
        data["Normalized_ATR"] = atr14 / (close + 1e-8)
        data["High_Low_Spread_Pct"] = (high - low) / (close + 1e-8)

    if "Open" in data.columns:
        data["Open_Close_Spread_Pct"] = (close - data["Open"]) / (close + 1e-8)

    data["Rolling_Volatility_20d"] = data["Daily_Return"].rolling(20, min_periods=5).std()

    # 7. Volume Ratios
    if "Volume" in data.columns:
        vol = data["Volume"]
        vol_ma20 = vol.rolling(20, min_periods=5).mean()
        data["Volume_Change"] = vol.pct_change()
        data["Volume_to_MA20"] = (vol / (vol_ma20 + 1e-8)) - 1.0

    return data


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard feature generator entrypoint for data_tool.
    Cleans dates, calculates scale-invariant indicators, and cleans infinite values.
    """
    if df is None or df.empty:
        raise ValueError("Cannot generate features from an empty dataset.")

    data = df.copy()
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    features_df = compute_scale_invariant_features(data)

    # Clean infinite values resulting from division by zero
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    return features_df
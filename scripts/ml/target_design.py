"""
Authoritative Production Target Generator for StockDNA-AI.

Defines the official 5-trading-day forward return target:
    Forward_Return_5d = Close[t+5] / Close[t] - 1

Label encoding:
    0: SELL  (Forward_Return_5d <= -2%)
    1: HOLD  (-2% < Forward_Return_5d < +2%)
    2: BUY   (Forward_Return_5d >= +2%)

Guarantees:
1. Panel data safety: Grouped by 'Ticker' if present, preventing cross-ticker target leakage.
2. Incomplete future rows (last `horizon` rows per ticker) dropped from supervised training.
3. Strict configurable thresholds and horizons.
"""

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

LABEL_MAP: Dict[int, str] = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}

CLASS_NAMES: Dict[str, int] = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2,
}

DEFAULT_HORIZON: int = 5
DEFAULT_BUY_THRESHOLD: float = 0.02
DEFAULT_SELL_THRESHOLD: float = -0.02


def generate_target(
    df: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    price_col: str = "Close",
    drop_unlabeled: bool = True,
) -> pd.DataFrame:
    """
    Computes the official 5-day forward return and generates 3-class target labels:
    0 = SELL, 1 = HOLD, 2 = BUY.

    Parameters
    ----------
    df : pd.DataFrame
        Input price data containing at least `price_col` and optionally 'Ticker'.
    horizon : int
        Number of trading days into the future (default: 5).
    buy_threshold : float
        Return required for BUY (default: +0.02).
    sell_threshold : float
        Return threshold for SELL (default: -0.02).
    price_col : str
        Price column name (default: "Close").
    drop_unlabeled : bool
        If True, drops rows where future price is NaN (the final `horizon` rows).

    Returns
    -------
    pd.DataFrame with 'Forward_Return_5d' and 'Target' columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if price_col not in data.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame.")

    # Compute future price grouped by Ticker if available
    target_return_col = f"Forward_Return_{horizon}d"
    
    if "Ticker" in data.columns:
        future_price = data.groupby("Ticker", sort=False)[price_col].shift(-horizon)
    else:
        future_price = data[price_col].shift(-horizon)

    data[target_return_col] = (future_price / data[price_col]) - 1.0

    conditions = [
        data[target_return_col] >= buy_threshold,
        data[target_return_col] <= sell_threshold,
    ]
    choices = [2, 0]  # 2: BUY, 0: SELL

    data["Target"] = np.select(conditions, choices, default=1)

    # In rows where future return is NaN, target should be NaN if not dropped
    data.loc[data[target_return_col].isna(), "Target"] = np.nan

    if drop_unlabeled:
        data = data.dropna(subset=[target_return_col]).reset_index(drop=True)
        data["Target"] = data["Target"].astype(int)

    return data


def get_label_distribution(
    df: pd.DataFrame, target_col: str = "Target"
) -> Dict[str, float]:
    """
    Computes the percentage distribution of SELL, HOLD, BUY classes.
    """
    if df.empty or target_col not in df.columns:
        return {}
    
    s = df[target_col].dropna().astype(int)
    counts = s.value_counts(normalize=True).to_dict()
    return {LABEL_MAP.get(k, str(k)): float(v) for k, v in sorted(counts.items())}
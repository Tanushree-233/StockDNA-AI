"""
Market Trends and Technical Charting Router for StockDNA-AI.

Serves historical OHLCV pricing, moving averages, and technical indicators for all
22 tracked Indian equities across custom time horizons (1M, 3M, 6M, 1Y, ALL).
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import os
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Query, status

from config.settings import get_company_universe, get_tickers

router = APIRouter(prefix="/market/trends", tags=["Market Trends"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PRICES_DIR = PROJECT_ROOT / "data" / "processed" / "prices"

# In-memory cache for processed price data to ensure high-speed charting
_PRICE_CACHE: Dict[str, pd.DataFrame] = {}


def _get_price_df(clean_ticker: str) -> pd.DataFrame:
    """Loads and caches cleaned price DataFrame with calculated moving averages."""
    if clean_ticker in _PRICE_CACHE:
        return _PRICE_CACHE[clean_ticker]

    csv_path = PRICES_DIR / f"{clean_ticker}.csv"
    if not csv_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historical price data for '{clean_ticker}' not found."
        )

    df = pd.read_csv(csv_path)
    if "Date" not in df.columns or "Close" not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Malformed price data for '{clean_ticker}'."
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Compute key technical indicators for charting
    df["SMA20"] = df["Close"].rolling(window=20, min_periods=1).mean()
    df["SMA50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    df["Volume_MA20"] = df["Volume"].rolling(window=20, min_periods=1).mean()

    _PRICE_CACHE[clean_ticker] = df
    return df


@router.get(
    "/summary",
    summary="Get current price and trend snapshot for all 22 tracked stocks",
    response_model=List[Dict[str, Any]],
)
def get_all_trends_summary() -> List[Dict[str, Any]]:
    """
    Returns live snapshot metrics for all 22 tracked companies including
    current price, 1-day change %, 1-month change %, 52-week high/low, and trend.
    """
    universe = get_company_universe()
    results = []

    for _, row in universe.iterrows():
        clean_ticker = row["clean_ticker"]
        company = row["company"]
        sector = row["sector"]

        try:
            df = _get_price_df(clean_ticker)
            if df.empty:
                continue

            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            month_prev = df.iloc[-22] if len(df) >= 22 else df.iloc[0]

            latest_close = float(latest["Close"])
            prev_close = float(prev["Close"])
            month_close = float(month_prev["Close"])

            change_1d = round(latest_close - prev_close, 2)
            change_pct_1d = round(((latest_close - prev_close) / prev_close) * 100, 2) if prev_close else 0.0
            change_pct_1m = round(((latest_close - month_close) / month_close) * 100, 2) if month_close else 0.0

            # 52-week (last 252 bars) high/low
            tail_252 = df.tail(252)
            high_52w = round(float(tail_252["High"].max()), 2)
            low_52w = round(float(tail_252["Low"].min()), 2)

            sma20 = float(latest["SMA20"])
            sma50 = float(latest["SMA50"])

            if latest_close >= sma50 and sma20 >= sma50:
                trend = "Bullish Uptrend"
            elif latest_close < sma50 and sma20 < sma50:
                trend = "Bearish Downtrend"
            else:
                trend = "Consolidating / Sideways"

            results.append({
                "ticker": clean_ticker,
                "company": company,
                "sector": sector,
                "latest_price": round(latest_close, 2),
                "change_1d": change_1d,
                "change_pct_1d": change_pct_1d,
                "change_pct_1m": change_pct_1m,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "trend": trend,
                "volume": int(latest.get("Volume", 0)),
                "date": latest["Date"].strftime("%Y-%m-%d"),
            })
        except Exception:
            continue

    return results


@router.get(
    "/{ticker}",
    summary="Get detailed historical price series and moving averages for interactive charting",
)
def get_ticker_trends(
    ticker: str,
    range: str = Query("6M", pattern="^(1M|3M|6M|1Y|ALL)$", description="Chart horizon range"),
) -> Dict[str, Any]:
    """
    Returns time series OHLCV data, 20-day SMA, 50-day SMA, and high/low stats for a single ticker.
    """
    clean_ticker = ticker.replace(".NS", "").strip().upper()
    universe = get_company_universe()
    match = universe[universe["clean_ticker"] == clean_ticker]

    if match.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticker '{clean_ticker}' is not in the supported 22-company universe."
        )

    company_name = match["company"].iloc[0]
    sector = match["sector"].iloc[0]

    df = _get_price_df(clean_ticker)
    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No price history found for '{clean_ticker}'."
        )

    # Determine slice window
    horizon_bars = {
        "1M": 21,
        "3M": 63,
        "6M": 126,
        "1Y": 252,
    }

    if range in horizon_bars:
        bars_count = horizon_bars[range]
        sliced_df = df.tail(bars_count).copy()
    else:  # ALL
        # If over 600 points, sample evenly to keep frontend responsive
        if len(df) > 600:
            step = max(1, len(df) // 400)
            sliced_df = pd.concat([df.iloc[::step], df.tail(1)]).drop_duplicates().sort_values("Date")
        else:
            sliced_df = df.copy()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest
    start_point = sliced_df.iloc[0]

    current_price = round(float(latest["Close"]), 2)
    prev_price = round(float(prev["Close"]), 2)
    start_price = round(float(start_point["Close"]), 2)

    change_1d = round(current_price - prev_price, 2)
    change_pct_1d = round(((current_price - prev_price) / prev_price) * 100, 2) if prev_price else 0.0

    period_change = round(current_price - start_price, 2)
    period_change_pct = round(((current_price - start_price) / start_price) * 100, 2) if start_price else 0.0

    tail_252 = df.tail(252)
    high_52w = round(float(tail_252["High"].max()), 2)
    low_52w = round(float(tail_252["Low"].min()), 2)
    avg_vol = int(tail_252["Volume"].mean()) if not tail_252.empty else 0

    sma20 = round(float(latest["SMA20"]), 2)
    sma50 = round(float(latest["SMA50"]), 2)

    if current_price >= sma50 and sma20 >= sma50:
        trend_status = "Bullish Uptrend"
        trend_desc = "Trading comfortably above 20-day and 50-day moving averages with positive upward momentum."
    elif current_price < sma50 and sma20 < sma50:
        trend_status = "Bearish Downtrend"
        trend_desc = "Trading below moving averages indicating persistent downward pressure and overhead supply."
    else:
        trend_status = "Consolidating / Sideways"
        trend_desc = "Price is oscillating between support and resistance, showing balanced buying and selling."

    series = []
    for _, r in sliced_df.iterrows():
        series.append({
            "date": r["Date"].strftime("%Y-%m-%d"),
            "open": round(float(r["Open"]), 2),
            "high": round(float(r["High"]), 2),
            "low": round(float(r["Low"]), 2),
            "close": round(float(r["Close"]), 2),
            "volume": int(r.get("Volume", 0)),
            "sma20": round(float(r["SMA20"]), 2) if not np.isnan(r["SMA20"]) else None,
            "sma50": round(float(r["SMA50"]), 2) if not np.isnan(r["SMA50"]) else None,
        })

    return {
        "ticker": clean_ticker,
        "company": company_name,
        "sector": sector,
        "range": range,
        "current_price": current_price,
        "change_1d": change_1d,
        "change_pct_1d": change_pct_1d,
        "period_change": period_change,
        "period_change_pct": period_change_pct,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "avg_volume_20d": avg_vol,
        "sma20": sma20,
        "sma50": sma50,
        "trend_status": trend_status,
        "trend_description": trend_desc,
        "data_points_count": len(series),
        "series": series,
    }

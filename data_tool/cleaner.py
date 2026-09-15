from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standard cleaning pipeline for tabular data while preserving column names
    and standardizing core aliases.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()
    data = data.dropna(how="all")
    # Drop unnamed empty columns (common in messy CSV exports)
    unnamed_empty = [c for c in data.columns if str(c).startswith("Unnamed") and data[c].isna().all()]
    if unnamed_empty:
        data = data.drop(columns=unnamed_empty)
    data.columns = [str(c).strip() for c in data.columns]

    column_mapping = {
        "date": "Date",
        "DATE": "Date",
        "Price": "Date",  # Yahoo finance raw files often have 'Price' header for Date
        "open": "Open",
        "OPEN": "Open",
        "high": "High",
        "HIGH": "High",
        "low": "Low",
        "LOW": "Low",
        "close": "Close",
        "CLOSE": "Close",
        "volume": "Volume",
        "VOLUME": "Volume",
        "adj close": "Adj Close",
        "Adj_Close": "Adj Close",
        "ticker": "Ticker",
        "TICKER": "Ticker",
        "symbol": "Ticker",
    }

    data = data.rename(columns=column_mapping)
    data = data.drop_duplicates()
    return data.reset_index(drop=True)


def clean_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans OHLCV stock price or market index time series.
    Enforces numeric pricing, removes invalid metadata rows, and sorts chronologically.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = clean_data(df)

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])
        data = data.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in data.columns:
            # Handle string formatting like commas or currency symbols
            if data[col].dtype == object:
                data[col] = (
                    data[col]
                    .astype(str)
                    .str.replace(r"[^\d.\-]", "", regex=True)
                )
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows where critical price columns are missing or non-positive
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in data.columns]
    if price_cols:
        data = data.dropna(subset=price_cols)
        # Remove non-positive price anomalies
        for col in price_cols:
            data = data[data[col] > 0]
    else:
        # If no price columns are present in market data, it is not usable
        return pd.DataFrame(columns=data.columns)

    return data.reset_index(drop=True)


def clean_earnings_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and standardizes quarterly earnings events and surprises.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = clean_data(df)

    date_candidates = ["Earnings Date", "EarningsDate", "Date", "announcement_date"]
    for dc in date_candidates:
        if dc in data.columns:
            data = data.rename(columns={dc: "Date"})
            break

    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
        data = data.dropna(subset=["Date"])

    # Standardize surprise and EPS column aliases
    eps_mapping = {
        "Reported EPS": "Reported_EPS",
        "EPS Estimate": "EPS_Estimate",
        "Surprise(%)": "EPS_Surprise_Pct",
        "Surprise": "EPS_Surprise",
    }
    data = data.rename(columns=eps_mapping)

    numeric_fields = ["Reported_EPS", "EPS_Estimate", "EPS_Surprise_Pct", "EPS_Surprise"]
    for field in numeric_fields:
        if field in data.columns:
            data[field] = pd.to_numeric(data[field], errors="coerce")

    return data.reset_index(drop=True)


def clean_news_data(news_input: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Standardizes raw news articles or RSS feed items.
    """
    if isinstance(news_input, list):
        if not news_input:
            return pd.DataFrame()
        df = pd.DataFrame(news_input)
    elif isinstance(news_input, pd.DataFrame):
        df = news_input.copy()
    else:
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = clean_data(df)

    # Standardize column names
    col_map = {
        "title": "Headline",
        "headline": "Headline",
        "published": "Published_At",
        "pubDate": "Published_At",
        "source": "Source",
        "link": "URL",
        "url": "URL",
    }
    df = df.rename(columns=col_map)

    if "Headline" in df.columns:
        df["Headline"] = df["Headline"].astype(str).str.strip()
        df = df[df["Headline"].str.len() > 0]

    if "Published_At" in df.columns:
        df["Published_At"] = pd.to_datetime(df["Published_At"], errors="coerce")
        df["Date"] = df["Published_At"].dt.normalize()
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df.reset_index(drop=True)
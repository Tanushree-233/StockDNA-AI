import pandas as pd

from scripts.collectors.download_prices import download_single_stock
from scripts.feature_engineering.calculate_indicators import calculate_indicators


def get_price_features(ticker: str):
    """
    Download the latest stock data, calculate technical indicators,
    and return the latest feature row.
    """

    # Download latest historical data
    df = download_single_stock(ticker)

    if df.empty:
        raise ValueError("Ticker not found.")

    if df.empty:
        raise ValueError(f"No data found for {ticker}")

    # Ensure Date column is datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Convert numeric columns
    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Calculate all technical indicators
    df = calculate_indicators(df)

    if df.empty:
        raise ValueError(
            "Not enough historical data to calculate indicators."
        )

    # Return latest row as dictionary
    latest = df.iloc[-1].to_dict()

    return latest
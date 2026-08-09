import sys
from pathlib import Path

import joblib
import pandas as pd
import yfinance as yf

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml.predict.predict_stock import predict


# ============================================================
# CONFIG
# ============================================================

TICKERS = {
    "INFY": "INFY.NS",
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
}

HISTORY_PERIOD = "2y"

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "internal"
    / "live_predictions.csv"
)


# ============================================================
# LOAD EXISTING FEATURE DATA
# ============================================================

FEATURE_DATA = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "internal"
    / "final_ml_dataset.csv"
)


# ============================================================
# DOWNLOAD LATEST MARKET DATA
# ============================================================

def download_market_data(ticker):

    print(f"Downloading market data: {ticker}")

    df = yf.download(
        ticker,
        period=HISTORY_PERIOD,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for {ticker}"
        )

    # yfinance may return MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    df["Date"] = pd.to_datetime(df["Date"])

    return df


# ============================================================
# BUILD TECHNICAL FEATURES
# ============================================================

def add_technical_features(df):

    df = df.copy()

    close = df["Close"]
    volume = df["Volume"]

    # Returns
    df["Daily_Return"] = close.pct_change()

    # Moving averages
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA100"] = close.rolling(100).mean()
    df["SMA200"] = close.rolling(200).mean()

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    # RSI
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_Histogram"] = (
        df["MACD"] -
        df["MACD_Signal"]
    )

    # Bollinger Bands
    rolling20 = close.rolling(20)

    df["BB_Middle"] = rolling20.mean()
    bb_std = rolling20.std()

    df["BB_High"] = (
        df["BB_Middle"] +
        2 * bb_std
    )

    df["BB_Low"] = (
        df["BB_Middle"] -
        2 * bb_std
    )

    # ATR
    high_low = df["High"] - df["Low"]

    high_close = (
        df["High"] -
        close.shift()
    ).abs()

    low_close = (
        df["Low"] -
        close.shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close,
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.rolling(14).mean()

    # Volatility
    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(20)
        .std()
    )

    # Lags
    df["Lag_Close_1"] = close.shift(1)
    df["Lag_Close_3"] = close.shift(3)
    df["Lag_Close_5"] = close.shift(5)

    # Momentum
    df["Momentum_5"] = close.pct_change(5)
    df["Momentum_10"] = close.pct_change(10)
    df["Momentum_20"] = close.pct_change(20)

    # ROC
    df["ROC"] = (
        (close / close.shift(10)) - 1
    )

    # Volume
    df["Volume_Change"] = volume.pct_change()
    df["Lag_Volume_1"] = volume.shift(1)

    df["Volume_MA20"] = (
        volume.rolling(20).mean()
    )

    # Spreads
    df["High_Low_Spread"] = (
        df["High"] - df["Low"]
    )

    df["Open_Close_Spread"] = (
        df["Open"] - df["Close"]
    )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("STOCKDNA-AI LIVE PREDICTION")
    print("=" * 60)

    # --------------------------------------------------------
    # IMPORTANT
    #
    # We use the existing processed dataset for financial,
    # earnings, NIFTY and VIX features.
    #
    # Fresh market prices are added separately.
    # --------------------------------------------------------

    historical = pd.read_csv(
        FEATURE_DATA
    )

    historical["Date"] = pd.to_datetime(
        historical["Date"]
    )

    predictions = []

    for company, yahoo_ticker in TICKERS.items():

        print(
            f"\nProcessing {company}..."
        )

        # ----------------------------------------------------
        # Download fresh market prices
        # ----------------------------------------------------

        market = download_market_data(
            yahoo_ticker
        )

        market = add_technical_features(
            market
        )

        market["Ticker"] = company

        # ----------------------------------------------------
        # Get latest market row
        # ----------------------------------------------------

        latest_market = (
            market
            .sort_values("Date")
            .tail(1)
            .copy()
        )

        latest_date = latest_market[
            "Date"
        ].iloc[0]

        # ----------------------------------------------------
        # Existing processed features
        #
        # We take the most recent historical feature row
        # available for this ticker.
        # ----------------------------------------------------

        existing = historical[
            historical["Ticker"] == company
        ].sort_values("Date")

        if existing.empty:
            raise ValueError(
                f"No existing feature data for {company}"
            )

        latest_existing = (
            existing
            .tail(1)
            .copy()
        )

        # ----------------------------------------------------
        # Replace market-derived columns with fresh values
        # ----------------------------------------------------

        technical_columns = [
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
            "Daily_Return",
            "SMA20",
            "SMA50",
            "EMA20",
            "EMA50",
            "RSI",
            "MACD",
            "MACD_Signal",
            "MACD_Histogram",
            "BB_High",
            "BB_Low",
            "BB_Middle",
            "ATR",
            "Volatility",
            "Lag_Close_1",
            "Lag_Close_3",
            "Lag_Close_5",
            "Momentum_5",
            "Momentum_10",
            "ROC",
            "Volume_Change",
            "High_Low_Spread",
            "SMA100",
            "SMA200",
            "EMA100",
            "Momentum_20",
            "Lag_Volume_1",
            "Volume_MA20",
            "Open_Close_Spread",
        ]

        for column in technical_columns:

            if column in latest_market.columns:

                latest_existing[column] = (
                    latest_market[column].iloc[0]
                )

        latest_existing["Date"] = latest_date
        latest_existing["Ticker"] = company

        # ----------------------------------------------------
        # Run model
        # ----------------------------------------------------

        result = predict(
            latest_existing
        )

        predictions.append(result)

    # --------------------------------------------------------
    # Combine predictions
    # --------------------------------------------------------

    final = pd.concat(
        predictions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    final.to_csv(
        OUTPUT,
        index=False
    )

    print("\n" + "=" * 60)
    print("LIVE PREDICTIONS")
    print("=" * 60)

    print(
        final.to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT)


if __name__ == "__main__":
    main()
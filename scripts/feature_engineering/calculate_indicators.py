import os
import pandas as pd
import numpy as np

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FOLDER = "data/raw/prices"
OUTPUT_FOLDER = "data/processed/features"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# CLEAN RAW STOCK DATA
# ============================================================

def clean_raw_data(df, filename):
    """
    Clean the raw stock CSV files.

    The raw files currently have this structure:

        Price, Close, High, Low, Open, Volume
        Ticker, INFY.NS, INFY.NS, ...
        Date, NaN, NaN, ...
        2018-01-01, 406..., 410..., ...

    Therefore:
        Price -> Date
    """

    print(f"\nCleaning {filename}...")

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    print("Original columns:")
    print(df.columns.tolist())

    # --------------------------------------------------------
    # Handle current raw format
    # --------------------------------------------------------

    if "Price" in df.columns and "Date" not in df.columns:

        print("Detected Yahoo-style raw format.")
        print("Converting 'Price' column to 'Date'.")

        df = df.rename(
            columns={
                "Price": "Date"
            }
        )

    # --------------------------------------------------------
    # Make sure required columns exist
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{filename}: Missing required columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Remove metadata rows
    # --------------------------------------------------------

    # In your raw file:
    #
    # Date
    # Ticker
    #
    # are metadata rows.
    #
    # We keep only rows where Date can be converted
    # into a real datetime.

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    before = len(df)

    df = df.dropna(
        subset=["Date"]
    ).copy()

    after = len(df)

    print(
        f"Removed {before - after} metadata/invalid rows."
    )

    # --------------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_columns:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(
                r"[^\d.\-]",
                "",
                regex=True
            )
        )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows with invalid prices
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    ).copy()

    # --------------------------------------------------------
    # Sort by date
    # --------------------------------------------------------

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Add ticker
    # --------------------------------------------------------

    ticker = filename.replace(
        ".csv",
        ""
    )

    df["Ticker"] = ticker

    print(
        f"Cleaned rows: {len(df)}"
    )

    print(
        f"Date range: "
        f"{df['Date'].min()} -> "
        f"{df['Date'].max()}"
    )

    return df


# ============================================================
# CALCULATE TECHNICAL INDICATORS
# ============================================================

def calculate_indicators(df):

    """
    Calculate technical indicators and target variables.
    """

    # ========================================================
    # BASIC PRICE DATA
    # ========================================================

    # Daily return

    df["Daily_Return"] = (
        df["Close"]
        .pct_change(fill_method=None)
    )


    # ========================================================
    # SIMPLE MOVING AVERAGES
    # ========================================================

    df["SMA20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["SMA50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["SMA100"] = (
        df["Close"]
        .rolling(window=100)
        .mean()
    )

    df["SMA200"] = (
        df["Close"]
        .rolling(window=200)
        .mean()
    )


    # ========================================================
    # EXPONENTIAL MOVING AVERAGES
    # ========================================================

    df["EMA20"] = EMAIndicator(
        close=df["Close"],
        window=20
    ).ema_indicator()

    df["EMA50"] = EMAIndicator(
        close=df["Close"],
        window=50
    ).ema_indicator()

    df["EMA100"] = EMAIndicator(
        close=df["Close"],
        window=100
    ).ema_indicator()


    # ========================================================
    # RSI
    # ========================================================

    df["RSI"] = RSIIndicator(
        close=df["Close"],
        window=14
    ).rsi()


    # ========================================================
    # MACD
    # ========================================================

    macd = MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = (
        macd.macd_signal()
    )

    df["MACD_Histogram"] = (
        macd.macd_diff()
    )


    # ========================================================
    # BOLLINGER BANDS
    # ========================================================

    bb = BollingerBands(
        close=df["Close"],
        window=20
    )

    df["BB_High"] = (
        bb.bollinger_hband()
    )

    df["BB_Low"] = (
        bb.bollinger_lband()
    )

    df["BB_Middle"] = (
        bb.bollinger_mavg()
    )


    # ========================================================
    # ATR
    # ========================================================

    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["ATR"] = (
        atr.average_true_range()
    )


    # ========================================================
    # VOLATILITY
    # ========================================================

    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=20)
        .std()
    )


    # ========================================================
    # LAG FEATURES
    # ========================================================

    df["Lag_Close_1"] = (
        df["Close"].shift(1)
    )

    df["Lag_Close_3"] = (
        df["Close"].shift(3)
    )

    df["Lag_Close_5"] = (
        df["Close"].shift(5)
    )

    df["Lag_Volume_1"] = (
        df["Volume"].shift(1)
    )


    # ========================================================
    # MOMENTUM FEATURES
    # ========================================================

    df["Momentum_5"] = (
        df["Close"]
        - df["Close"].shift(5)
    )

    df["Momentum_10"] = (
        df["Close"]
        - df["Close"].shift(10)
    )

    df["Momentum_20"] = (
        df["Close"]
        - df["Close"].shift(20)
    )


    # ========================================================
    # RATE OF CHANGE
    # ========================================================

    df["ROC"] = (
        (
            df["Close"]
            - df["Close"].shift(10)
        )
        / df["Close"].shift(10)
    ) * 100


    # ========================================================
    # VOLUME FEATURES
    # ========================================================

    df["Volume_Change"] = (
        df["Volume"]
        .pct_change(fill_method=None)
    )

    df["Volume_MA20"] = (
        df["Volume"]
        .rolling(window=20)
        .mean()
    )


    # ========================================================
    # PRICE ACTION FEATURES
    # ========================================================

    df["High_Low_Spread"] = (
        df["High"]
        - df["Low"]
    )

    df["Open_Close_Spread"] = (
        df["Open"]
        - df["Close"]
    )


    # ========================================================
    # TARGET RETURN
    # ========================================================

    # IMPORTANT:
    #
    # This uses NEXT DAY'S CLOSE.
    #
    # This is the target, NOT a model feature.
    #
    # It must not be included in X during training.

    df["Target_Return"] = (
        df["Close"].shift(-1)
        - df["Close"]
    ) / df["Close"]


    # ========================================================
    # TARGET CLASSIFICATION
    # ========================================================

    def classify_target(return_value):

        if return_value > 0.01:
            return "BUY"

        elif return_value < -0.01:
            return "SELL"

        else:
            return "HOLD"


    df["Target"] = (
        df["Target_Return"]
        .apply(classify_target)
    )


    # ========================================================
    # REMOVE INFINITE VALUES
    # ========================================================

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )


    # ========================================================
    # REMOVE ROWS WITH MISSING FEATURES
    # ========================================================

    # Technical indicators such as SMA200 need
    # approximately 200 observations before they
    # become valid.

    before = len(df)

    df = df.dropna().copy()

    after = len(df)

    print(
        f"Removed {before - after} rows "
        f"because of insufficient history/NaN values."
    )


    # ========================================================
    # RESET INDEX
    # ========================================================

    df = df.reset_index(
        drop=True
    )

    return df


# ============================================================
# PROCESS ALL STOCKS
# ============================================================

def process_all_stocks():

    print("\n")
    print("=" * 60)
    print("CALCULATING STOCK FEATURES")
    print("=" * 60)

    processed_count = 0

    for file in os.listdir(INPUT_FOLDER):

        if not file.endswith(".csv"):
            continue

        print("\n" + "-" * 60)
        print(f"Processing {file}...")

        input_path = os.path.join(
            INPUT_FOLDER,
            file
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            file.replace(
                ".csv",
                "_features.csv"
            )
        )

        try:

            # =================================================
            # READ RAW CSV
            # =================================================

            df = pd.read_csv(
                input_path
            )

            print(
                f"Original rows: {len(df)}"
            )


            # =================================================
            # CLEAN RAW DATA
            # =================================================

            df = clean_raw_data(
                df,
                file
            )


            # =================================================
            # CALCULATE FEATURES
            # =================================================

            df = calculate_indicators(
                df
            )


            # =================================================
            # SAVE
            # =================================================

            df.to_csv(
                output_path,
                index=False
            )

            print(
                f"Final rows: {len(df)}"
            )

            print(
                f"Final columns: {len(df.columns)}"
            )

            print(
                f"Saved: {output_path}"
            )

            processed_count += 1

        except Exception as e:

            print(
                f"ERROR processing {file}: {e}"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    print(
        f"Successfully processed: "
        f"{processed_count} stock files"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    process_all_stocks()
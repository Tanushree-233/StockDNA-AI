import os
import pandas as pd

from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

INPUT_FOLDER = "data/raw/prices"
OUTPUT_FOLDER = "data/processed/features"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calculate_indicators(df):
    """
    Calculate technical indicators for a stock DataFrame.
    """
    # 1. Clean and convert price columns to numeric data types
    # This removes commas or special characters and forces strings into numbers
    print(df.dtypes)
    print(df.head())



    price_cols = ["Open", "High", "Low", "Close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # 2. Drop rows where Close is missing after conversion to prevent crashes
    df = df.dropna(subset=["Close"]).copy()

    # Daily Return (fill_method=None removes the deprecation warning)
    df["Daily_Return"] = df["Close"].pct_change(fill_method=None)

    # Simple Moving Averages
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()

    # Exponential Moving Averages
    df["EMA20"] = EMAIndicator(close=df["Close"], window=20).ema_indicator()
    df["EMA50"] = EMAIndicator(close=df["Close"], window=50).ema_indicator()

    # RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()

    # MACD
    macd = MACD(close=df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Histogram"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(close=df["Close"], window=20)
    df["BB_High"] = bb.bollinger_hband()
    df["BB_Low"] = bb.bollinger_lband()
    df["BB_Middle"] = bb.bollinger_mavg()

    # ATR
    atr = AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )
    df["ATR"] = atr.average_true_range()

    # Rolling Volatility (20-day)
    df["Volatility"] = df["Daily_Return"].rolling(window=20).std()

    # Lag Features
    df["Lag_Close_1"] = df["Close"].shift(1)
    df["Lag_Close_3"] = df["Close"].shift(3)
    df["Lag_Close_5"] = df["Close"].shift(5)

    # Momentum
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)

    # Rate of Change
    df["ROC"] = (
        (df["Close"] - df["Close"].shift(10))
        / df["Close"].shift(10)
    ) * 100

    # Volume Change
    df["Volume_Change"] = df["Volume"].pct_change()

    # Daily Price Spread
    df["High_Low_Spread"] = df["High"] - df["Low"]

    # Target (Next Day Return)
    df["Target_Return"] = df["Close"].shift(-1) / df["Close"] - 1

    def classify(x):

        if x > 0.01:
            return "BUY"

        elif x < -0.01:
            return "SELL"

        else:
            return "HOLD"

    df["Target"] = df["Target_Return"].apply(classify)

    return df

def process_all_stocks():

    for file in os.listdir(INPUT_FOLDER):

        if file.endswith(".csv"):

            print(f"Processing {file}...")

            input_path = os.path.join(INPUT_FOLDER, file)

            output_path = os.path.join(
                OUTPUT_FOLDER,
                file.replace(".csv", "_features.csv")
            )

            df = pd.read_csv(input_path, skiprows=[1])

            df.rename(columns={"Price": "Date"}, inplace=True)

            numeric_columns = ["Open", "High", "Low", "Close", "Volume"]

            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df.dropna(inplace=True)

            df = calculate_indicators(df)

            df.to_csv(output_path, index=False)

            print(f"Saved: {output_path}")

if __name__ == "__main__":
    process_all_stocks()

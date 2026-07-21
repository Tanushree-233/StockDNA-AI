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

            df = pd.read_csv(input_path)

            df = calculate_indicators(df)

            df.to_csv(output_path, index=False)

            print(f"Saved: {output_path}")

if __name__ == "__main__":
    process_all_stocks()

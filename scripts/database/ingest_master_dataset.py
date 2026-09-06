import os
import pandas as pd

from sqlalchemy.orm import Session

from backend.database.database import (
    SessionLocal,
    engine,
    Base
)

from backend.database.models import StockData


DATASET_PATH = "data/final/master_dataset.csv"


def ingest_master_dataset():

    print("=" * 60)
    print("STOCK DATA DATABASE INGESTION")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):

        raise FileNotFoundError(
            f"Master dataset not found: {DATASET_PATH}"
        )

    print(
        f"\nLoading dataset: {DATASET_PATH}"
    )

    df = pd.read_csv(
        DATASET_PATH
    )

    if df.empty:

        raise ValueError(
            "Master dataset is empty."
        )

    print(
        f"Rows loaded: {len(df)}"
    )

    # Normalize dates
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(
        subset=["Date"]
    )

    # Normalize tickers
    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Remove duplicate Ticker + Date records
    df = df.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last"
    )

    print(
        f"Rows after cleaning: {len(df)}"
    )

    # Create any missing database tables
    print(
        "\nChecking database tables..."
    )

    Base.metadata.create_all(
        bind=engine
    )

    print(
        "Database tables are ready."
    )

    db: Session = SessionLocal()

    try:

        print(
            "\nClearing existing stock_data records..."
        )

        db.query(
            StockData
        ).delete()

        db.commit()

        print(
            "Existing stock_data records removed."
        )

        records = []

        for _, row in df.iterrows():

            record = StockData(

                # Identification
                Date=row["Date"].date(),
                Ticker=row["Ticker"],

                # Market data
                Close=row["Close"],
                High=row["High"],
                Low=row["Low"],
                Open=row["Open"],
                Volume=row["Volume"],

                # Return / trend features
                Daily_Return=row["Daily_Return"],

                SMA20=row["SMA20"],
                SMA50=row["SMA50"],
                SMA100=row["SMA100"],
                SMA200=row["SMA200"],

                EMA20=row["EMA20"],
                EMA50=row["EMA50"],
                EMA100=row["EMA100"],

                # Momentum indicators
                RSI=row["RSI"],

                MACD=row["MACD"],
                MACD_Signal=row["MACD_Signal"],
                MACD_Histogram=row["MACD_Histogram"],

                # Bollinger Bands
                BB_High=row["BB_High"],
                BB_Low=row["BB_Low"],
                BB_Middle=row["BB_Middle"],

                # Volatility
                ATR=row["ATR"],
                Volatility=row["Volatility"],

                # Lag features
                Lag_Close_1=row["Lag_Close_1"],
                Lag_Close_3=row["Lag_Close_3"],
                Lag_Close_5=row["Lag_Close_5"],

                Lag_Volume_1=row["Lag_Volume_1"],

                # Momentum
                Momentum_5=row["Momentum_5"],
                Momentum_10=row["Momentum_10"],
                Momentum_20=row["Momentum_20"],

                ROC=row["ROC"],

                # Volume features
                Volume_Change=row["Volume_Change"],
                Volume_MA20=row["Volume_MA20"],

                # Price spread features
                High_Low_Spread=row["High_Low_Spread"],
                Open_Close_Spread=row["Open_Close_Spread"],

                # Target
                Target_Return=row["Target_Return"],
                Target=row["Target"]
            )

            records.append(
                record
            )

        print(
            f"\nPrepared {len(records)} database records."
        )

        # Insert in batches
        batch_size = 500

        for i in range(
            0,
            len(records),
            batch_size
        ):

            batch = records[
                i:i + batch_size
            ]

            db.add_all(
                batch
            )

            db.commit()

            inserted = min(
                i + batch_size,
                len(records)
            )

            print(
                f"Inserted {inserted} / "
                f"{len(records)} records"
            )

        total_records = db.query(
            StockData
        ).count()

        print(
            "\n" + "=" * 60
        )

        print(
            "DATABASE INGESTION COMPLETED"
        )

        print(
            "=" * 60
        )

        print(
            f"Records in stock_data: "
            f"{total_records}"
        )

    except Exception:

        db.rollback()

        raise

    finally:

        db.close()


if __name__ == "__main__":
    ingest_master_dataset()
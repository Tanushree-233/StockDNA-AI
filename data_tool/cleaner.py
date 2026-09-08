import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans incoming data while preserving the column names
    expected by the StockDNA ML pipeline.
    """

    df = df.copy()

    df = df.dropna(how="all")

    df = df.dropna(axis=1, how="all")

    df.columns = df.columns.str.strip()

    column_mapping = {
        "date": "Date",
        "DATE": "Date",

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

        "ticker": "Ticker",
        "TICKER": "Ticker",
    }

    df = df.rename(columns=column_mapping)

    df = df.drop_duplicates()

    return df.reset_index(drop=True)
import os
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from scripts.utils.config import load_config
from scripts.utils.logger import logger

import pandas as pd


def download_single_stock(
    ticker: str,
    period: str = "2y"
) -> pd.DataFrame:
    """
    Download historical data for a single ticker.
    """

    if not ticker.endswith(".NS"):
        ticker += ".NS"

    df = yf.download(
        ticker,
        period=period,
        progress=False,
        auto_adjust=False,
        group_by="column"
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df

def download_prices():

    config = load_config()

    companies = pd.read_csv(config["paths"]["companies"])

    save_folder = config["paths"]["prices"]
    os.makedirs(save_folder, exist_ok=True)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    for _, row in tqdm(
        companies.iterrows(),
        total=len(companies),
        desc="Downloading Prices"
    ):

        ticker = row["ticker"]

        filename = ticker.replace(".NS", "") + ".csv"
        filepath = os.path.join(save_folder, filename)

        # Skip if file already exists
        existing = None

        if os.path.exists(filepath):
            try:
                existing = pd.read_csv(filepath)

                if not existing.empty and "Date" in existing.columns:
                    existing["Date"] = pd.to_datetime(
                        existing["Date"],
                        errors="coerce"
                    )

                    last_date = existing["Date"].max()

                    if pd.notna(last_date):
                        download_start = (
                            last_date + pd.Timedelta(days=1)
                        ).strftime("%Y-%m-%d")
                    else:
                        download_start = start
                else:
                    download_start = start

            except Exception as e:
                logger.warning(
                    f"Could not read existing {filename}: {e}"
                )
                existing = None
                download_start = start
        else:
            download_start = start

        logger.info(f"Downloading {ticker}")

        try:

            df = yf.download(
                ticker,
                start=download_start,
                end=end,
                progress=False,
                auto_adjust=False,
                group_by="column"
            )

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if df.empty:
                logger.error(f"No data found for {ticker}")
                continue

            if existing is not None and not existing.empty:

                existing["Date"] = pd.to_datetime(
                    existing["Date"],
                    errors="coerce"
                )

                df["Date"] = pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                )

                df = pd.concat(
                    [existing, df],
                    ignore_index=True
                )

                df = df.drop_duplicates(
                    subset=["Date"],
                    keep="last"
                )

                df = df.sort_values("Date").reset_index(
                    drop=True
                )

            df.to_csv(filepath, index=False)
            df.to_csv(filepath)

            logger.info(f"Saved {filename}")

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
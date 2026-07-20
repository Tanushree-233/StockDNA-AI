import os
import time

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from scripts.utils.config import load_config
from scripts.utils.logger import logger


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
        if os.path.exists(filepath):
            logger.info(f"{filename} already exists. Skipping...")
            continue

        logger.info(f"Downloading {ticker}")

        try:

            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False
            )

            if df.empty:
                logger.error(f"No data found for {ticker}")
                continue

            df.to_csv(filepath)

            logger.info(f"Saved {filename}")

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
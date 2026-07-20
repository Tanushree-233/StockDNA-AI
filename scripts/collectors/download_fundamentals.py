import os
import pandas as pd
import yfinance as yf

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def download_fundamentals():

    config = load_config()

    companies = pd.read_csv(config["paths"]["companies"])

    save_folder = "data/raw/fundamentals"

    os.makedirs(save_folder, exist_ok=True)

    fundamental_data = []

    for _, row in companies.iterrows():

        ticker = row["ticker"]

        logger.info(f"Downloading fundamentals for {ticker}")

        try:

            stock = yf.Ticker(ticker)

            info = stock.info

            fundamental_data.append({
                "Ticker": ticker,
                "Company": info.get("longName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "MarketCap": info.get("marketCap"),
                "PE_Ratio": info.get("trailingPE"),
                "BookValue": info.get("bookValue"),
                "DividendYield": info.get("dividendYield"),
                "Beta": info.get("beta")
            })

        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")

    df = pd.DataFrame(fundamental_data)

    df.to_csv(
        os.path.join(save_folder, "fundamentals.csv"),
        index=False
    )

    logger.info("Fundamentals downloaded and saved.")
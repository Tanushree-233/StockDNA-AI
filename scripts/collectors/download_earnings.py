import os
import pandas as pd
import yfinance as yf

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def download_earnings():

    config = load_config()

    companies = pd.read_csv(config["paths"]["companies"])

    save_folder = "data/raw/earnings"

    os.makedirs(save_folder, exist_ok=True)

    earnings_data = []

    for _, row in companies.iterrows():

        ticker = row["ticker"]

        logger.info(f"Downloading earnings for {ticker}")

        try:

            stock = yf.Ticker(ticker)

            info = stock.info

            earnings_data.append({

                "Ticker": ticker,
                "Company": info.get("longName"),
                "CurrentPrice": info.get("currentPrice"),
                "TrailingEPS": info.get("trailingEps"),
                "ForwardEPS": info.get("forwardEps"),
                "PE_Ratio": info.get("trailingPE"),
                "ForwardPE": info.get("forwardPE"),
                "EarningsQuarterlyGrowth": info.get("earningsQuarterlyGrowth")

            })

        except Exception as e:

            logger.error(f"Error downloading {ticker}: {e}")

    df = pd.DataFrame(earnings_data)

    df.to_csv(
        os.path.join(save_folder, "earnings.csv"),
        index=False
    )

    logger.info("Earnings downloaded and saved.")
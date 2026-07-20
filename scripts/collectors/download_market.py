import os
import yfinance as yf

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def download_market():

    config = load_config()

    market_folder = config["paths"]["market"]

    os.makedirs(market_folder, exist_ok=True)

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    indices = {
        "nifty50": config["market"]["nifty"],
        "indiavix": config["market"]["vix"]
    }

    for name, symbol in indices.items():

        logger.info(f"Downloading {name}")

        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False
        )

        if df.empty:
            logger.error(f"No data for {name}")
            continue

        df.to_csv(
            os.path.join(
                market_folder,
                f"{name}.csv"
            )
        )

        logger.info(f"Saved {name}.csv")
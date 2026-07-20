import os
import yfinance as yf
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def download_macro():

    config = load_config()

    save_folder = "data/raw/macro"

    os.makedirs(save_folder, exist_ok=True)

    macro_symbols = {
        "IndiaVIX": "^INDIAVIX",
        "USDINR": "INR=X",
        "CrudeOil": "CL=F",
        "Gold": "GC=F"
    }

    start = config["data"]["start_date"]
    end = config["data"]["end_date"]

    for name, symbol in macro_symbols.items():

        logger.info(f"Downloading {name}")

        try:

            df = yf.download(
                symbol,
                start=start,
                end=end,
                progress=False
            )

            if df.empty:
                logger.warning(f"No data found for {name}")
                continue

            df.to_csv(
                os.path.join(save_folder, f"{name}.csv")
            )

            logger.info(f"Saved {name}.csv")

        except Exception as e:

            logger.error(f"Error downloading {name}: {e}")

    logger.info("Macro data downloaded successfully.")
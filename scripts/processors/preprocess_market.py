import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_market():

    config = load_config()

    raw_folder = config["paths"]["market"]
    processed_folder = "data/processed/market"

    os.makedirs(processed_folder, exist_ok=True)

    files = os.listdir(raw_folder)

    for file in files:

        if not file.endswith(".csv"):
            continue

        logger.info(f"Processing {file}")

        df = pd.read_csv(os.path.join(raw_folder, file))

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove empty rows
        df = df.dropna(how="all")

        # Fill missing values
        df = df.ffill().bfill()

        # Convert Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")

        df.to_csv(
            os.path.join(processed_folder, file),
            index=False
        )

        logger.info(f"Saved cleaned {file}")

    logger.info("Market preprocessing completed.")
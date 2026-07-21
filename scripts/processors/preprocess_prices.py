import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_prices():

    config = load_config()

    raw_folder = config["paths"]["prices"]
    processed_folder = "data/processed/prices"

    os.makedirs(processed_folder, exist_ok=True)

    files = os.listdir(raw_folder)

    for file in files:

        if not file.endswith(".csv"):
            continue

        logger.info(f"Processing {file}")

        file_path = os.path.join(raw_folder, file)

        df = pd.read_csv(file_path)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove rows having all null values
        df = df.dropna(how="all")

        # Fill missing values
        df = df.ffill().bfill()

        # Convert Date column
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

            df = df.sort_values("Date")

        output_path = os.path.join(processed_folder, file)

        df.to_csv(output_path, index=False)

        logger.info(f"Saved cleaned {file}")

    logger.info("Price preprocessing completed.")
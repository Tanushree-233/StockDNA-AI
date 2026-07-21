import os
import pandas as pd

from scripts.utils.logger import logger


def preprocess_macro():

    raw_folder = "data/raw/macro"
    processed_folder = "data/processed/macro"

    os.makedirs(processed_folder, exist_ok=True)

    for file in os.listdir(raw_folder):

        if not file.endswith(".csv"):
            continue

        logger.info(f"Processing {file}")

        filepath = os.path.join(raw_folder, file)

        df = pd.read_csv(filepath)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove empty rows
        df = df.dropna(how="all")

        # Fill missing values
        df = df.ffill().bfill()

        # Clean column names
        df.columns = df.columns.str.strip()

        output = os.path.join(processed_folder, file)

        df.to_csv(output, index=False)

        logger.info(f"Saved cleaned {file}")

    logger.info("Macro preprocessing completed.")
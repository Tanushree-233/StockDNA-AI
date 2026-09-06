import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_earnings():

    config = load_config()

    raw_file = os.path.join(
        config["paths"]["raw"]["earnings"],
        "earnings.csv"
    )

    processed_folder = (
        config["paths"]["processed"]["earnings"]
    )

    os.makedirs(
        processed_folder,
        exist_ok=True
    )

    logger.info(
        "Processing earnings.csv"
    )

    df = pd.read_csv(raw_file)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove empty rows
    df = df.dropna(
        how="all"
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    output_file = os.path.join(
        processed_folder,
        "earnings.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        "Earnings preprocessing completed."
    )


if __name__ == "__main__":
    preprocess_earnings()
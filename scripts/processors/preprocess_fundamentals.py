import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_fundamentals():

    config = load_config()

    raw_file = os.path.join(
        config["paths"]["raw"]["fundamentals"],
        "fundamentals.csv"
    )

    processed_folder = (
        config["paths"]["processed"]["fundamentals"]
    )

    os.makedirs(
        processed_folder,
        exist_ok=True
    )

    logger.info(
        "Processing fundamentals.csv"
    )

    df = pd.read_csv(raw_file)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove completely empty rows
    df = df.dropna(
        how="all"
    )

    # Clean column names
    df.columns = df.columns.str.strip()

    output_file = os.path.join(
        processed_folder,
        "fundamentals.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    logger.info(
        "Fundamentals preprocessing completed."
    )


if __name__ == "__main__":
    preprocess_fundamentals()
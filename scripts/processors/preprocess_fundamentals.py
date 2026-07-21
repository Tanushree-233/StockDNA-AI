import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_fundamentals():

    config = load_config()

    raw_file = "data/raw/fundamentals/fundamentals.csv"
    processed_folder = "data/processed/fundamentals"

    os.makedirs(processed_folder, exist_ok=True)

    logger.info("Processing fundamentals.csv")

    df = pd.read_csv(raw_file)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove empty rows
    df = df.dropna(how="all")

    # Fill missing values
    df = df.ffill().bfill()

    # Standardize column names
    df.columns = df.columns.str.strip()

    output_file = os.path.join(processed_folder, "fundamentals.csv")

    df.to_csv(output_file, index=False)

    logger.info("Fundamentals preprocessing completed.")
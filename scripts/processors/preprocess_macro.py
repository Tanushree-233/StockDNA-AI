import os
import pandas as pd

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def preprocess_macro():

    config = load_config()

    raw_folder = config["paths"]["raw"]["macro"]
    processed_folder = config["paths"]["processed"]["macro"]

    os.makedirs(
        processed_folder,
        exist_ok=True
    )

    for file in os.listdir(raw_folder):

        if not file.endswith(".csv"):
            continue

        logger.info(f"Processing {file}")

        filepath = os.path.join(
            raw_folder,
            file
        )

        df = pd.read_csv(filepath)

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Remove empty rows
        df = df.dropna(how="all")

        # Clean column names
        df.columns = df.columns.str.strip()

        # Convert and normalize Date column
        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = df.dropna(
                subset=["Date"]
            )

            df = df.sort_values(
                "Date"
            )

        output = os.path.join(
            processed_folder,
            file
        )

        df.to_csv(
            output,
            index=False
        )

        logger.info(
            f"Saved cleaned {file}"
        )

    logger.info(
        "Macro preprocessing completed."
    )


if __name__ == "__main__":
    preprocess_macro()
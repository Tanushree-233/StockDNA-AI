import os
import pandas as pd

from scripts.utils.logger import logger


def merge_dataset():

    processed = "data/processed"

    output_folder = os.path.join(processed, "merged")
    os.makedirs(output_folder, exist_ok=True)

    merged = []

    # Read cleaned price files
    price_folder = os.path.join(processed, "prices")

    for file in os.listdir(price_folder):

        if not file.endswith(".csv"):
            continue

        ticker = file.replace(".csv", "")

        logger.info(f"Merging {ticker}")

        df = pd.read_csv(os.path.join(price_folder, file))

        df["Ticker"] = ticker

        merged.append(df)

    final_df = pd.concat(merged, ignore_index=True)

    output_file = os.path.join(output_folder, "master_dataset.csv")

    final_df.to_csv(output_file, index=False)

    logger.info("Master dataset created successfully.")
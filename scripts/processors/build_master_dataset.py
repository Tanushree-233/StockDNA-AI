import os
import pandas as pd

from scripts.utils.logger import logger


FEATURES_FOLDER = "data/processed/features"
OUTPUT_FILE = "data/final/master_dataset.csv"


def build_master_dataset():

    logger.info("Starting master dataset build.")

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    feature_files = [
        file
        for file in os.listdir(FEATURES_FOLDER)
        if file.endswith("_features.csv")
    ]

    if not feature_files:
        logger.error(
            "No feature files found."
        )
        raise FileNotFoundError(
            f"No feature files found in {FEATURES_FOLDER}"
        )

    datasets = []

    for file in sorted(feature_files):

        filepath = os.path.join(
            FEATURES_FOLDER,
            file
        )

        logger.info(
            f"Loading feature file: {file}"
        )

        df = pd.read_csv(filepath)

        if df.empty:
            logger.warning(
                f"Skipping empty file: {file}"
            )
            continue

        # Normalize column names
        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        # Add Ticker if it does not already exist
        if "Ticker" not in df.columns:

            ticker = file.replace(
                "_features.csv",
                ""
            )

            df["Ticker"] = ticker

        # Normalize ticker values
        df["Ticker"] = (
            df["Ticker"]
            .astype("string")
            .str.strip()
            .str.upper()
        )

        # Normalize Date
        if "Date" not in df.columns:

            logger.error(
                f"Date column missing in {file}"
            )

            raise ValueError(
                f"Date column missing in {file}"
            )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        invalid_dates = df["Date"].isna().sum()

        if invalid_dates > 0:

            logger.warning(
                f"{file}: removing "
                f"{invalid_dates} invalid dates."
            )

            df = df.dropna(
                subset=["Date"]
            )

        # Remove completely empty rows
        df = df.dropna(
            how="all"
        )

        datasets.append(df)

        logger.info(
            f"Loaded {len(df)} rows from {file}"
        )

    if not datasets:

        logger.error(
            "No valid feature datasets available."
        )

        raise ValueError(
            "No valid feature datasets available."
        )

    # Combine all feature datasets
    master = pd.concat(
        datasets,
        ignore_index=True
    )

    logger.info(
        f"Combined dataset rows: {len(master)}"
    )

    # Remove exact duplicate rows
    before_duplicates = len(master)

    master = master.drop_duplicates()

    removed_duplicates = (
        before_duplicates - len(master)
    )

    logger.info(
        f"Removed {removed_duplicates} "
        f"duplicate rows."
    )

    # Check Ticker + Date uniqueness
    duplicate_ticker_dates = master.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    if duplicate_ticker_dates > 0:

        logger.error(
            "Duplicate Ticker + Date combinations "
            f"found: {duplicate_ticker_dates}"
        )

        raise ValueError(
            "Duplicate Ticker + Date combinations "
            "found in master dataset."
        )

    logger.info(
        "Ticker + Date uniqueness check passed."
    )

    # Sort final dataset
    master = master.sort_values(
        ["Ticker", "Date"]
    ).reset_index(
        drop=True
    )

    # Required columns
    required_columns = [
        "Date",
        "Ticker",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in master.columns
    ]

    if missing_columns:

        logger.error(
            "Missing required columns: "
            f"{missing_columns}"
        )

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # Final dataset validation
    if master.empty:

        logger.error(
            "Final master dataset is empty."
        )

        raise ValueError(
            "Final master dataset is empty."
        )

    # Save final dataset
    master.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        f"Master dataset saved: {OUTPUT_FILE}"
    )

    logger.info(
        f"Final rows: {len(master)}"
    )

    logger.info(
        f"Final columns: {len(master.columns)}"
    )

    logger.info(
        "Master dataset build completed successfully."
    )


if __name__ == "__main__":
    build_master_dataset()
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

        filepath = os.path.join(
            save_folder,
            f"{name}.csv"
        )

        existing = None

        if os.path.exists(filepath):

            existing = pd.read_csv(filepath)

            if "Date" in existing.columns:

                existing["Date"] = pd.to_datetime(
                    existing["Date"],
                    errors="coerce"
                )

                last_date = existing["Date"].max()

                if pd.notna(last_date):
                    start = (
                        last_date +
                        pd.Timedelta(days=1)
                    ).strftime("%Y-%m-%d")
                else:
                    start = config["data"]["start_date"]

            else:
                start = config["data"]["start_date"]

        else:
            start = config["data"]["start_date"]

        end = (
            pd.Timestamp.today()
            + pd.Timedelta(days=1)
        ).strftime("%Y-%m-%d")

        logger.info(
            f"Updating {name}: {start} → {end}"
        )

        df = yf.download(
            symbol,
            start=start,
            end=end,
            progress=False
        )

        if df.empty:
            logger.info(
                f"No new data for {name}"
            )
            continue

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):
            df.columns = (
                df.columns
                .get_level_values(0)
            )

        df.reset_index(inplace=True)

        if existing is not None:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            df = pd.concat(
                [existing, df],
                ignore_index=True
            )

            df = df.drop_duplicates(
                subset=["Date"],
                keep="last"
            )

            df = df.sort_values(
                "Date"
            ).reset_index(drop=True)

        df.to_csv(
            filepath,
            index=False
        )

        logger.info(
            f"Saved {filepath}"
        )

    logger.info("Macro data downloaded successfully.")
import os
import pandas as pd
import yfinance as yf

from scripts.utils.config import load_config
from scripts.utils.logger import logger


def update_market():

    config = load_config()

    market_folder = config["paths"]["raw"]["market"]

    os.makedirs(
        market_folder,
        exist_ok=True
    )

    indices = {
        "nifty50": config["market"]["nifty"],
        "indiavix": config["market"]["vix"]
    }

    end = (
        pd.Timestamp.today()
        + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")

    for name, symbol in indices.items():

        filepath = os.path.join(
            market_folder,
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

        logger.info(
            f"Updating {name}: {start} -> {end}"
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


if __name__ == "__main__":
    update_market()
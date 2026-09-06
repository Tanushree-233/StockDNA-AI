import os
import pandas as pd
import yfinance as yf

from scripts.utils.config import load_config


TICKERS = [
    "TCS.NS",
    "INFY.NS",
    "RELIANCE.NS"
]


def download_earnings_events():

    config = load_config()

    output_folder = config["paths"]["raw"]["internal"]

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    output = os.path.join(
        output_folder,
        "earnings_events.csv"
    )

    all_events = []

    for ticker in TICKERS:

        print(
            f"\nCollecting earnings events for {ticker}..."
        )

        stock = yf.Ticker(ticker)

        try:

            earnings = stock.get_earnings_dates(
                limit=100
            )

            if earnings is None or earnings.empty:

                print(
                    f"No earnings data found for {ticker}"
                )

                continue

            earnings = earnings.reset_index()

            # Yahoo names the date column "Earnings Date"
            date_column = "Earnings Date"

            earnings["EarningsDate"] = pd.to_datetime(
                earnings[date_column],
                utc=True,
                errors="coerce"
            )

            earnings["EarningsDate"] = (
                earnings["EarningsDate"]
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )

            earnings["Date"] = (
                earnings["EarningsDate"]
                .dt.normalize()
            )

            earnings["Ticker"] = (
                ticker.replace(".NS", "")
            )

            # Keep only historical events
            earnings = earnings[
                earnings["Reported EPS"].notna()
            ].copy()

            today = pd.Timestamp.today().normalize()

            earnings = earnings[
                earnings["Date"] <= today
            ].copy()

            # EPS surprise
            earnings["EPS_Surprise"] = pd.to_numeric(
                earnings["Surprise(%)"],
                errors="coerce"
            )

            earnings["EPS_Estimate"] = pd.to_numeric(
                earnings["EPS Estimate"],
                errors="coerce"
            )

            earnings["Reported_EPS"] = pd.to_numeric(
                earnings["Reported EPS"],
                errors="coerce"
            )

            # Event indicators
            earnings["Earnings_Event"] = 1

            earnings["Positive_Earnings_Surprise"] = (
                earnings["EPS_Surprise"] > 0
            ).astype(int)

            earnings["Negative_Earnings_Surprise"] = (
                earnings["EPS_Surprise"] < 0
            ).astype(int)

            events = earnings[
                [
                    "Ticker",
                    "Date",
                    "EPS_Estimate",
                    "Reported_EPS",
                    "EPS_Surprise",
                    "Earnings_Event",
                    "Positive_Earnings_Surprise",
                    "Negative_Earnings_Surprise"
                ]
            ].copy()

            all_events.append(events)

            print(
                f"Events collected: {len(events)}"
            )

        except Exception as e:

            print(
                f"Error collecting {ticker}: {e}"
            )

    if not all_events:

        raise RuntimeError(
            "No earnings events collected."
        )

    final = pd.concat(
        all_events,
        ignore_index=True
    )

    final = final.drop_duplicates(
        subset=["Ticker", "Date"],
        keep="last"
    )

    final = final.sort_values(
        ["Ticker", "Date"]
    ).reset_index(drop=True)

    final.to_csv(
        output,
        index=False
    )

    print("\n" + "=" * 60)
    print("EARNINGS EVENT DATASET")
    print("=" * 60)

    print("Shape:", final.shape)

    print("\nEvents by company:")
    print(final["Ticker"].value_counts())

    print("\nDate range:")
    print(
        final["Date"].min(),
        "→",
        final["Date"].max()
    )

    print("\nEvents:")
    print(final.to_string(index=False))

    print("\nSaved:")
    print(output)


if __name__ == "__main__":
    download_earnings_events()
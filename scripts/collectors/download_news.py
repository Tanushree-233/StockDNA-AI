import os
import logging
from datetime import datetime

import pandas as pd
import yfinance as yf

from scripts.utils.config import load_config


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

config = load_config()

OUTPUT_FOLDER = os.path.join(
    config["paths"]["raw"]["internal"],
    "news"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_FOLDER,
    "news.csv"
)

TICKERS = {
    "TCS.NS": "TCS",
    "INFY.NS": "INFY",
    "RELIANCE.NS": "RELIANCE"
}


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def clean_text(value):
    """Convert a value to clean text."""
    if value is None:
        return ""

    return str(value).strip()


def normalize_news_item(item, ticker, company):
    """Convert a Yahoo Finance news item into a standard row."""

    content = item.get("content", {})

    title = clean_text(content.get("title"))

    if not title:
        return None

    provider = content.get("provider", {})
    source = clean_text(provider.get("displayName"))

    pub_date = content.get("pubDate")

    if pub_date:
        try:
            published_at = pd.to_datetime(pub_date, utc=True)
        except Exception:
            published_at = pd.NaT
    else:
        published_at = pd.NaT

    canonical_url = ""

    click_through = content.get("clickThroughUrl")

    if isinstance(click_through, dict):
        canonical_url = clean_text(
            click_through.get("url")
        )

    if not canonical_url:
        canonical_url = clean_text(
            content.get("canonicalUrl", {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else ""
        )

    return {
        "Ticker": ticker,
        "Company": company,
        "Published_At": published_at,
        "Date": (
            published_at.tz_convert("Asia/Kolkata")
            .tz_localize(None)
            .normalize()
            if pd.notna(published_at)
            else pd.NaT
        ),
        "Headline": title,
        "Source": source,
        "URL": canonical_url
    }


# ---------------------------------------------------------
# Main collector
# ---------------------------------------------------------

def collect_news():

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    rows = []

    logger.info("Starting news collection...")

    for yahoo_ticker, company in TICKERS.items():

        logger.info(
            "Collecting news for %s",
            company
        )

        try:
            ticker = yf.Ticker(yahoo_ticker)

            news_items = ticker.news or []

            logger.info(
                "%s news items found for %s",
                len(news_items),
                company
            )

            for item in news_items:

                row = normalize_news_item(
                    item,
                    yahoo_ticker.replace(".NS", ""),
                    company
                )

                if row is not None:
                    rows.append(row)

        except Exception as exc:

            logger.error(
                "Failed to collect news for %s: %s",
                company,
                exc
            )

    if not rows:
        logger.warning("No news data collected.")
        return

    df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Standardization
    # -----------------------------------------------------

    df["Ticker"] = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["Company"] = (
        df["Company"]
        .astype(str)
        .str.strip()
    )

    df["Headline"] = (
        df["Headline"]
        .astype(str)
        .str.strip()
    )

    df["Source"] = (
        df["Source"]
        .astype(str)
        .str.strip()
    )

    df["URL"] = (
        df["URL"]
        .astype(str)
        .str.strip()
    )

    df["Published_At"] = pd.to_datetime(
        df["Published_At"],
        errors="coerce"
    )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # Remove invalid rows
    df = df.dropna(
        subset=["Ticker", "Date", "Headline"]
    )

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Remove duplicate articles
    duplicate_columns = [
        "Ticker",
        "Headline",
        "URL"
    ]

    df = df.drop_duplicates(
        subset=duplicate_columns,
        keep="first"
    )

    # Sort newest first
    df = df.sort_values(
        ["Date", "Ticker"],
        ascending=[False, True]
    )

    # -----------------------------------------------------
    # Incremental merge with existing news
    # -----------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        try:

            existing = pd.read_csv(
                OUTPUT_FILE
            )

            if not existing.empty:

                existing["Published_At"] = pd.to_datetime(
                    existing["Published_At"],
                    errors="coerce"
                )

                existing["Date"] = pd.to_datetime(
                    existing["Date"],
                    errors="coerce"
                )

                combined = pd.concat(
                    [existing, df],
                    ignore_index=True
                )

                combined = combined.drop_duplicates(
                    subset=duplicate_columns,
                    keep="first"
                )

                combined = combined.sort_values(
                    ["Date", "Ticker"],
                    ascending=[False, True]
                )

                df = combined

        except Exception as exc:

            logger.warning(
                "Could not merge existing news file: %s",
                exc
            )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    logger.info(
        "News dataset saved: %s",
        OUTPUT_FILE
    )

    logger.info(
        "Total news records: %d",
        len(df)
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    collect_news()
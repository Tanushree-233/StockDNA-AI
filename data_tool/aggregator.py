from typing import List, Optional
import pandas as pd
import numpy as np


def aggregate_data(*datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates multiple datasets of the same modality along rows.
    """
    valid_datasets = [
        df for df in datasets
        if df is not None and not df.empty
    ]

    if not valid_datasets:
        raise ValueError(
            "No valid datasets were provided for aggregation."
        )

    combined = pd.concat(
        valid_datasets,
        ignore_index=True,
        sort=False
    )

    return combined.reset_index(drop=True)


def aggregate_point_in_time(
    market_df: pd.DataFrame,
    earnings_df: Optional[pd.DataFrame] = None,
    macro_df: Optional[pd.DataFrame] = None,
    news_sentiment_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Aggregates multi-source feeds into a single unified point-in-time historical table.
    Enforces backward-only asof merging for events to prevent lookahead bias.
    """
    if market_df is None or market_df.empty:
        raise ValueError("Base market dataframe cannot be empty.")

    base = market_df.copy()
    if "Date" in base.columns:
        base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
        base = base.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # 1. Merge Macro Data (Date-exact match)
    if macro_df is not None and not macro_df.empty:
        macro = macro_df.copy()
        if "Date" in macro.columns:
            macro["Date"] = pd.to_datetime(macro["Date"], errors="coerce")
            macro = macro.dropna(subset=["Date"]).sort_values("Date").drop_duplicates(subset=["Date"])
            base = pd.merge(base, macro, on="Date", how="left")

    # 2. Merge Earnings Events (Point-in-Time backward match using Announcement Date)
    if earnings_df is not None and not earnings_df.empty:
        events = earnings_df.copy()
        date_col = "Date"
        if "AnnouncementDate" in events.columns:
            events["AnnouncementDate"] = pd.to_datetime(events["AnnouncementDate"], errors="coerce")
            date_col = "AnnouncementDate"
        elif "Date" in events.columns:
            events["Date"] = pd.to_datetime(events["Date"], errors="coerce")

        events = events.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        if "Ticker" in base.columns and "Ticker" in events.columns:
            # Per-ticker point-in-time merge
            merged_tickers = []
            for ticker in base["Ticker"].unique():
                b_sub = base[base["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
                e_sub = events[events["Ticker"] == ticker].sort_values(date_col).reset_index(drop=True)

                if e_sub.empty:
                    merged_tickers.append(b_sub)
                    continue

                m = pd.merge_asof(
                    b_sub,
                    e_sub.drop(columns=["Ticker"], errors="ignore"),
                    left_on="Date",
                    right_on=date_col,
                    direction="backward",
                    suffixes=("", "_earnings")
                )
                merged_tickers.append(m)
            base = pd.concat(merged_tickers, ignore_index=True)
        else:
            base = pd.merge_asof(
                base,
                events,
                left_on="Date",
                right_on=date_col,
                direction="backward"
            )

    # 3. Merge News Sentiment (Date-aligned rolling aggregation)
    if news_sentiment_df is not None and not news_sentiment_df.empty:
        news = news_sentiment_df.copy()
        if "Date" in news.columns:
            news["Date"] = pd.to_datetime(news["Date"], errors="coerce")
            # If multiple news per date, aggregate sentiment score
            if "Sentiment_Score" in news.columns:
                agg_news = news.groupby("Date", as_index=False)["Sentiment_Score"].mean()
                agg_news.rename(columns={"Sentiment_Score": "News_Sentiment_Score"}, inplace=True)
                base = pd.merge(base, agg_news, on="Date", how="left")
                base["News_Sentiment_Score"] = base["News_Sentiment_Score"].fillna(0.0)

    return base.sort_values(["Ticker", "Date"] if "Ticker" in base.columns else "Date").reset_index(drop=True)
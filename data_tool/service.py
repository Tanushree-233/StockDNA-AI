from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid
import pandas as pd
import numpy as np

from data_tool.schemas import (
    DataCategory,
    DataType,
    ProvenanceMetadata,
    StructuredRecord,
    ProcessedDataBundle,
)
from data_tool.cleaner import (
    clean_data,
    clean_market_data,
    clean_earnings_data,
    clean_news_data,
)
from data_tool.deduplicator import remove_duplicates
from data_tool.feature_engineer import generate_features
from data_tool.classifier import (
    infer_data_type,
    classify_factor_category,
    classify_market_regime,
    classify_earnings_surprise,
    classify_text_sentiment,
    categorize_features,
)
from data_tool.aggregator import aggregate_point_in_time


class DataToolService:
    """
    Independent, decoupled Data Ingestion, Cleaning, Normalization,
    Deduplication, Classification, and Structuring Service.

    Guarantees strict point-in-time provenance metadata and emits a standardized
    contract (ProcessedDataBundle) for consumption by any downstream ML pipeline or API.
    """

    def __init__(self, default_source: str = "data_tool_service"):
        self.default_source = default_source

    def process(
        self,
        data: Union[pd.DataFrame, List[Dict[str, Any]], Dict[str, Any]],
        ticker: Optional[str] = None,
        data_type: Optional[DataType] = None,
        source: Optional[str] = None,
        available_at: Optional[str] = None,
        generate_technical_features: bool = True,
    ) -> ProcessedDataBundle:
        """
        Main polymorphic entry point to process any raw data into a structured bundle.
        """
        src = source or self.default_source
        avail = available_at or datetime.now(timezone.utc).isoformat()

        # Convert input to DataFrame
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data format: {type(data)}")

        if df.empty:
            return ProcessedDataBundle(
                ticker=ticker,
                records_count=0,
                data_types_included=[],
                records=[]
            )

        # 1. Infer Data Type if not explicitly passed
        inferred_type = data_type or infer_data_type(df)

        # 2. Extract or Normalize Ticker
        resolved_ticker = ticker
        if not resolved_ticker and "Ticker" in df.columns:
            tickers = df["Ticker"].dropna().unique()
            if len(tickers) == 1:
                resolved_ticker = str(tickers[0])
            elif len(tickers) > 1:
                resolved_ticker = "MULTI"
        resolved_ticker = resolved_ticker or "UNKNOWN"

        # 3. Clean and Deduplicate based on Data Type
        if inferred_type == DataType.MARKET_PRICE:
            cleaned = clean_market_data(df)
            cleaned = remove_duplicates(cleaned, subset=["Date"] if "Date" in cleaned.columns else None)
            if generate_technical_features and len(cleaned) >= 5:
                cleaned = generate_features(cleaned)
                cleaned = classify_market_regime(cleaned)
            data_cat = DataCategory.EXTERNAL

        elif inferred_type == DataType.EARNINGS_EVENT:
            cleaned = clean_earnings_data(df)
            cleaned = remove_duplicates(cleaned, subset=["Date"] if "Date" in cleaned.columns else None)
            cleaned = classify_earnings_surprise(cleaned)
            data_cat = DataCategory.INTERNAL

        elif inferred_type == DataType.NEWS_SENTIMENT:
            cleaned = clean_news_data(df)
            cleaned = remove_duplicates(cleaned, subset=["Headline"] if "Headline" in cleaned.columns else None)
            # Apply automated text sentiment
            if "Headline" in cleaned.columns:
                sentiments = cleaned["Headline"].apply(classify_text_sentiment)
                cleaned["Sentiment"] = sentiments.apply(lambda x: x[0].value)
                cleaned["Sentiment_Score"] = sentiments.apply(lambda x: x[1])
            data_cat = DataCategory.INTERNAL

        elif inferred_type == DataType.FUNDAMENTAL_STATEMENT:
            cleaned = clean_data(df)
            cleaned = remove_duplicates(cleaned, subset=["FinancialDate"] if "FinancialDate" in cleaned.columns else None)
            data_cat = DataCategory.INTERNAL

        else:
            cleaned = clean_data(df)
            cleaned = remove_duplicates(cleaned)
            data_cat = DataCategory.EXTERNAL

        # 4. Construct Structured Records with Provenance
        records: List[StructuredRecord] = []
        for idx, row in cleaned.iterrows():
            row_dict = row.to_dict()
            ts = str(row_dict.get("Date", row_dict.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d"))))

            # Exclude metadata and non-feature columns from feature payload
            features_dict = {
                k: v for k, v in row_dict.items()
                if k not in ["Date", "Ticker", "Symbol", "data_source", "available_at"]
                and pd.notna(v)
            }

            rec_ticker = str(row_dict.get("Ticker", resolved_ticker))

            provenance = ProvenanceMetadata(
                source=src,
                ingested_at=datetime.now(timezone.utc).isoformat(),
                available_at=avail,
                data_category=data_cat,
                data_type=inferred_type,
            )

            records.append(
                StructuredRecord(
                    ticker=rec_ticker,
                    timestamp=ts,
                    provenance=provenance,
                    features=features_dict,
                )
            )

        return ProcessedDataBundle(
            bundle_id=str(uuid.uuid4()),
            ticker=resolved_ticker,
            records_count=len(records),
            data_types_included=[inferred_type],
            records=records,
        )

    def process_market_data(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None,
        source: str = "market_feed",
    ) -> ProcessedDataBundle:
        """Dedicated processor for OHLCV stock/index time series."""
        return self.process(
            data=df,
            ticker=ticker,
            data_type=DataType.MARKET_PRICE,
            source=source,
            generate_technical_features=True,
        )

    def process_earnings(
        self,
        df: pd.DataFrame,
        ticker: Optional[str] = None,
        source: str = "earnings_calendar",
    ) -> ProcessedDataBundle:
        """Dedicated processor for quarterly earnings announcement dates and EPS surprises."""
        return self.process(
            data=df,
            ticker=ticker,
            data_type=DataType.EARNINGS_EVENT,
            source=source,
        )

    def process_news(
        self,
        news_items: Union[pd.DataFrame, List[Dict[str, Any]]],
        ticker: Optional[str] = None,
        source: str = "news_feed",
    ) -> ProcessedDataBundle:
        """Dedicated processor for financial news headlines and sentiment extraction."""
        return self.process(
            data=news_items,
            ticker=ticker,
            data_type=DataType.NEWS_SENTIMENT,
            source=source,
        )

    def aggregate_bundles(
        self,
        market_bundle: ProcessedDataBundle,
        earnings_bundle: Optional[ProcessedDataBundle] = None,
        macro_bundle: Optional[ProcessedDataBundle] = None,
        news_bundle: Optional[ProcessedDataBundle] = None,
    ) -> ProcessedDataBundle:
        """
        Combines disparate structured data bundles into a point-in-time aligned
        canonical bundle using strict backward asof joins.
        """
        market_df = market_bundle.to_dataframe()
        earnings_df = earnings_bundle.to_dataframe() if earnings_bundle else None
        macro_df = macro_bundle.to_dataframe() if macro_bundle else None
        news_df = news_bundle.to_dataframe() if news_bundle else None

        aligned_df = aggregate_point_in_time(
            market_df=market_df,
            earnings_df=earnings_df,
            macro_df=macro_df,
            news_sentiment_df=news_df,
        )

        all_types = list(set(
            market_bundle.data_types_included
            + (earnings_bundle.data_types_included if earnings_bundle else [])
            + (macro_bundle.data_types_included if macro_bundle else [])
            + (news_bundle.data_types_included if news_bundle else [])
        ))

        # Re-package as bundle
        return self.process(
            data=aligned_df,
            ticker=market_bundle.ticker,
            data_type=DataType.GENERIC,
            source="aggregated_multi_source",
            generate_technical_features=False,
        )

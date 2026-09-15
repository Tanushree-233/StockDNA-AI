from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import pandas as pd
from pydantic import BaseModel, Field


class DataCategory(str, Enum):
    """Broad classification of financial data origin for XAI & modeling."""
    INTERNAL = "INTERNAL"  # Company-specific: earnings, balance sheet, company news
    EXTERNAL = "EXTERNAL"  # Macro, broad market indices (NIFTY), VIX, general market regime


class DataType(str, Enum):
    """Specific modality of financial data."""
    MARKET_PRICE = "MARKET_PRICE"
    MARKET_INDEX = "MARKET_INDEX"
    EARNINGS_EVENT = "EARNINGS_EVENT"
    FUNDAMENTAL_STATEMENT = "FUNDAMENTAL_STATEMENT"
    NEWS_SENTIMENT = "NEWS_SENTIMENT"
    COMPANY_METADATA = "COMPANY_METADATA"
    MACRO_INDICATOR = "MACRO_INDICATOR"
    GENERIC = "GENERIC"


class MarketRegime(str, Enum):
    """Categorization of market price dynamics."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    NORMAL_VOLATILITY = "NORMAL_VOLATILITY"


class EarningsSurpriseCategory(str, Enum):
    """Classification of quarterly earnings surprise."""
    BEAT = "BEAT"
    MISS = "MISS"
    INLINE = "INLINE"
    UNKNOWN = "UNKNOWN"


class SentimentCategory(str, Enum):
    """Financial sentiment classification."""
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class ProvenanceMetadata(BaseModel):
    """
    Data provenance metadata tracking source, collection time,
    and critical point-in-time information availability.
    """
    source: str = Field(description="Origin data source (e.g. yfinance, rss, csv)")
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    available_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Point-in-time timestamp when data was publicly available (prevents lookahead bias)"
    )
    data_category: DataCategory = Field(default=DataCategory.EXTERNAL)
    data_type: DataType = Field(default=DataType.GENERIC)
    version: str = Field(default="1.0.0")
    quality_flags: List[str] = Field(default_factory=list)
    validation_status: str = Field(default="VALID")


class StructuredRecord(BaseModel):
    """
    Canonical standardized output record for a single observation.
    Guarantees consistent structure across disparate data feeds.
    """
    ticker: str
    timestamp: str  # YYYY-MM-DD or ISO datetime string
    provenance: ProvenanceMetadata
    features: Dict[str, Any] = Field(default_factory=dict)
    raw_payload: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True


class ProcessedDataBundle(BaseModel):
    """
    Collection of structured records produced by the Data Tool.
    Exposes conversion helpers for downstream analytical and ML pipelines.
    """
    bundle_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ticker: Optional[str] = None
    records_count: int = 0
    data_types_included: List[DataType] = Field(default_factory=list)
    records: List[StructuredRecord] = Field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten records into a tabular pandas DataFrame suitable for feature engineering
        or model consumption, preserving provenance attributes as prefixed columns.
        """
        if not self.records:
            return pd.DataFrame()

        rows = []
        for r in self.records:
            row = {
                "Ticker": r.ticker,
                "Date": r.timestamp,
                "data_source": r.provenance.source,
                "data_category": r.provenance.data_category.value,
                "data_type": r.provenance.data_type.value,
                "available_at": r.provenance.available_at,
            }
            row.update(r.features)
            rows.append(row)

        df = pd.DataFrame(rows)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.sort_values("Date").reset_index(drop=True)
        return df

    def get_features_matrix(self) -> pd.DataFrame:
        """Extract purely numerical and categorical feature columns."""
        df = self.to_dataframe()
        metadata_cols = [
            "Ticker", "Date", "data_source", "data_category",
            "data_type", "available_at"
        ]
        return df.drop(columns=[c for c in metadata_cols if c in df.columns], errors="ignore")

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Summarize data sources, date ranges, and categories in this bundle."""
        return {
            "bundle_id": self.bundle_id,
            "records_count": self.records_count,
            "ticker": self.ticker,
            "data_types": [dt.value for dt in self.data_types_included],
            "sources": list({r.provenance.source for r in self.records}),
            "categories": list({r.provenance.data_category.value for r in self.records}),
            "earliest_date": min([r.timestamp for r in self.records]) if self.records else None,
            "latest_date": max([r.timestamp for r in self.records]) if self.records else None,
        }

    def filter_by_category(self, category: DataCategory) -> List[StructuredRecord]:
        """Filter records by INTERNAL (company) or EXTERNAL (market/macro) category."""
        return [r for r in self.records if r.provenance.data_category == category]

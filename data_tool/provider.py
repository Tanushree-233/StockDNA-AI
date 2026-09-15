from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd

from data_tool.service import DataToolService
from data_tool.schemas import ProcessedDataBundle, DataCategory
from data_tool.classifier import categorize_features
from config.settings import PROJECT_ROOT


class BaseDataProvider(ABC):
    """
    Abstract interface for downstream consumers (Prediction Pipeline, Backtester, API).
    Decouples ML models from data collection, cleaning, and storage internals.
    """

    @abstractmethod
    def get_market_data(self, ticker: str) -> pd.DataFrame:
        """
        Fetch cleaned market OHLCV data for the given ticker.
        """
        pass

    @abstractmethod
    def get_live_features(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch the most recent point-in-time feature row for real-time inference.
        Returns clean dictionary of scale-invariant features.
        """
        pass

    @abstractmethod
    def get_structured_bundle(self, ticker: str) -> ProcessedDataBundle:
        """
        Fetch full structured bundle with complete provenance metadata.
        """
        pass

    @abstractmethod
    def get_factor_decomposition(self, ticker: str) -> Dict[str, Dict[str, Any]]:
        """
        Decomposes features into 'internal' (company-specific)
        and 'external' (macro/market) factor sets for the XAI Engine.
        """
        pass


class DataToolProvider(BaseDataProvider):
    """
    Concrete provider implementation wrapping DataToolService.
    Serves as the bridge between Data Tool and the Prediction/XAI engines.
    """

    def __init__(self, data_service: Optional[DataToolService] = None):
        self.service = data_service or DataToolService(default_source="stockdna_provider")

    def get_market_data(self, ticker: str) -> pd.DataFrame:
        """
        Retrieves clean market OHLCV data for ticker from local verified storage,
        with automated yfinance fallback.
        """
        clean_ticker = ticker.replace(".NS", "").strip().upper()

        # Check local storage first
        local_raw = PROJECT_ROOT / "data" / "raw" / "prices" / f"{clean_ticker}.csv"
        local_proc = PROJECT_ROOT / "data" / "processed" / "prices" / f"{clean_ticker}.csv"
        target_path = local_raw if local_raw.exists() else local_proc

        if target_path.exists():
            df = pd.read_csv(target_path)
            if "Price" in df.columns and "Date" not in df.columns:
                df = df.rename(columns={"Price": "Date"})
            df["Ticker"] = clean_ticker
            df["Date"] = pd.to_datetime(df["Date"])
            return df.sort_values("Date").reset_index(drop=True)

        # Fallback to yfinance if not available locally
        try:
            import yfinance as yf
            symbol = f"{clean_ticker}.NS" if not ticker.endswith(".NS") else ticker
            df = yf.download(symbol, period="1y", auto_adjust=False, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.reset_index()
                df["Ticker"] = clean_ticker
                df["Date"] = pd.to_datetime(df["Date"])
                return df.sort_values("Date").reset_index(drop=True)
        except Exception as e:
            raise ValueError(f"Failed to fetch market data for ticker '{clean_ticker}': {e}")

        raise FileNotFoundError(f"No price history found for ticker '{clean_ticker}' at {target_path}")

    def get_structured_bundle(self, ticker: str) -> ProcessedDataBundle:
        """
        Fetches and processes data for ticker via DataToolService.
        """
        clean_ticker = ticker.replace(".NS", "").strip().upper()
        df = self.get_market_data(clean_ticker)
        return self.service.process_market_data(df, ticker=clean_ticker)

    def get_live_features(self, ticker: str) -> Dict[str, Any]:
        """
        Retrieves latest available clean feature row for a given equity.
        """
        bundle = self.get_structured_bundle(ticker)
        if not bundle.records:
            return {}
        # Return features from the most recent observation
        latest_record = bundle.records[-1]
        return latest_record.features

    def get_factor_decomposition(self, ticker: str) -> Dict[str, Dict[str, Any]]:
        """
        Returns factor dictionary split into internal and external factor groups.
        """
        features = self.get_live_features(ticker)
        return categorize_features(features)

import os
import sys
import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tool.schemas import (
    DataCategory,
    DataType,
    MarketRegime,
    EarningsSurpriseCategory,
    SentimentCategory,
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
from data_tool.deduplicator import remove_duplicates, deduplicate_with_stats
from data_tool.classifier import (
    infer_data_type,
    classify_factor_category,
    categorize_features,
    classify_market_regime,
    classify_earnings_surprise,
    classify_text_sentiment,
)
from data_tool.feature_engineer import compute_scale_invariant_features, generate_features
from data_tool.service import DataToolService
from data_tool.pipeline import run_data_tool
from data_tool.provider import DataToolProvider, BaseDataProvider


class TestDataToolIngestionAndCleaning(unittest.TestCase):
    """Test raw data ingestion, column normalization, and type conversion."""

    def test_clean_market_data_basic(self):
        raw_data = pd.DataFrame([
            {" date ": "2024-01-01", " open ": "3,500.50", "high": 3550.0, "low": 3480.0, "close": 3520.0, "volume": "1,200,000", "ticker": "tcs"},
            {" date ": "2024-01-02", " open ": 3520.0, "high": 3570.0, "low": 3510.0, "close": 3560.0, "volume": 1400000, "ticker": "tcs"},
        ])
        cleaned = clean_market_data(raw_data)
        self.assertEqual(len(cleaned), 2)
        self.assertIn("Date", cleaned.columns)
        self.assertIn("Open", cleaned.columns)
        self.assertIn("Close", cleaned.columns)
        self.assertEqual(cleaned["Open"].iloc[0], 3500.50)
        self.assertEqual(cleaned["Volume"].iloc[0], 1200000)

    def test_clean_market_data_drops_invalid_prices(self):
        invalid_data = pd.DataFrame([
            {"Date": "2024-01-01", "Open": -100.0, "High": 3500.0, "Low": 3400.0, "Close": 3450.0, "Volume": 1000},
            {"Date": "2024-01-02", "Open": 3500.0, "High": 3600.0, "Low": 3480.0, "Close": 3550.0, "Volume": 1000},
        ])
        cleaned = clean_market_data(invalid_data)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["Close"].iloc[0], 3550.0)

    def test_clean_news_data(self):
        raw_news = [
            {"title": "  TCS Reports Record Profit in Q3  ", "pubDate": "2024-01-10T12:00:00Z", "source": "Reuters"},
            {"title": "", "pubDate": "2024-01-11", "source": "Bloomberg"},  # empty headline should be dropped
        ]
        cleaned = clean_news_data(raw_news)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned["Headline"].iloc[0], "TCS Reports Record Profit in Q3")
        self.assertIn("Published_At", cleaned.columns)


class TestDataToolDeduplication(unittest.TestCase):
    """Test duplicate record detection and removal."""

    def test_deduplicate_by_subset(self):
        df = pd.DataFrame([
            {"Ticker": "TCS", "Date": "2024-01-01", "Close": 3500.0},
            {"Ticker": "TCS", "Date": "2024-01-01", "Close": 3510.0},  # Updated close
            {"Ticker": "TCS", "Date": "2024-01-02", "Close": 3550.0},
        ])
        deduped, removed = deduplicate_with_stats(df, subset=["Ticker", "Date"], keep="last")
        self.assertEqual(len(deduped), 2)
        self.assertEqual(removed, 1)
        # Should keep the last record
        self.assertEqual(deduped[deduped["Date"] == "2024-01-01"]["Close"].iloc[0], 3510.0)


class TestDataToolClassification(unittest.TestCase):
    """Test automated categorization, data type inference, regimes, and sentiment."""

    def test_infer_data_type(self):
        prices_df = pd.DataFrame([{"Open": 100, "High": 105, "Low": 98, "Close": 102}])
        earnings_df = pd.DataFrame([{"Date": "2024-01-01", "Reported_EPS": 10.5, "EPS_Estimate": 10.0}])
        news_df = pd.DataFrame([{"Headline": "Infosys secures deal", "Source": "PTI"}])
        fundamentals_df = pd.DataFrame([{"TotalAssets": 50000, "NetIncome": 8000}])

        self.assertEqual(infer_data_type(prices_df), DataType.MARKET_PRICE)
        self.assertEqual(infer_data_type(earnings_df), DataType.EARNINGS_EVENT)
        self.assertEqual(infer_data_type(news_df), DataType.NEWS_SENTIMENT)
        self.assertEqual(infer_data_type(fundamentals_df), DataType.FUNDAMENTAL_STATEMENT)

    def test_factor_taxonomy_classification(self):
        # Company-specific internal factors
        self.assertEqual(classify_factor_category("NetIncome"), DataCategory.INTERNAL)
        self.assertEqual(classify_factor_category("OperatingMargin"), DataCategory.INTERNAL)
        self.assertEqual(classify_factor_category("EPS_Surprise_Pct"), DataCategory.INTERNAL)
        self.assertEqual(classify_factor_category("Days_Since_Earnings"), DataCategory.INTERNAL)

        # Market/Macro external factors
        self.assertEqual(classify_factor_category("NIFTY_Return"), DataCategory.EXTERNAL)
        self.assertEqual(classify_factor_category("VIX_Close"), DataCategory.EXTERNAL)
        self.assertEqual(classify_factor_category("Close_to_SMA20"), DataCategory.EXTERNAL)
        self.assertEqual(classify_factor_category("RSI_14"), DataCategory.EXTERNAL)
        self.assertEqual(classify_factor_category("Rolling_Volatility_20d"), DataCategory.EXTERNAL)

    def test_categorize_features_split(self):
        features = {
            "OperatingMargin": 0.25,
            "DebtToEquity": 0.10,
            "Close_to_SMA20": 0.03,
            "RSI_14": 55.0,
            "VIX_Close": 14.5,
        }
        split = categorize_features(features)
        self.assertIn("OperatingMargin", split["internal"])
        self.assertIn("DebtToEquity", split["internal"])
        self.assertIn("Close_to_SMA20", split["external"])
        self.assertIn("RSI_14", split["external"])
        self.assertIn("VIX_Close", split["external"])

    def test_classify_earnings_surprise(self):
        events = pd.DataFrame([
            {"Date": "2024-01-01", "EPS_Surprise_Pct": 5.0},
            {"Date": "2024-04-01", "EPS_Surprise_Pct": -4.5},
            {"Date": "2024-07-01", "EPS_Surprise_Pct": 0.2},
        ])
        classified = classify_earnings_surprise(events)
        self.assertEqual(classified["Earnings_Surprise_Category"].tolist(), ["BEAT", "MISS", "INLINE"])

    def test_classify_market_regime(self):
        # 10 days of steady upward price action
        prices = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=10),
            "Close": [100.0 + i * 5.0 for i in range(10)]
        })
        regimes = classify_market_regime(prices)
        self.assertIn("Trend_Regime", regimes.columns)
        self.assertIn("Volatility_Regime", regimes.columns)

    def test_classify_text_sentiment(self):
        cat_pos, score_pos = classify_text_sentiment("Company beats quarterly estimates with record profit growth")
        self.assertEqual(cat_pos, SentimentCategory.POSITIVE)
        self.assertGreater(score_pos, 0.0)

        cat_neg, score_neg = classify_text_sentiment("Shares slump as quarterly revenue miss and loss widens")
        self.assertEqual(cat_neg, SentimentCategory.NEGATIVE)
        self.assertLess(score_neg, 0.0)

        cat_neu, score_neu = classify_text_sentiment("Board announces date for annual general meeting")
        self.assertEqual(cat_neu, SentimentCategory.NEUTRAL)
        self.assertEqual(score_neu, 0.0)


class TestDataToolScaleInvariantFeatures(unittest.TestCase):
    """Test that generated features are scale-invariant and contain zero raw nominal prices."""

    def test_scale_invariant_feature_generation(self):
        dates = pd.date_range("2024-01-01", periods=30)
        np.random.seed(42)
        base_price = 3000.0
        returns = np.random.normal(0.001, 0.015, 30)
        prices = [base_price]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        df = pd.DataFrame({
            "Date": dates,
            "Open": [p * 0.995 for p in prices],
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [1000000 + i * 10000 for i in range(30)],
        })

        feat_df = generate_features(df)

        expected_features = [
            "Daily_Return", "Return_5d", "Close_to_SMA20",
            "Normalized_MACD", "Bollinger_PctB", "Normalized_ATR",
            "High_Low_Spread_Pct", "Rolling_Volatility_20d"
        ]
        for feat in expected_features:
            self.assertIn(feat, feat_df.columns)

        # Scale-invariant values should be bounded reasonable ratios
        self.assertTrue((feat_df["Normalized_ATR"].dropna() < 0.10).all())
        self.assertTrue((feat_df["High_Low_Spread_Pct"].dropna() < 0.10).all())


class TestDataToolServiceAndProvenance(unittest.TestCase):
    """Test DataToolService execution, provenance metadata, and structured output bundle."""

    def setUp(self):
        self.service = DataToolService(default_source="unit_test_suite")

    def test_process_market_bundle_provenance(self):
        dates = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "Date": dates,
            "Open": [100.0 + i for i in range(10)],
            "High": [102.0 + i for i in range(10)],
            "Low": [99.0 + i for i in range(10)],
            "Close": [101.0 + i for i in range(10)],
            "Volume": [10000] * 10,
        })

        available_at_time = "2024-01-11T09:00:00Z"
        bundle = self.service.process(
            data=df,
            ticker="TEST_TICKER",
            source="mock_feed",
            available_at=available_at_time
        )

        self.assertIsInstance(bundle, ProcessedDataBundle)
        self.assertEqual(bundle.ticker, "TEST_TICKER")
        self.assertEqual(bundle.records_count, 10)

        # Check provenance on a record
        rec = bundle.records[0]
        self.assertEqual(rec.provenance.source, "mock_feed")
        self.assertEqual(rec.provenance.available_at, available_at_time)
        self.assertEqual(rec.provenance.data_category, DataCategory.EXTERNAL)

        # Test bundle conversion helpers
        bundle_df = bundle.to_dataframe()
        self.assertEqual(len(bundle_df), 10)
        self.assertIn("data_category", bundle_df.columns)
        self.assertIn("available_at", bundle_df.columns)

        summary = bundle.get_provenance_summary()
        self.assertEqual(summary["ticker"], "TEST_TICKER")
        self.assertIn("mock_feed", summary["sources"])

    def test_pipeline_entrypoint(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=6),
            "Open": [50.0] * 6,
            "High": [55.0] * 6,
            "Low": [48.0] * 6,
            "Close": [52.0] * 6,
            "Volume": [5000] * 6,
        })
        # Test return as bundle
        bundle = run_data_tool(df, ticker="INFY", return_bundle=True)
        self.assertIsInstance(bundle, ProcessedDataBundle)
        self.assertEqual(bundle.ticker, "INFY")

        # Test return as DataFrame
        out_df = run_data_tool(df, ticker="INFY", return_bundle=False)
        self.assertIsInstance(out_df, pd.DataFrame)
        self.assertFalse(out_df.empty)

    def test_decoupled_data_provider_contract(self):
        """Verify the BaseDataProvider abstraction functions without ML coupling."""
        provider = DataToolProvider(data_service=self.service)
        self.assertIsInstance(provider, BaseDataProvider)


class TestDataToolEdgeCases(unittest.TestCase):
    """Test malformed inputs, missing values, and empty structures."""

    def setUp(self):
        self.service = DataToolService(default_source="edge_cases")

    def test_empty_dataframe(self):
        bundle = self.service.process(pd.DataFrame())
        self.assertEqual(bundle.records_count, 0)
        self.assertEqual(len(bundle.records), 0)

    def test_all_nan_series(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [np.nan, np.nan],
            "Open": [np.nan, np.nan],
        })
        cleaned = clean_market_data(df)
        self.assertEqual(len(cleaned), 0)


if __name__ == "__main__":
    unittest.main()

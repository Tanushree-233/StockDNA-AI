import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tool.service import DataToolService
from data_tool.pipeline import run_data_tool
from data_tool.schemas import DataType, DataCategory


def main():
    print("\n========================================")
    print("STOCKDNA DATA TOOL INDEPENDENT TEST")
    print("========================================")

    service = DataToolService(default_source="test_runner")

    # 1. Test with synthetic multi-source market data
    sample_prices = pd.DataFrame([
        {"Date": "2024-01-01", "Open": 3500.0, "High": 3550.0, "Low": 3480.0, "Close": 3520.0, "Volume": 1200000, "Ticker": "TCS"},
        {"Date": "2024-01-02", "Open": 3520.0, "High": 3570.0, "Low": 3510.0, "Close": 3560.0, "Volume": 1400000, "Ticker": "TCS"},
        {"Date": "2024-01-03", "Open": 3560.0, "High": 3580.0, "Low": 3530.0, "Close": 3540.0, "Volume": 1100000, "Ticker": "TCS"},
        {"Date": "2024-01-04", "Open": 3540.0, "High": 3600.0, "Low": 3535.0, "Close": 3590.0, "Volume": 1600000, "Ticker": "TCS"},
        {"Date": "2024-01-05", "Open": 3590.0, "High": 3620.0, "Low": 3570.0, "Close": 3610.0, "Volume": 1500000, "Ticker": "TCS"},
        # Duplicate row to test deduplication
        {"Date": "2024-01-05", "Open": 3590.0, "High": 3620.0, "Low": 3570.0, "Close": 3610.0, "Volume": 1500000, "Ticker": "TCS"},
    ])

    print(f"\n1. Ingesting raw market data ({len(sample_prices)} rows with 1 duplicate)...")
    market_bundle = service.process_market_data(sample_prices, ticker="TCS")
    print(f"   Processed records: {market_bundle.records_count} (Duplicates removed: {len(sample_prices) - market_bundle.records_count})")
    print(f"   Provenance: {market_bundle.get_provenance_summary()}")

    # 2. Test with sample earnings event
    sample_earnings = pd.DataFrame([
        {
            "Date": "2024-01-04",
            "Ticker": "TCS",
            "Reported_EPS": 32.5,
            "EPS_Estimate": 30.0,
            "EPS_Surprise_Pct": 8.33,
        }
    ])
    print(f"\n2. Ingesting earnings event...")
    earnings_bundle = service.process_earnings(sample_earnings, ticker="TCS")
    print(f"   Earnings records: {earnings_bundle.records_count}")
    print(f"   Category: {earnings_bundle.records[0].provenance.data_category.value}")
    print(f"   Surprise Classification: {earnings_bundle.records[0].features.get('Earnings_Surprise_Category')}")

    # 3. Test with sample news
    sample_news = [
        {"title": "TCS beats quarterly profit estimates on major UK deal", "published": "2024-01-04"},
        {"title": "Tech stocks face minor headwind amid global cues", "published": "2024-01-05"},
    ]
    print(f"\n3. Ingesting news items with automated sentiment classification...")
    news_bundle = service.process_news(sample_news, ticker="TCS")
    for r in news_bundle.records:
        print(f"   Headline: '{r.features.get('Headline')[:45]}...' -> Sentiment: {r.features.get('Sentiment')} (Score: {r.features.get('Sentiment_Score')})")

    # 4. Test multi-source point-in-time aggregation
    print(f"\n4. Performing point-in-time aggregation...")
    combined_bundle = service.aggregate_bundles(
        market_bundle=market_bundle,
        earnings_bundle=earnings_bundle,
    )
    combined_df = combined_bundle.to_dataframe()
    print(f"   Combined DataFrame Shape: {combined_df.shape}")
    print(f"   Columns: {list(combined_df.columns)}")

    # 5. Verify Structured Output Contract
    print(f"\n5. Verifying Structured Output Contract:")
    sample_rec = combined_bundle.records[-1]
    print(f"   Ticker: {sample_rec.ticker}")
    print(f"   Timestamp: {sample_rec.timestamp}")
    print(f"   Available At: {sample_rec.provenance.available_at}")
    print(f"   Data Category: {sample_rec.provenance.data_category.value}")
    print(f"   Sample scale-invariant features: Close_to_SMA20={sample_rec.features.get('Close_to_SMA20')}, RSI_14={sample_rec.features.get('RSI_14')}")

    print("\n========================================")
    print("DATA TOOL STANDALONE TEST COMPLETE (SUCCESS)")
    print("========================================")


if __name__ == "__main__":
    main()
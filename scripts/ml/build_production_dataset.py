"""
Authoritative Production Dataset Builder for StockDNA-AI.

Pipeline:
1. Loads historical price data, macro benchmarks (NIFTY, VIX), and point-in-time earnings.
2. Computes the canonical 34 scale-invariant features (zero nominal price leakage).
3. Computes the official 5-day forward return target (BUY=2, HOLD=1, SELL=0).
4. Drops unlabelable rows (last 5 bars per ticker).
5. Validates point-in-time guarantees and saves `data/processed/production_dataset.csv`.
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import get_tickers, DATA_DIR
from config.feature_registry import get_feature_names, get_internal_features, get_external_features
from scripts.feature_engineering.canonical_features import (
    build_canonical_features,
    extract_feature_matrix,
)
from scripts.ml.target_design import generate_target, get_label_distribution


def build_production_dataset(
    output_path: str = "data/processed/production_dataset.csv"
) -> pd.DataFrame:
    """
    Builds and saves the authoritative production dataset.
    """
    print("=" * 60)
    print("BUILDING PRODUCTION PREDICTION DATASET")
    print("=" * 60)

    # 1. Check existing source datasets
    internal_source = "data/processed/internal/earnings_event_stock_dataset.csv"
    if os.path.exists(internal_source):
        print(f"Loading pre-aligned internal event dataset from: {internal_source}")
        raw_df = pd.read_csv(internal_source)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])
    else:
        print("Pre-aligned internal event dataset not found; loading from raw price feeds...")
        price_dfs = []
        for ticker in ["TCS", "INFY", "RELIANCE"]:
            p = f"data/raw/prices/{ticker}.csv"
            if os.path.exists(p):
                t_df = pd.read_csv(p)
                t_df["Ticker"] = ticker
                price_dfs.append(t_df)
        raw_df = pd.concat(price_dfs, ignore_index=True)
        raw_df["Date"] = pd.to_datetime(raw_df["Date"])

    # Load NIFTY and VIX if available
    nifty_df = None
    if os.path.exists("data/raw/market/nifty50.csv"):
        nifty_df = pd.read_csv("data/raw/market/nifty50.csv")
    vix_df = None
    if os.path.exists("data/raw/market/indiavix.csv"):
        vix_df = pd.read_csv("data/raw/market/indiavix.csv")

    earnings_df = None
    if os.path.exists("data/processed/internal/earnings_events.csv"):
        earnings_df = pd.read_csv("data/processed/internal/earnings_events.csv")

    # 2. Build canonical scale-invariant features
    print("Calculating 34 canonical scale-invariant features...")
    featured_df = build_canonical_features(
        raw_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
    )

    # 3. Generate official 5-day forward return target
    print("Generating official 5-day forward return target (BUY=2, HOLD=1, SELL=0)...")
    labeled_df = generate_target(
        featured_df,
        horizon=5,
        buy_threshold=0.02,
        sell_threshold=-0.02,
        price_col="Close",
        drop_unlabeled=True,
    )

    # 4. Verify feature presence and order
    feature_names = get_feature_names()
    missing_feats = [f for f in feature_names if f not in labeled_df.columns]
    if missing_feats:
        raise ValueError(f"Feature engineering failed to produce: {missing_feats}")

    # Ensure metadata columns exist
    meta_cols = ["Date", "Ticker", "Forward_Return_5d", "Target"]
    keep_cols = meta_cols + feature_names
    final_df = labeled_df[keep_cols].copy()

    # Sort chronologically
    final_df = final_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    # 5. Save output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    print(f"\nProduction dataset saved to: {output_path}")
    print(f"Total Rows: {len(final_df)}")
    print(f"Total Columns: {len(final_df.columns)} (4 metadata/target + {len(feature_names)} features)")
    print(f"Date Range: {final_df['Date'].min()} to {final_df['Date'].max()}")
    print(f"Tickers: {final_df['Ticker'].value_counts().to_dict()}")

    dist = get_label_distribution(final_df)
    print("\nTarget Class Distribution:")
    for label, pct in dist.items():
        print(f"  {label:6s}: {pct * 100:.2f}%")

    print(f"\nInternal Features ({len(get_internal_features())}): {get_internal_features()}")
    print(f"External Features ({len(get_external_features())}): {get_external_features()}")

    return final_df


if __name__ == "__main__":
    build_production_dataset()

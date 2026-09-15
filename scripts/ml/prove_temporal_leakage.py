"""
Programmatic Proof of Temporal Leakage Prevention and Purge/Embargo Guarantees.

Validates:
A. Target formula: Forward_Return_5d[t] = Close[t+5] / Close[t] - 1
B. Target construction performed independently per ticker
C. Target is never present in X
D. Final 5 observations per ticker cannot receive a valid target (must be NaN or dropped)
E. For every ticker, the final TRAIN observation's t+5 target date is strictly BEFORE the first VALIDATION feature date
F. For every ticker, the final VALIDATION observation's t+5 target date is strictly BEFORE the first TEST feature date
G. Exact dates used and exact number of trading observations separating boundaries
"""

import os
import sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.feature_registry import get_feature_names
from scripts.ml.train_production_model import load_and_split_data
from scripts.ml.target_design import generate_target


def prove_temporal_guarantees():
    print("=" * 80)
    print("PROGRAMMATIC PROOF OF TEMPORAL LEAKAGE PREVENTION AND PURGE/EMBARGO")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PROOF A: Target Formula
    # -------------------------------------------------------------------------
    print("\n[PROOF A] Target Formula: Forward_Return_5d[t] = Close[t+5] / Close[t] - 1")
    toy_close = [100.0, 102.0, 98.0, 105.0, 101.0, 110.0, 95.0, 100.0]
    toy_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=len(toy_close), freq="D"),
        "Close": toy_close,
        "Ticker": "TOY",
    })
    toy_labeled = generate_target(toy_df, horizon=5, drop_unlabeled=False)
    # At t=0, close is 100, t+5 is 110 -> return = 110 / 100 - 1 = +0.10 (+10%)
    expected_ret_0 = (toy_close[5] / toy_close[0]) - 1.0
    actual_ret_0 = toy_labeled.loc[0, "Forward_Return_5d"]
    assert np.isclose(expected_ret_0, actual_ret_0), f"Mismatch: expected {expected_ret_0}, got {actual_ret_0}"
    print(f"  Verified: t=0 Close={toy_close[0]}, t=5 Close={toy_close[5]} -> Return={actual_ret_0:.4f} (Expected: {expected_ret_0:.4f}) -> PROVED.")

    # -------------------------------------------------------------------------
    # PROOF B: Independent Target Construction per Ticker
    # -------------------------------------------------------------------------
    print("\n[PROOF B] Independent Target Construction per Ticker (No Cross-Ticker Bleeding)")
    multi_toy = pd.DataFrame({
        "Date": list(pd.date_range("2024-01-01", periods=6, freq="D")) + list(pd.date_range("2024-01-01", periods=6, freq="D")),
        "Close": [100.0] * 6 + [500.0] * 6,
        "Ticker": ["STOCK_A"] * 6 + ["STOCK_B"] * 6,
    })
    multi_labeled = generate_target(multi_toy, horizon=5, drop_unlabeled=True)
    # If grouping failed, STOCK_A t=5 would shift into STOCK_B close (500 / 100 - 1 = +400%)
    # With grouping, STOCK_A close is 100 at t=0 and 100 at t=5 -> return is 0.0
    assert (multi_labeled["Forward_Return_5d"] == 0.0).all(), "Target shifted across ticker boundary!"
    print(f"  Verified: Multi-ticker panel returns are exactly 0.0, zero cross-ticker target bleeding -> PROVED.")

    # -------------------------------------------------------------------------
    # PROOF C: Target is Never in X Feature Matrix
    # -------------------------------------------------------------------------
    print("\n[PROOF C] Target is Never Present in X Feature Matrix")
    feature_names = get_feature_names()
    forbidden = ["Target", "target", "Forward_Return_5d", "future_price", "Target_Return"]
    for f in forbidden:
        assert f not in feature_names, f"CRITICAL: Forbidden column {f} found in feature matrix!"
    print(f"  Verified: All {len(feature_names)} features in production registry are free of target/future references -> PROVED.")

    # -------------------------------------------------------------------------
    # PROOF D: Final 5 Observations Cannot Receive Target
    # -------------------------------------------------------------------------
    print("\n[PROOF D] Final 5 Observations per Ticker Cannot Receive Target (Dropped)")
    toy_all = generate_target(toy_df, horizon=5, drop_unlabeled=False)
    final_5 = toy_all.tail(5)
    assert final_5["Forward_Return_5d"].isna().all(), "Final 5 rows unexpectedly received a target!"
    assert final_5["Target"].isna().all(), "Final 5 rows unexpectedly received a class label!"
    print(f"  Verified: Last 5 rows of input series have Forward_Return_5d=NaN and Target=NaN -> PROVED.")

    # -------------------------------------------------------------------------
    # PROOFS E, F, G: Exact Split Dates & Zero-Overlap Trading Day Embargo
    # -------------------------------------------------------------------------
    print("\n[PROOFS E, F, G] Split Boundaries & Non-Overlapping Trading Day Embargo")
    df = pd.read_csv("data/processed/production_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    train_df, val_df, test_df = load_and_split_data()

    train_start = train_df["Date"].min().strftime("%Y-%m-%d")
    train_end = train_df["Date"].max().strftime("%Y-%m-%d")
    val_start = val_df["Date"].min().strftime("%Y-%m-%d")
    val_end = val_df["Date"].max().strftime("%Y-%m-%d")
    test_start = test_df["Date"].min().strftime("%Y-%m-%d")
    test_end = test_df["Date"].max().strftime("%Y-%m-%d")

    print(f"  Split Date Ranges:")
    print(f"    TRAIN : {train_start} to {train_end} ({len(train_df)} rows, {len(train_df)//3} bars/ticker)")
    print(f"    VAL   : {val_start} to {val_end} ({len(val_df)} rows, {len(val_df)//3} bars/ticker)")
    print(f"    TEST  : {test_start} to {test_end} ({len(test_df)} rows, {len(test_df)//3} bars/ticker)")

    # Per-ticker trading-day forward return realization check
    tickers = sorted(df["Ticker"].unique())
    print("\n  Per-Ticker Embargo Verification Table:")
    print(f"  {'Ticker':<8} | {'Boundary':<12} | {'Observation Date':<16} | {'Realized (t+5)':<16} | {'Next Split Start':<16} | {'Purge Margin':<12} | {'Status'}")
    print("  " + "-" * 105)

    all_passed = True
    for ticker in tickers:
        sub = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

        # 1. Train to Val Boundary
        last_tr_idx = sub[sub["Date"] <= train_df["Date"].max()].index[-1]
        tr_obs_date = sub.loc[last_tr_idx, "Date"].strftime("%Y-%m-%d")
        tr_t5_date = sub.loc[last_tr_idx + 5, "Date"].strftime("%Y-%m-%d")
        val_first_idx = sub[sub["Date"] >= val_df["Date"].min()].index[0]
        val_first_date = sub.loc[val_first_idx, "Date"].strftime("%Y-%m-%d")

        # Trading days separating realized target from val start
        trading_days_gap_tr_val = val_first_idx - (last_tr_idx + 5)
        is_tr_val_safe = sub.loc[last_tr_idx + 5, "Date"] < sub.loc[val_first_idx, "Date"]
        status_tr_val = "PASS (CLEAN)" if (is_tr_val_safe and trading_days_gap_tr_val >= 1) else "FAIL"

        print(f"  {ticker:<8} | {'TRAIN -> VAL':<12} | {tr_obs_date:<16} | {tr_t5_date:<16} | {val_first_date:<16} | {trading_days_gap_tr_val} trading bar(s) | {status_tr_val}")

        # 2. Val to Test Boundary
        last_val_idx = sub[sub["Date"] <= val_df["Date"].max()].index[-1]
        val_obs_date = sub.loc[last_val_idx, "Date"].strftime("%Y-%m-%d")
        val_t5_date = sub.loc[last_val_idx + 5, "Date"].strftime("%Y-%m-%d")
        test_first_idx = sub[sub["Date"] >= test_df["Date"].min()].index[0]
        test_first_date = sub.loc[test_first_idx, "Date"].strftime("%Y-%m-%d")

        trading_days_gap_val_test = test_first_idx - (last_val_idx + 5)
        is_val_test_safe = sub.loc[last_val_idx + 5, "Date"] < sub.loc[test_first_idx, "Date"]
        status_val_test = "PASS (CLEAN)" if (is_val_test_safe and trading_days_gap_val_test >= 1) else "FAIL"

        print(f"  {ticker:<8} | {'VAL -> TEST':<12} | {val_obs_date:<16} | {val_t5_date:<16} | {test_first_date:<16} | {trading_days_gap_val_test} trading bar(s) | {status_val_test}")

        if not (is_tr_val_safe and is_val_test_safe):
            all_passed = False

    assert all_passed, "CRITICAL: Embargo boundary overlap detected!"
    print("  " + "-" * 105)
    print("\nAll temporal leakage and embargo conditions A-G have been mathematically PROVED.")
    print("=" * 80)


if __name__ == "__main__":
    prove_temporal_guarantees()

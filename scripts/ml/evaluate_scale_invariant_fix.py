"""
Chronological Validation of Scale-Invariant Earnings Feature Improvement.
Evaluates the feature fix STRICTLY on pre-2024 chronological splits (2018-2023).
Does not touch or iterate on the 2024 holdout.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_engineering.scale_invariant_earnings import (
    get_scale_invariant_feature_matrix,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_dataset.csv"
CONTRACT_PATH = PROJECT_ROOT / "models" / "production" / "model_contract.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "diagnostics" / "pre2024_fix_evaluation.json"
TICKERS = ["INFY", "RELIANCE", "TCS"]


def evaluate_split_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sell_f1": float(f1_score(y_true == 0, y_pred == 0, zero_division=0)),
        "hold_f1": float(f1_score(y_true == 1, y_pred == 1, zero_division=0)),
        "buy_f1": float(f1_score(y_true == 2, y_pred == 2, zero_division=0)),
    }


def main():
    print("=" * 80)
    print("PRE-2024 CHRONOLOGICAL EVALUATION: SCALE-INVARIANT EARNINGS IMPROVEMENT")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Enforcing: 2024 holdout strictly excluded from model comparison.")
    print("=" * 80)

    # 1. Load dataset and production contract
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    with open(CONTRACT_PATH, "r") as f:
        contract = json.load(f)

    base_features = contract["feature_names"]
    imputation_dict = contract["imputation_values"]

    # Fill base features using contract medians
    for feat in base_features:
        fill_val = imputation_dict.get(feat, 0.0)
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(fill_val)

    # Merge Close price for scale-invariant computation
    price_dfs = []
    for t in TICKERS:
        raw_p = PROJECT_ROOT / "data" / "raw" / "prices" / f"{t}.csv"
        p_df = pd.read_csv(raw_p)
        p_df["Ticker"] = t
        p_df["Date"] = pd.to_datetime(p_df["Date"])
        price_dfs.append(p_df[["Date", "Ticker", "Close"]])
    prices_df = pd.concat(price_dfs).reset_index(drop=True)
    df = pd.merge(df, prices_df, on=["Date", "Ticker"], how="left")

    # Generate 34 scale-invariant feature matrix
    X_scaled, scaled_feature_names = get_scale_invariant_feature_matrix(
        df, base_features, replace_nominal_eps=True
    )
    for col in scaled_feature_names:
        df[col] = X_scaled[col]

    # Pre-2024 chronological validation splits
    splits = {
        "Inner Split A (Val 2021)": {"tr_yrs": [2018, 2019, 2020], "val_yr": 2021},
        "Inner Split B (Val 2022)": {"tr_yrs": [2018, 2019, 2020, 2021], "val_yr": 2022},
        "Inner Split C (Val 2023)": {"tr_yrs": [2018, 2019, 2020, 2021, 2022], "val_yr": 2023},
    }

    results = {
        "evaluation_scope": "Pre-2024 Development Data Only (2018-2023)",
        "2024_holdout_status": "EXCLUDED (Untouched out-of-sample holdout)",
        "feature_change": "Replaced nominal Reported_EPS and EPS_Estimate with scale-invariant Earnings_Yield and Forward_Earnings_Yield",
        "feature_count": len(scaled_feature_names),
        "models": {},
    }

    for model_key, m_type in [("XGBoost", "xgb"), ("HistGradientBoosting", "hgb")]:
        canon_splits, scaled_splits = {}, {}

        for s_name, s_spec in splits.items():
            tr_list, val_list = [], []
            for t in TICKERS:
                tdf = df[df["Ticker"] == t].sort_values("Date").reset_index(drop=True)
                tr = tdf[tdf["Date"].dt.year.isin(s_spec["tr_yrs"])].iloc[:-5]
                vl = tdf[tdf["Date"].dt.year == s_spec["val_yr"]].iloc[:-5]
                tr_list.append(tr)
                val_list.append(vl)

            tr_df = pd.concat(tr_list).reset_index(drop=True)
            val_df = pd.concat(val_list).reset_index(drop=True)

            y_tr = tr_df["Target"].values.astype(int)
            y_vl = val_df["Target"].values.astype(int)

            # Canonical features
            X_tr_c = tr_df[base_features]
            X_vl_c = val_df[base_features]

            # Scaled features
            X_tr_s = tr_df[scaled_feature_names]
            X_vl_s = val_df[scaled_feature_names]

            if m_type == "xgb":
                sw = compute_sample_weight("balanced", y_tr)
                mc = XGBClassifier(
                    objective="multi:softprob", num_class=3, n_estimators=200, learning_rate=0.03,
                    max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    random_state=42, eval_metric="mlogloss"
                )
                mc.fit(X_tr_c, y_tr, sample_weight=sw, verbose=False)
                ms = XGBClassifier(
                    objective="multi:softprob", num_class=3, n_estimators=200, learning_rate=0.03,
                    max_depth=4, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                    random_state=42, eval_metric="mlogloss"
                )
                ms.fit(X_tr_s, y_tr, sample_weight=sw, verbose=False)
            else:
                mc = HistGradientBoostingClassifier(
                    learning_rate=0.03, max_iter=150, max_depth=4, min_samples_leaf=15,
                    l2_regularization=1.5, class_weight="balanced", random_state=42
                )
                mc.fit(X_tr_c, y_tr)
                ms = HistGradientBoostingClassifier(
                    learning_rate=0.03, max_iter=150, max_depth=4, min_samples_leaf=15,
                    l2_regularization=1.5, class_weight="balanced", random_state=42
                )
                ms.fit(X_tr_s, y_tr)

            pred_c = mc.predict(X_vl_c)
            pred_s = ms.predict(X_vl_s)

            canon_splits[s_name] = evaluate_split_metrics(y_vl, pred_c)
            scaled_splits[s_name] = evaluate_split_metrics(y_vl, pred_s)

        # Compute aggregate means across the 3 splits
        mean_c_acc = float(np.mean([m["accuracy"] for m in canon_splits.values()]))
        mean_c_bal = float(np.mean([m["balanced_accuracy"] for m in canon_splits.values()]))
        mean_c_f1 = float(np.mean([m["macro_f1"] for m in canon_splits.values()]))

        mean_s_acc = float(np.mean([m["accuracy"] for m in scaled_splits.values()]))
        mean_s_bal = float(np.mean([m["balanced_accuracy"] for m in scaled_splits.values()]))
        mean_s_f1 = float(np.mean([m["macro_f1"] for m in scaled_splits.values()]))

        results["models"][model_key] = {
            "canonical_features": {
                "splits": canon_splits,
                "mean_accuracy": mean_c_acc,
                "mean_balanced_accuracy": mean_c_bal,
                "mean_macro_f1": mean_c_f1,
            },
            "scale_invariant_features": {
                "splits": scaled_splits,
                "mean_accuracy": mean_s_acc,
                "mean_balanced_accuracy": mean_s_bal,
                "mean_macro_f1": mean_s_f1,
            },
            "delta": {
                "accuracy": mean_s_acc - mean_c_acc,
                "balanced_accuracy": mean_s_bal - mean_c_bal,
                "macro_f1": mean_s_f1 - mean_c_f1,
            }
        }

        print(f"\n--- {model_key} PRE-2024 CHRONOLOGICAL RESULTS ---")
        print(f"Canonical Mean Macro F1: {mean_c_f1*100:.2f}% | BalAcc: {mean_c_bal*100:.2f}% | Acc: {mean_c_acc*100:.2f}%")
        print(f"Scaled    Mean Macro F1: {mean_s_f1*100:.2f}% | BalAcc: {mean_s_bal*100:.2f}% | Acc: {mean_s_acc*100:.2f}%")
        print(f"Delta:    Macro F1: {(mean_s_f1 - mean_c_f1)*100:+.2f}% | BalAcc: {(mean_s_bal - mean_c_bal)*100:+.2f}% | Acc: {(mean_s_acc - mean_c_acc)*100:+.2f}%")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation metrics successfully written to: {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()

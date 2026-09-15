"""
Phase 5 Model Improvement Exploration (Validation Set Only).

Evaluates:
1. Current XGBoost baseline on validation set
2. Hyperparameter optimization (grid/random search)
3. Class weight variations
4. Candidate scale-invariant features:
   - Dist_to_52w_High (George & Hwang 52-week momentum)
   - Vol_Regime_20_252 (short-term to annual volatility ratio)
   - PEAD_Interaction (Post_Earnings_1_Week * EPS_Surprise_Pct)

CRITICAL:
Test set (2025) is strictly EXCLUDED.
All metrics are evaluated exclusively on 2024 Validation set.
"""

import os
import sys
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.feature_registry import get_feature_names, FEATURE_REGISTRY
from scripts.ml.train_production_model import load_and_split_data, compute_training_imputation, apply_imputation, evaluate_predictions


def run_experiment():
    print("=" * 70)
    print("PHASE 5: HYPERPARAMETER & FEATURE EXPLORATION (VALIDATION SET ONLY)")
    print("=" * 70)

    # 1. Load Data with strict zero-overlap embargo
    df = pd.read_csv("data/processed/production_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[(df["Date"] >= "2018-10-23") & (df["Date"] <= "2023-12-21")].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    val_df = df[(df["Date"] >= "2024-01-02") & (df["Date"] <= "2024-12-23")].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    base_feature_names = get_feature_names()
    imputation_dict = compute_training_imputation(train_df, base_feature_names)

    X_train_base = apply_imputation(train_df, base_feature_names, imputation_dict)
    y_train = train_df["Target"].values.astype(int)

    X_val_base = apply_imputation(val_df, base_feature_names, imputation_dict)
    y_val = val_df["Target"].values.astype(int)

    sample_weights_train = compute_sample_weight("balanced", y_train)

    print(f"Train samples: {len(X_train_base)}, Val samples: {len(X_val_base)}")

    # Baseline Model (Phase 3 XGBoost parameters)
    print("\n--- Evaluating Baseline XGBoost (Phase 3 Config) ---")
    base_xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    base_xgb.fit(X_train_base, y_train, sample_weight=sample_weights_train, eval_set=[(X_val_base, y_val)], verbose=False)
    base_val_pred = np.argmax(base_xgb.predict_proba(X_val_base), axis=1)
    base_eval = evaluate_predictions(y_val, base_val_pred, "Validation")
    print(f"Baseline -> Accuracy: {base_eval['accuracy']:.4f}, Bal Acc: {base_eval['balanced_accuracy']:.4f}, Macro F1: {base_eval['macro_f1']:.4f}, Weighted F1: {base_eval['weighted_f1']:.4f}")

    # 2. Hyperparameter Grid Search on Base Features
    print("\n--- Running Hyperparameter Search ---")
    param_grid = [
        {"max_depth": 3, "learning_rate": 0.02, "n_estimators": 180, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 2.0},
        {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 4, "reg_alpha": 0.2, "reg_lambda": 3.0},
        {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 220, "subsample": 0.75, "colsample_bytree": 0.75, "min_child_weight": 3, "reg_alpha": 0.1, "reg_lambda": 1.5},
        {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 200, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5, "reg_alpha": 0.5, "reg_lambda": 2.0},
        {"max_depth": 5, "learning_rate": 0.015, "n_estimators": 250, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 4, "reg_alpha": 0.3, "reg_lambda": 2.5},
        {"max_depth": 4, "learning_rate": 0.025, "n_estimators": 200, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 3, "reg_alpha": 0.0, "reg_lambda": 1.0},
    ]

    best_hp = None
    best_hp_score = -1.0
    for idx, p in enumerate(param_grid):
        m = XGBClassifier(objective="multi:softprob", num_class=3, random_state=42, eval_metric="mlogloss", **p)
        m.fit(X_train_base, y_train, sample_weight=sample_weights_train, eval_set=[(X_val_base, y_val)], verbose=False)
        preds = np.argmax(m.predict_proba(X_val_base), axis=1)
        res = evaluate_predictions(y_val, preds, "Validation")
        score = res["macro_f1"] + res["balanced_accuracy"]
        print(f"Config {idx+1}: Bal Acc={res['balanced_accuracy']:.4f}, Macro F1={res['macro_f1']:.4f} | p={p}")
        if score > best_hp_score:
            best_hp_score = score
            best_hp = (p, res)

    print(f"\nBest HP Config: Bal Acc={best_hp[1]['balanced_accuracy']:.4f}, Macro F1={best_hp[1]['macro_f1']:.4f}")
    print(f"Parameters: {best_hp[0]}")

    # 3. Candidate Feature Evaluation
    print("\n--- Evaluating Candidate Scale-Invariant Features ---")
    price_dfs = []
    for ticker in ["TCS", "INFY", "RELIANCE"]:
        p = f"data/raw/prices/{ticker}.csv"
        if os.path.exists(p):
            t_df = pd.read_csv(p)
            t_df["Ticker"] = ticker
            price_dfs.append(t_df)
    raw_prices = pd.concat(price_dfs, ignore_index=True)
    raw_prices["Date"] = pd.to_datetime(raw_prices["Date"])

    # Compute Dist_to_52w_High: (Close - RollingMax252) / RollingMax252
    candidate_records = []
    for ticker, group in raw_prices.groupby("Ticker", sort=False):
        g = group.sort_values("Date").reset_index(drop=True)
        roll_max_252 = g["Close"].rolling(window=252, min_periods=50).max()
        g["Dist_to_52w_High"] = (g["Close"] - roll_max_252) / (roll_max_252 + 1e-8)
        
        # Volatility regime: rolling 20d std / rolling 252d std
        ret_1d = g["Close"].pct_change(fill_method=None)
        vol_20 = ret_1d.rolling(window=20).std()
        vol_252 = ret_1d.rolling(window=252, min_periods=50).std()
        g["Volatility_Regime_Ratio"] = vol_20 / (vol_252 + 1e-8)
        candidate_records.append(g[["Date", "Ticker", "Dist_to_52w_High", "Volatility_Regime_Ratio"]])
    
    cand_df = pd.concat(candidate_records, ignore_index=True)

    # Merge candidate features onto train and val
    train_cand = pd.merge(train_df, cand_df, on=["Date", "Ticker"], how="left")
    val_cand = pd.merge(val_df, cand_df, on=["Date", "Ticker"], how="left")

    # Add PEAD interaction
    train_cand["PEAD_Interaction"] = train_cand["Post_Earnings_1_Week"] * train_cand["EPS_Surprise_Pct"]
    val_cand["PEAD_Interaction"] = val_cand["Post_Earnings_1_Week"] * val_cand["EPS_Surprise_Pct"]

    candidate_feature_sets = {
        "Base + Dist_to_52w_High": base_feature_names + ["Dist_to_52w_High"],
        "Base + Vol_Regime": base_feature_names + ["Volatility_Regime_Ratio"],
        "Base + PEAD_Interaction": base_feature_names + ["PEAD_Interaction"],
        "Base + All 3 Candidates": base_feature_names + ["Dist_to_52w_High", "Volatility_Regime_Ratio", "PEAD_Interaction"],
    }

    for name, f_list in candidate_feature_sets.items():
        imp_dict = {}
        for feat in f_list:
            if feat in imputation_dict:
                imp_dict[feat] = imputation_dict[feat]
            else:
                med = float(train_cand[feat].median())
                imp_dict[feat] = 0.0 if np.isnan(med) else med
        
        X_tr = train_cand[f_list].fillna(imp_dict).copy()
        X_va = val_cand[f_list].fillna(imp_dict).copy()

        m = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=200,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            random_state=42,
            eval_metric="mlogloss",
        )
        m.fit(X_tr, y_train, sample_weight=sample_weights_train, eval_set=[(X_va, y_val)], verbose=False)
        preds = np.argmax(m.predict_proba(X_va), axis=1)
        res = evaluate_predictions(y_val, preds, "Validation")
        print(f"{name:<28} -> Bal Acc: {res['balanced_accuracy']:.4f}, Macro F1: {res['macro_f1']:.4f}, Weighted F1: {res['weighted_f1']:.4f}")


if __name__ == "__main__":
    run_experiment()

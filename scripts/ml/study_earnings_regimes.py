"""
Scientific Post-Earnings Regime Study for StockDNA-AI.

Investigates whether market behavior and model predictability differ around earnings announcements.

Evaluates:
- All observations
- Post-earnings 0–1 trading days (Immediate announcement reaction)
- Post-earnings 2–5 trading days (Short-term post-earnings announcement drift / PEAD)
- Post-earnings 6–20 trading days (Medium-term consolidation)
- Non-earnings observations (>20 trading days)

Metrics per regime:
- Sample size (N)
- Class balance (% SELL, % HOLD, % BUY)
- Model predictive metrics: Accuracy, Balanced Accuracy, Macro F1
- Realized return volatility
- Internal vs External feature SHAP contribution

Saves output to `results/earnings_regime_analysis.csv`.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from xgboost import XGBClassifier
import shap

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.feature_registry import get_feature_names, get_internal_features, get_external_features
from scripts.ml.train_production_model import compute_training_imputation, apply_imputation


def analyze_earnings_regimes():
    print("=" * 70)
    print("STUDYING POST-EARNINGS PREDICTABILITY AND REGIMES")
    print("=" * 70)

    df = pd.read_csv("data/processed/production_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Splits
    train_df = df[(df["Date"] >= "2018-10-23") & (df["Date"] <= "2023-12-21")].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    val_df = df[(df["Date"] >= "2024-01-02") & (df["Date"] <= "2024-12-23")].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    test_df = df[(df["Date"] >= "2025-01-02") & (df["Date"] <= "2025-12-19")].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_names = get_feature_names()
    internal_features = get_internal_features()
    external_features = get_external_features()

    imp = compute_training_imputation(train_df, feature_names)

    X_train = apply_imputation(train_df, feature_names, imp)
    y_train = train_df["Target"].values.astype(int)

    # Train production XGBoost on train set
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight("balanced", y_train)

    xgb = XGBClassifier(
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
    xgb.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(xgb)

    # Evaluate across combined evaluation sets (Val + Test) as well as splits
    eval_df = pd.concat([val_df, test_df], ignore_index=True)
    X_eval = apply_imputation(eval_df, feature_names, imp)
    y_eval = eval_df["Target"].values.astype(int)

    eval_preds = np.argmax(xgb.predict_proba(X_eval), axis=1)
    eval_df["Predicted"] = eval_preds

    # Compute SHAP values for predicted class
    shap_vals_all = explainer.shap_values(X_eval)  # shape (N, 34, 3)

    internal_indices = [feature_names.index(f) for f in internal_features]
    external_indices = [feature_names.index(f) for f in external_features]

    # Attribution per row
    internal_pcts = []
    external_pcts = []
    internal_mags = []
    external_mags = []

    for i in range(len(eval_df)):
        pred_c = eval_preds[i]
        sample_shap = shap_vals_all[i, :, pred_c]
        int_mag = np.sum(np.abs(sample_shap[internal_indices]))
        ext_mag = np.sum(np.abs(sample_shap[external_indices]))
        tot_mag = int_mag + ext_mag
        if tot_mag > 0:
            int_pct = (int_mag / tot_mag) * 100.0
            ext_pct = (ext_mag / tot_mag) * 100.0
        else:
            int_pct = 0.0
            ext_pct = 0.0
        internal_pcts.append(int_pct)
        external_pcts.append(ext_pct)
        internal_mags.append(int_mag)
        external_mags.append(ext_mag)

    eval_df["Internal_Pct"] = internal_pcts
    eval_df["External_Pct"] = external_pcts
    eval_df["Internal_Mag"] = internal_mags
    eval_df["External_Mag"] = external_mags

    # Regimes to evaluate
    regimes = {
        "All Observations (Val+Test)": eval_df,
        "Post-Earnings 0-1 Days": eval_df[eval_df["Days_Since_Earnings"].between(0, 1)],
        "Post-Earnings 2-5 Days": eval_df[eval_df["Days_Since_Earnings"].between(2, 5)],
        "Post-Earnings 6-20 Days": eval_df[eval_df["Days_Since_Earnings"].between(6, 20)],
        "Non-Earnings (>20 Days)": eval_df[eval_df["Days_Since_Earnings"] > 20],
        "Post-Earnings Window (0-5 Days)": eval_df[eval_df["Days_Since_Earnings"].between(0, 5)],
    }

    results = []
    for reg_name, reg_sub in regimes.items():
        n_samples = len(reg_sub)
        if n_samples == 0:
            continue

        y_t = reg_sub["Target"].values.astype(int)
        y_p = reg_sub["Predicted"].values.astype(int)

        acc = accuracy_score(y_t, y_p)
        bal_acc = balanced_accuracy_score(y_t, y_p) if len(np.unique(y_t)) > 1 else acc
        macro_f1 = f1_score(y_t, y_p, average="macro", zero_division=0)

        sell_pct = (y_t == 0).mean() * 100.0
        hold_pct = (y_t == 1).mean() * 100.0
        buy_pct = (y_t == 2).mean() * 100.0

        mean_int_pct = reg_sub["Internal_Pct"].mean()
        median_int_pct = reg_sub["Internal_Pct"].median()
        mean_ext_pct = reg_sub["External_Pct"].mean()
        median_ext_pct = reg_sub["External_Pct"].median()

        ret_vol = reg_sub["Forward_Return_5d"].std() * 100.0

        results.append({
            "Regime": reg_name,
            "Sample_Count": n_samples,
            "Sample_Pct": round(n_samples / len(eval_df) * 100.0, 2),
            "SELL_Pct": round(sell_pct, 2),
            "HOLD_Pct": round(hold_pct, 2),
            "BUY_Pct": round(buy_pct, 2),
            "Forward_5d_Vol_Pct": round(ret_vol, 2),
            "Accuracy": round(acc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
            "Macro_F1": round(macro_f1, 4),
            "Mean_Internal_SHAP_Pct": round(mean_int_pct, 2),
            "Median_Internal_SHAP_Pct": round(median_int_pct, 2),
            "Mean_External_SHAP_Pct": round(mean_ext_pct, 2),
            "Median_External_SHAP_Pct": round(median_ext_pct, 2),
        })

    res_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/earnings_regime_analysis.csv", index=False)
    print(f"Saved earnings regime analysis to results/earnings_regime_analysis.csv\n")

    print(f"{'Regime':<34} | {'N':<6} | {'Bal Acc':<8} | {'Macro F1':<8} | {'Internal %':<10} | {'External %':<10}")
    print("-" * 88)
    for _, r in res_df.iterrows():
        print(f"{r['Regime']:<34} | {r['Sample_Count']:<6} | {r['Balanced_Accuracy']:<8.4f} | {r['Macro_F1']:<8.4f} | {r['Mean_Internal_SHAP_Pct']:<10.2f} | {r['Mean_External_SHAP_Pct']:<10.2f}")

    return res_df


if __name__ == "__main__":
    analyze_earnings_regimes()

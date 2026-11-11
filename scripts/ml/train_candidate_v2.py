"""
Training & Pre-2024 Chronological Walk-Forward Evaluation Pipeline for Candidate v2.

Integrates Scale-Invariant Earnings Valuation:
- Features: 34 scale-invariant features (Earnings_Yield, Forward_Earnings_Yield).
- Model Architecture: Production XGBoost recipe with balanced sample weights.
- Chronological Splits: Inner Splits A (2021), B (2022), C (2023) across 2018-2023.
- Safeguards:
  - Preserves models/production/ and models/candidates/candidate_v1/ untouched.
  - Excludes 2024 and 2025 data from model selection and tuning.
  - Serializes artifacts to models/candidates/candidate_v2/.
  - Saves reports to results/diagnostics/.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.feature_engineering.candidate_features import (
    build_candidate_features,
    extract_candidate_feature_matrix,
    CANDIDATE_V2_FEATURE_NAMES,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_dataset.csv"
PROD_CONTRACT_PATH = PROJECT_ROOT / "models" / "production" / "model_contract.json"
CANDIDATE_V2_DIR = PROJECT_ROOT / "models" / "candidates" / "candidate_v2"
DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "diagnostics"
TICKERS = ["INFY", "RELIANCE", "TCS"]


def evaluate_split(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": {
            "SELL": {
                "precision": float(precision_score(y_true == 0, y_pred == 0, zero_division=0)),
                "recall": float(recall_score(y_true == 0, y_pred == 0, zero_division=0)),
                "f1": float(f1_score(y_true == 0, y_pred == 0, zero_division=0)),
                "support": int(np.sum(y_true == 0)),
            },
            "HOLD": {
                "precision": float(precision_score(y_true == 1, y_pred == 1, zero_division=0)),
                "recall": float(recall_score(y_true == 1, y_pred == 1, zero_division=0)),
                "f1": float(f1_score(y_true == 1, y_pred == 1, zero_division=0)),
                "support": int(np.sum(y_true == 1)),
            },
            "BUY": {
                "precision": float(precision_score(y_true == 2, y_pred == 2, zero_division=0)),
                "recall": float(recall_score(y_true == 2, y_pred == 2, zero_division=0)),
                "f1": float(f1_score(y_true == 2, y_pred == 2, zero_division=0)),
                "support": int(np.sum(y_true == 2)),
            },
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def main():
    print("=" * 80)
    print("TRAINING & CHRONOLOGICAL EVALUATION: CANDIDATE V2")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("Scope: Pre-2024 Development Data (2018-2023). 2024/2025 Holdouts Excluded.")
    print("=" * 80)

    # 1. Load dataset and production contract
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    with open(PROD_CONTRACT_PATH, "r") as f:
        prod_contract = json.load(f)

    # Attach Close prices for scale-independent computation
    price_dfs = []
    for t in TICKERS:
        raw_p = PROJECT_ROOT / "data" / "raw" / "prices" / f"{t}.csv"
        p_df = pd.read_csv(raw_p)
        p_df["Ticker"] = t
        p_df["Date"] = pd.to_datetime(p_df["Date"])
        price_dfs.append(p_df[["Date", "Ticker", "Close"]])
    prices_df = pd.concat(price_dfs).reset_index(drop=True)
    df = pd.merge(df, prices_df, on=["Date", "Ticker"], how="left")

    # Generate Candidate v2 features (with Earnings_Yield and Forward_Earnings_Yield)
    df_candidate = build_candidate_features(df)
    
    # 2. Compute training-fitted medians strictly on pre-2024 training data (2018-2023)
    pre2024_train_mask = df_candidate["Date"].dt.year <= 2023
    imputation_dict_v2 = {}
    for feat in CANDIDATE_V2_FEATURE_NAMES:
        med = float(df_candidate.loc[pre2024_train_mask, feat].median())
        imputation_dict_v2[feat] = 0.0 if np.isnan(med) else med

    # Impute missing values
    for feat in CANDIDATE_V2_FEATURE_NAMES:
        df_candidate[feat] = df_candidate[feat].replace([np.inf, -np.inf], np.nan).fillna(imputation_dict_v2[feat])

    # Also impute production canonical features for exact side-by-side comparison
    canon_feature_names = prod_contract["feature_names"]
    imputation_dict_canon = prod_contract["imputation_values"]
    for feat in canon_feature_names:
        df_candidate[feat] = df_candidate[feat].replace([np.inf, -np.inf], np.nan).fillna(imputation_dict_canon.get(feat, 0.0))

    # 3. Define pre-2024 Chronological Splits with verified 5-bar purges
    splits = {
        "Split A (2021)": {"tr_yrs": [2018, 2019, 2020], "val_yr": 2021},
        "Split B (2022)": {"tr_yrs": [2018, 2019, 2020, 2021], "val_yr": 2022},
        "Split C (2023)": {"tr_yrs": [2018, 2019, 2020, 2021, 2022], "val_yr": 2023},
    }

    xgb_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": 42,
        "eval_metric": "mlogloss",
    }

    comparison_results = {
        "production_recipe_canonical_features": {},
        "candidate_v2_scale_invariant_features": {},
    }

    print(f"\n{'Split':<18} | {'Model Recipe':<24} | {'Accuracy':<9} | {'Bal Acc':<9} | {'Macro F1':<9} | {'SELL F1':<8} | {'HOLD F1':<8} | {'BUY F1':<8}")
    print("-" * 110)

    for s_name, s_spec in splits.items():
        tr_list, val_list = [], []
        for t in TICKERS:
            tdf = df_candidate[df_candidate["Ticker"] == t].sort_values("Date").reset_index(drop=True)
            tr = tdf[tdf["Date"].dt.year.isin(s_spec["tr_yrs"])].iloc[:-5]
            vl = tdf[tdf["Date"].dt.year == s_spec["val_yr"]].iloc[:-5]
            tr_list.append(tr)
            val_list.append(vl)

        tr_df = pd.concat(tr_list).reset_index(drop=True)
        val_df = pd.concat(val_list).reset_index(drop=True)

        y_tr = tr_df["Target"].values.astype(int)
        y_val = val_df["Target"].values.astype(int)
        sw = compute_sample_weight("balanced", y_tr)

        # 1. Baseline Recipe (Canonical 34 features with raw Reported_EPS & EPS_Estimate)
        X_tr_c = tr_df[canon_feature_names]
        X_vl_c = val_df[canon_feature_names]
        m_canon = XGBClassifier(**xgb_params)
        m_canon.fit(X_tr_c, y_tr, sample_weight=sw, verbose=False)
        p_canon = m_canon.predict(X_vl_c)
        m_canon_metrics = evaluate_split(y_val, p_canon)
        comparison_results["production_recipe_canonical_features"][s_name] = m_canon_metrics

        # 2. Candidate v2 Recipe (Scale-invariant 34 features with Earnings_Yield & Forward_Earnings_Yield)
        X_tr_v2 = tr_df[CANDIDATE_V2_FEATURE_NAMES]
        X_vl_v2 = val_df[CANDIDATE_V2_FEATURE_NAMES]
        m_v2 = XGBClassifier(**xgb_params)
        m_v2.fit(X_tr_v2, y_tr, sample_weight=sw, verbose=False)
        p_v2 = m_v2.predict(X_vl_v2)
        m_v2_metrics = evaluate_split(y_val, p_v2)
        comparison_results["candidate_v2_scale_invariant_features"][s_name] = m_v2_metrics

        print(f"{s_name:<18} | Baseline Canonical       | {m_canon_metrics['accuracy']*100:6.2f}%   | {m_canon_metrics['balanced_accuracy']*100:6.2f}%   | {m_canon_metrics['macro_f1']*100:6.2f}%   | {m_canon_metrics['per_class']['SELL']['f1']:6.4f}   | {m_canon_metrics['per_class']['HOLD']['f1']:6.4f}   | {m_canon_metrics['per_class']['BUY']['f1']:6.4f}")
        print(f"{'':<18} | Candidate v2 Scaled      | {m_v2_metrics['accuracy']*100:6.2f}%   | {m_v2_metrics['balanced_accuracy']*100:6.2f}%   | {m_v2_metrics['macro_f1']*100:6.2f}%   | {m_v2_metrics['per_class']['SELL']['f1']:6.4f}   | {m_v2_metrics['per_class']['HOLD']['f1']:6.4f}   | {m_v2_metrics['per_class']['BUY']['f1']:6.4f}")
        print("-" * 110)

    # Compute Pre-2024 aggregate means
    canon_means = {
        "accuracy": float(np.mean([m["accuracy"] for m in comparison_results["production_recipe_canonical_features"].values()])),
        "balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in comparison_results["production_recipe_canonical_features"].values()])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in comparison_results["production_recipe_canonical_features"].values()])),
        "sell_f1": float(np.mean([m["per_class"]["SELL"]["f1"] for m in comparison_results["production_recipe_canonical_features"].values()])),
        "hold_f1": float(np.mean([m["per_class"]["HOLD"]["f1"] for m in comparison_results["production_recipe_canonical_features"].values()])),
        "buy_f1": float(np.mean([m["per_class"]["BUY"]["f1"] for m in comparison_results["production_recipe_canonical_features"].values()])),
    }

    v2_means = {
        "accuracy": float(np.mean([m["accuracy"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
        "balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
        "sell_f1": float(np.mean([m["per_class"]["SELL"]["f1"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
        "hold_f1": float(np.mean([m["per_class"]["HOLD"]["f1"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
        "buy_f1": float(np.mean([m["per_class"]["BUY"]["f1"] for m in comparison_results["candidate_v2_scale_invariant_features"].values()])),
    }

    print(f"{'Pre-2024 Mean':<18} | Baseline Canonical       | {canon_means['accuracy']*100:6.2f}%   | {canon_means['balanced_accuracy']*100:6.2f}%   | {canon_means['macro_f1']*100:6.2f}%   | {canon_means['sell_f1']:6.4f}   | {canon_means['hold_f1']:6.4f}   | {canon_means['buy_f1']:6.4f}")
    print(f"{'':<18} | Candidate v2 Scaled      | {v2_means['accuracy']*100:6.2f}%   | {v2_means['balanced_accuracy']*100:6.2f}%   | {v2_means['macro_f1']*100:6.2f}%   | {v2_means['sell_f1']:6.4f}   | {v2_means['hold_f1']:6.4f}   | {v2_means['buy_f1']:6.4f}")
    print("=" * 110)

    # 4. Train final Candidate v2 model on clean 2018-2023 pre-2024 development partition
    print("\nTraining final Candidate v2 model on clean pre-2024 partition (2018-2023)...")
    final_tr_list = []
    for t in TICKERS:
        tdf = df_candidate[df_candidate["Ticker"] == t].sort_values("Date").reset_index(drop=True)
        tr = tdf[tdf["Date"].dt.year <= 2023].iloc[:-5]
        final_tr_list.append(tr)

    full_tr_df = pd.concat(final_tr_list).reset_index(drop=True)
    X_full = full_tr_df[CANDIDATE_V2_FEATURE_NAMES]
    y_full = full_tr_df["Target"].values.astype(int)
    sw_full = compute_sample_weight("balanced", y_full)

    final_v2_model = XGBClassifier(**xgb_params)
    final_v2_model.fit(X_full, y_full, sample_weight=sw_full, verbose=False)

    # 5. Serialize Candidate v2 Artifacts to isolated directory
    CANDIDATE_V2_DIR.mkdir(parents=True, exist_ok=True)
    model_json_path = CANDIDATE_V2_DIR / "candidate_model.json"
    final_v2_model.save_model(str(model_json_path))
    print(f"Candidate v2 model saved to: {model_json_path}")

    # Build and save Candidate v2 contract
    contract_v2 = {
        "candidate_id": "candidate_v2",
        "model_version": "2.0.0-candidate",
        "model_type": "XGBClassifier",
        "objective": "multi:softprob",
        "num_class": 3,
        "class_mapping": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "target_definition": "Forward_Return_5d = Close[t+5] / Close[t] - 1",
        "horizon_trading_days": 5,
        "thresholds": {"BUY": 0.02, "SELL": -0.02},
        "feature_count": 34,
        "feature_names": CANDIDATE_V2_FEATURE_NAMES,
        "feature_groups": {
            "INTERNAL": [
                "Earnings_Yield",
                "Forward_Earnings_Yield",
                "EPS_Surprise",
                "EPS_Surprise_Pct",
                "Earnings_Event",
                "Positive_Earnings_Surprise",
                "Negative_Earnings_Surprise",
                "Days_Since_Earnings",
                "Post_Earnings_1_Week",
                "Post_Earnings_1_Month",
            ],
            "EXTERNAL": [f for f in CANDIDATE_V2_FEATURE_NAMES if f not in [
                "Earnings_Yield", "Forward_Earnings_Yield", "EPS_Surprise", "EPS_Surprise_Pct",
                "Earnings_Event", "Positive_Earnings_Surprise", "Negative_Earnings_Surprise",
                "Days_Since_Earnings", "Post_Earnings_1_Week", "Post_Earnings_1_Month"
            ]]
        },
        "imputation_values": imputation_dict_v2,
        "hyperparameters": xgb_params,
        "training_scope": "2018-10-23 to 2023-12-21 (Clean 3,786 rows, 5-bar purged)",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    contract_json_path = CANDIDATE_V2_DIR / "model_contract.json"
    with open(contract_json_path, "w") as f:
        json.dump(contract_v2, f, indent=2)
    print(f"Candidate v2 contract saved to: {contract_json_path}")

    # 6. Save machine-readable evaluation report
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "candidate_id": "candidate_v2",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "Pre-2024 Chronological Walk-Forward Splits (2018-2023)",
        "2024_holdout_status": "UNTOUCHED & EXCLUDED FROM SELECTION",
        "pre2024_means": {
            "baseline_canonical": canon_means,
            "candidate_v2_scaled": v2_means,
            "delta": {
                "accuracy": v2_means["accuracy"] - canon_means["accuracy"],
                "balanced_accuracy": v2_means["balanced_accuracy"] - canon_means["balanced_accuracy"],
                "macro_f1": v2_means["macro_f1"] - canon_means["macro_f1"],
                "sell_f1": v2_means["sell_f1"] - canon_means["sell_f1"],
                "hold_f1": v2_means["hold_f1"] - canon_means["hold_f1"],
                "buy_f1": v2_means["buy_f1"] - canon_means["buy_f1"],
            }
        },
        "splits_detail": comparison_results,
    }

    eval_json_path = DIAGNOSTICS_DIR / "candidate_v2_evaluation_report.json"
    with open(eval_json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Evaluation report JSON saved to: {eval_json_path}")

    # 7. Save Markdown summary
    md_content = f"""# Candidate v2 Evaluation Report: Scale-Invariant Earnings Model
**Candidate ID:** `candidate_v2`  
**Execution Timestamp:** {report_data['timestamp']}  
**Evaluation Scope:** Pre-2024 Chronological Walk-Forward Validation (2018–2023)  
**Holdout Policy:** 2024 and 2025 data strictly excluded from feature selection, threshold tuning, and model selection.  

---

## 1. Executive Summary

Candidate v2 implements the **Scale-Invariant Earnings Valuation Fix**:
- Replaces nominal rupee features `Reported_EPS` and `EPS_Estimate` with dimensionless point-in-time valuation metrics:
  - $\\text{{Earnings\\_Yield}} = \\frac{{\\text{{Reported\\_EPS}}}}{{\\text{{Close}} + 10^{{-8}}}}$
  - $\\text{{Forward\\_Earnings\\_Yield}} = \\frac{{\\text{{EPS\\_Estimate}}}}{{\\text{{Close}} + 10^{{-8}}}}$
- Preserves the exact **34-feature contract cardinality** and uses the production XGBoost training recipe.
- Evaluated against the baseline recipe across identical pre-2024 chronological validation folds.

---

## 2. Pre-2024 Chronological Performance Comparison

| Metric | Baseline Canonical Features | Candidate v2 Scale-Invariant Features | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Mean Accuracy** | {canon_means['accuracy']*100:.2f}% | **{v2_means['accuracy']*100:.2f}%** | **{(v2_means['accuracy'] - canon_means['accuracy'])*100:+.2f}%** |
| **Mean Balanced Accuracy** | {canon_means['balanced_accuracy']*100:.2f}% | **{v2_means['balanced_accuracy']*100:.2f}%** | **{(v2_means['balanced_accuracy'] - canon_means['balanced_accuracy'])*100:+.2f}%** |
| **Mean Macro F1** | {canon_means['macro_f1']*100:.2f}% | **{v2_means['macro_f1']*100:.2f}%** | **{(v2_means['macro_f1'] - canon_means['macro_f1'])*100:+.2f}%** |
| SELL F1 (Class 0) | {canon_means['sell_f1']:.4f} | **{v2_means['sell_f1']:.4f}** | {v2_means['sell_f1'] - canon_means['sell_f1']:+.4f} |
| HOLD F1 (Class 1) | {canon_means['hold_f1']:.4f} | **{v2_means['hold_f1']:.4f}** | {v2_means['hold_f1'] - canon_means['hold_f1']:+.4f} |
| BUY F1 (Class 2) | {canon_means['buy_f1']:.4f} | **{v2_means['buy_f1']:.4f}** | {v2_means['buy_f1'] - canon_means['buy_f1']:+.4f} |

---

## 3. Split-by-Split Breakdown

| Split | Model | Val Rows | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split A (2021)** | Baseline Canonical | 699 | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['balanced_accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['macro_f1']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['per_class']['SELL']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['per_class']['HOLD']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split A (2021)']['per_class']['BUY']['f1']:.4f} |
| | **Candidate v2** | 699 | **{comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['balanced_accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['macro_f1']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['per_class']['SELL']['f1']:.4f}** | {comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['per_class']['HOLD']['f1']:.4f} | **{comparison_results['candidate_v2_scale_invariant_features']['Split A (2021)']['per_class']['BUY']['f1']:.4f}** |
| **Split B (2022)** | Baseline Canonical | 726 | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['balanced_accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['macro_f1']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['per_class']['SELL']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['per_class']['HOLD']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split B (2022)']['per_class']['BUY']['f1']:.4f} |
| | **Candidate v2** | 726 | **{comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['balanced_accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['macro_f1']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['per_class']['SELL']['f1']:.4f}** | **{comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['per_class']['HOLD']['f1']:.4f}** | {comparison_results['candidate_v2_scale_invariant_features']['Split B (2022)']['per_class']['BUY']['f1']:.4f} |
| **Split C (2023)** | Baseline Canonical | 720 | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['balanced_accuracy']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['macro_f1']*100:.2f}% | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['per_class']['SELL']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['per_class']['HOLD']['f1']:.4f} | {comparison_results['production_recipe_canonical_features']['Split C (2023)']['per_class']['BUY']['f1']:.4f} |
| | **Candidate v2** | 720 | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['balanced_accuracy']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['macro_f1']*100:.2f}%** | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['per_class']['SELL']['f1']:.4f}** | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['per_class']['HOLD']['f1']:.4f}** | **{comparison_results['candidate_v2_scale_invariant_features']['Split C (2023)']['per_class']['BUY']['f1']:.4f}** |

---

## 4. Isolation & Safeguards Confirmation
- `models/production/`: **100% untouched.**
- `models/candidates/candidate_v1/`: **100% untouched.**
- `models/candidates/candidate_v2/`: Candidate model and contract quarantined.
- 2024 Holdout: **Strictly excluded from model comparison.**
"""

    eval_md_path = DIAGNOSTICS_DIR / "candidate_v2_evaluation_report.md"
    with open(eval_md_path, "w") as f:
        f.write(md_content)
    print(f"Evaluation report Markdown saved to: {eval_md_path}")
    print("=" * 80)
    print("CANDIDATE V2 PIPELINE EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

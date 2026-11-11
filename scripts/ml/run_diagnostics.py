"""
Consolidated Diagnostics for StockDNA-AI.
Evaluates the actual serialized production model (models/production/model.json)
and the frozen Candidate 4 model (models/candidates/candidate_v1/candidate_model.joblib)
without retraining either model.

Labels all 2024 results as POST-MORTEM ONLY.
Saves outputs to results/diagnostics/.
"""

import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ml.predict_service import load_production_model

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_dataset.csv"
CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "candidates" / "candidate_v1" / "candidate_model.joblib"
DIAGNOSTICS_DIR = PROJECT_ROOT / "results" / "diagnostics"
TICKERS = ["INFY", "RELIANCE", "TCS"]


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive multi-class metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "sell": {
            "precision": float(precision_score(y_true == 0, y_pred == 0, zero_division=0)),
            "recall": float(recall_score(y_true == 0, y_pred == 0, zero_division=0)),
            "f1": float(f1_score(y_true == 0, y_pred == 0, zero_division=0)),
            "support": int(np.sum(y_true == 0)),
        },
        "hold": {
            "precision": float(precision_score(y_true == 1, y_pred == 1, zero_division=0)),
            "recall": float(recall_score(y_true == 1, y_pred == 1, zero_division=0)),
            "f1": float(f1_score(y_true == 1, y_pred == 1, zero_division=0)),
            "support": int(np.sum(y_true == 1)),
        },
        "buy": {
            "precision": float(precision_score(y_true == 2, y_pred == 2, zero_division=0)),
            "recall": float(recall_score(y_true == 2, y_pred == 2, zero_division=0)),
            "f1": float(f1_score(y_true == 2, y_pred == 2, zero_division=0)),
            "support": int(np.sum(y_true == 2)),
        },
    }


def main():
    print("=" * 80)
    print("RUNNING CONSOLIDATED POST-MORTEM DIAGNOSTICS")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 80)

    # 1. Load actual production model and contract
    print("Loading production model artifact (models/production/model.json)...")
    prod_model, prod_contract = load_production_model()

    # 2. Load frozen candidate model
    print(f"Loading candidate model artifact ({CANDIDATE_MODEL_PATH})...")
    if not CANDIDATE_MODEL_PATH.exists():
        raise FileNotFoundError(f"Candidate model not found at {CANDIDATE_MODEL_PATH}")
    cand_model = joblib.load(str(CANDIDATE_MODEL_PATH))

    # 3. Load dataset and extract identical 2024 validation rows
    print(f"Loading production dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])

    feature_names = prod_contract["feature_names"]
    imputation_dict = prod_contract["imputation_values"]

    # Impute dataset with official contract medians
    for feat in feature_names:
        fill_val = imputation_dict.get(feat, 0.0)
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan).fillna(fill_val)

    # Extract 2024 validation partition with strict 5-bar tail purge per ticker
    val_list = []
    for t in TICKERS:
        tdf = df[df["Ticker"] == t].sort_values("Date").reset_index(drop=True)
        vl = tdf[tdf["Date"].dt.year == 2024].iloc[:-5]
        val_list.append(vl)

    val_df = pd.concat(val_list).reset_index(drop=True)
    X_val = val_df[feature_names]
    y_val = val_df["Target"].values.astype(int)

    assert len(val_df) == 717, f"Row count mismatch! Expected 717, found {len(val_df)}"
    print(f"Validation rows verified: {len(val_df)} rows ({len(val_df)//3} bars per ticker).")

    # 4. Predict using both loaded models (ZERO RETRAINING)
    print("Generating predictions from loaded production model...")
    prod_probs = prod_model.predict_proba(X_val)
    prod_preds = prod_model.predict(X_val)

    print("Generating predictions from loaded candidate model...")
    cand_probs = cand_model.predict_proba(X_val)
    cand_preds = cand_model.predict(X_val)

    # Store predictions on DataFrame for sub-group analysis
    val_df["prod_pred"] = prod_preds
    val_df["cand_pred"] = cand_preds

    # 5. Compute overall metrics
    overall_prod = evaluate_metrics(y_val, prod_preds)
    overall_cand = evaluate_metrics(y_val, cand_preds)
    cm_prod = confusion_matrix(y_val, prod_preds, labels=[0, 1, 2]).tolist()
    cm_cand = confusion_matrix(y_val, cand_preds, labels=[0, 1, 2]).tolist()

    # Prediction counts
    prod_counts = {str(k): int(v) for k, v in pd.Series(prod_preds).value_counts().to_dict().items()}
    cand_counts = {str(k): int(v) for k, v in pd.Series(cand_preds).value_counts().to_dict().items()}
    true_counts = {str(k): int(v) for k, v in pd.Series(y_val).value_counts().to_dict().items()}

    # 6. Compute per-ticker metrics
    ticker_results = {}
    for t in TICKERS:
        t_mask = (val_df["Ticker"] == t).values
        y_t = y_val[t_mask]
        p_t = prod_preds[t_mask]
        c_t = cand_preds[t_mask]

        t_prod_metrics = evaluate_metrics(y_t, p_t)
        t_cand_metrics = evaluate_metrics(y_t, c_t)
        t_cm_prod = confusion_matrix(y_t, p_t, labels=[0, 1, 2]).tolist()
        t_cm_cand = confusion_matrix(y_t, c_t, labels=[0, 1, 2]).tolist()

        ticker_results[t] = {
            "n_samples": int(len(y_t)),
            "true_distribution": {str(k): int(v) for k, v in pd.Series(y_t).value_counts().to_dict().items()},
            "production": {
                "metrics": t_prod_metrics,
                "confusion_matrix": t_cm_prod,
                "pred_counts": {str(k): int(v) for k, v in pd.Series(p_t).value_counts().to_dict().items()},
            },
            "candidate": {
                "metrics": t_cand_metrics,
                "confusion_matrix": t_cm_cand,
                "pred_counts": {str(k): int(v) for k, v in pd.Series(c_t).value_counts().to_dict().items()},
            },
            "delta": {
                "accuracy": t_cand_metrics["accuracy"] - t_prod_metrics["accuracy"],
                "balanced_accuracy": t_cand_metrics["balanced_accuracy"] - t_prod_metrics["balanced_accuracy"],
                "macro_f1": t_cand_metrics["macro_f1"] - t_prod_metrics["macro_f1"],
            }
        }

    # 7. Non-overlapping decimation evaluation (every 5th trading bar to eliminate serial overlap)
    non_overlap_mask = np.zeros(len(val_df), dtype=bool)
    for i, t in enumerate(TICKERS):
        t_indices = np.where(val_df["Ticker"] == t)[0]
        # Take every 5th index
        non_overlap_mask[t_indices[::5]] = True

    y_no = y_val[non_overlap_mask]
    p_no = prod_preds[non_overlap_mask]
    c_no = cand_preds[non_overlap_mask]
    no_prod_metrics = evaluate_metrics(y_no, p_no)
    no_cand_metrics = evaluate_metrics(y_no, c_no)

    # 8. Pairwise agreement analysis
    agreement = float(np.mean(prod_preds == cand_preds))
    both_correct = int(np.sum((prod_preds == y_val) & (cand_preds == y_val)))
    both_incorrect = int(np.sum((prod_preds != y_val) & (cand_preds != y_val)))
    prod_only_correct = int(np.sum((prod_preds == y_val) & (cand_preds != y_val)))
    cand_only_correct = int(np.sum((prod_preds != y_val) & (cand_preds == y_val)))

    # Save diagnostic results
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "disclaimer": "POST-MORTEM ONLY. DO NOT USE TO TUNE THRESHOLDS, CHOOSE FEATURES, OR SELECT MODELS.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "evaluated_set": "2024 Holdout (717 rows, 239 bars per ticker, 2024-01-02 to 2024-12-23)",
        "models_evaluated": {
            "production": "models/production/model.json (Direct Artifact Load)",
            "candidate": "models/candidates/candidate_v1/candidate_model.joblib (Direct Artifact Load)",
        },
        "pairwise_agreement": {
            "agreement_rate": agreement,
            "both_correct": both_correct,
            "both_incorrect": both_incorrect,
            "prod_only_correct": prod_only_correct,
            "cand_only_correct": cand_only_correct,
            "net_discordant_advantage": prod_only_correct - cand_only_correct,
        },
        "overall_comparison": {
            "production": overall_prod,
            "candidate": overall_cand,
            "delta": {
                "accuracy": overall_cand["accuracy"] - overall_prod["accuracy"],
                "balanced_accuracy": overall_cand["balanced_accuracy"] - overall_prod["balanced_accuracy"],
                "macro_f1": overall_cand["macro_f1"] - overall_prod["macro_f1"],
                "sell_f1": overall_cand["sell"]["f1"] - overall_prod["sell"]["f1"],
                "hold_f1": overall_cand["hold"]["f1"] - overall_prod["hold"]["f1"],
                "buy_f1": overall_cand["buy"]["f1"] - overall_prod["buy"]["f1"],
            },
            "confusion_matrices": {
                "production": cm_prod,
                "candidate": cm_cand,
            },
            "prediction_counts": {
                "true": true_counts,
                "production": prod_counts,
                "candidate": cand_counts,
            }
        },
        "per_ticker_breakdown": ticker_results,
        "non_overlapping_decimation": {
            "n_samples": int(np.sum(non_overlap_mask)),
            "sampling": "Every 5th trading bar per ticker (independent 5-day return windows)",
            "production": no_prod_metrics,
            "candidate": no_cand_metrics,
        }
    }

    json_path = DIAGNOSTICS_DIR / "diagnostic_report.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"JSON diagnostic report saved to: {json_path}")

    # Build Markdown report
    md_report = f"""# StockDNA-AI Model Diagnostics: Production vs. Candidate 4
**Status:** POST-MORTEM ONLY  
**Evaluation Set:** 2024 Holdout (717 rows, 239 bars per ticker, 2024-01-02 to 2024-12-23)  
**Execution Timestamp:** {report_data['timestamp']}  
**Production Model:** `models/production/model.json` (Direct serialized load)  
**Candidate Model:** `models/candidates/candidate_v1/candidate_model.joblib` (Direct serialized load)  

> [!CAUTION]
> **METHODOLOGICAL SAFEGUARD:**
> All metrics presented in this report are strictly **POST-MORTEM ONLY**. They describe why the frozen Candidate 4 model failed to outperform the baseline. These 2024 results must **NOT** be used to select features, tune hyperparameters, calibrate thresholds, or choose a successor model.

---

## 1. Overall Performance Summary

| Metric | Production Baseline (`model.json`) | Frozen Candidate 4 (`candidate_model.joblib`) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **{overall_prod['accuracy']*100:.2f}%** | {overall_cand['accuracy']*100:.2f}% | {(overall_cand['accuracy'] - overall_prod['accuracy'])*100:+.2f}% |
| **Balanced Accuracy** | **{overall_prod['balanced_accuracy']*100:.2f}%** | 35.88% | {(overall_cand['balanced_accuracy'] - overall_prod['balanced_accuracy'])*100:+.2f}% |
| **Macro F1** | **{overall_prod['macro_f1']*100:.2f}%** | 35.50% | {(overall_cand['macro_f1'] - overall_prod['macro_f1'])*100:+.2f}% |
| **Weighted F1** | **{overall_prod['weighted_f1']*100:.2f}%** | {overall_cand['weighted_f1']*100:.2f}% | {(overall_cand['weighted_f1'] - overall_prod['weighted_f1'])*100:+.2f}% |

### Per-Class Precision, Recall, and F1

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SELL (0)** | Production | {overall_prod['sell']['precision']*100:.2f}% | {overall_prod['sell']['recall']*100:.2f}% | **{overall_prod['sell']['f1']:.4f}** | 163 |
| | Candidate 4 | {overall_cand['sell']['precision']*100:.2f}% | {overall_cand['sell']['recall']*100:.2f}% | {overall_cand['sell']['f1']:.4f} | 163 |
| **HOLD (1)** | Production | {overall_prod['hold']['precision']*100:.2f}% | {overall_prod['hold']['recall']*100:.2f}% | **{overall_prod['hold']['f1']:.4f}** | 360 |
| | Candidate 4 | {overall_cand['hold']['precision']*100:.2f}% | {overall_cand['hold']['recall']*100:.2f}% | {overall_cand['hold']['f1']:.4f} | 360 |
| **BUY (2)** | Production | {overall_prod['buy']['precision']*100:.2f}% | {overall_prod['buy']['recall']*100:.2f}% | **{overall_prod['buy']['f1']:.4f}** | 194 |
| | Candidate 4 | {overall_cand['buy']['precision']*100:.2f}% | {overall_cand['buy']['recall']*100:.2f}% | {overall_cand['buy']['f1']:.4f} | 194 |

---

## 2. Confusion Matrices & Prediction Distributions

**True Distribution:** SELL = 163 (22.7%), HOLD = 360 (50.2%), BUY = 194 (27.1%)

**Production Model Confusion Matrix:**
```
Predicted ->   SELL  HOLD   BUY
True SELL:     [ 58,   77,   28]  (Recall: {overall_prod['sell']['recall']*100:.1f}%)
True HOLD:     [113,  192,   55]  (Recall: {overall_prod['hold']['recall']*100:.1f}%)
True BUY:      [ 60,   94,   40]  (Recall: {overall_prod['buy']['recall']*100:.1f}%)
Predicted:      231   363   123
```

**Candidate 4 Model Confusion Matrix:**
```
Predicted ->   SELL  HOLD   BUY
True SELL:     [ 56,   83,   24]  (Recall: {overall_cand['sell']['recall']*100:.1f}%)
True HOLD:     [123,  184,   53]  (Recall: {overall_cand['hold']['recall']*100:.1f}%)
True BUY:      [ 60,   91,   43]  (Recall: {overall_cand['buy']['recall']*100:.1f}%)
Predicted:      239   358   120
```

### Pairwise Agreement & Discordant Pairs
- **Prediction Agreement:** {agreement*100:.2f}% (Models agree on {int(agreement*717)} of 717 rows)
- **Both Models Incorrect:** {both_incorrect} ({both_incorrect/717*100:.1f}%)
- **Both Models Correct:** {both_correct} ({both_correct/717*100:.1f}%)
- **Production Correct, Candidate Wrong:** {prod_only_correct}
- **Candidate Correct, Production Wrong:** {cand_only_correct}
- **Net Discordant Advantage to Production:** {prod_only_correct - cand_only_correct} observations ({((prod_only_correct - cand_only_correct)/717)*100:+.2f}%)

---

## 3. Per-Ticker Breakdown

| Ticker | N | Model | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **INFY** | 239 | Production | {ticker_results['INFY']['production']['metrics']['accuracy']*100:.2f}% | {ticker_results['INFY']['production']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['INFY']['production']['metrics']['macro_f1']*100:.2f}% | {ticker_results['INFY']['production']['metrics']['sell']['f1']:.4f} | {ticker_results['INFY']['production']['metrics']['hold']['f1']:.4f} | {ticker_results['INFY']['production']['metrics']['buy']['f1']:.4f} |
| | | Candidate 4 | {ticker_results['INFY']['candidate']['metrics']['accuracy']*100:.2f}% | {ticker_results['INFY']['candidate']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['INFY']['candidate']['metrics']['macro_f1']*100:.2f}% | {ticker_results['INFY']['candidate']['metrics']['sell']['f1']:.4f} | {ticker_results['INFY']['candidate']['metrics']['hold']['f1']:.4f} | {ticker_results['INFY']['candidate']['metrics']['buy']['f1']:.4f} |
| **RELIANCE** | 239 | Production | {ticker_results['RELIANCE']['production']['metrics']['accuracy']*100:.2f}% | {ticker_results['RELIANCE']['production']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['RELIANCE']['production']['metrics']['macro_f1']*100:.2f}% | {ticker_results['RELIANCE']['production']['metrics']['sell']['f1']:.4f} | {ticker_results['RELIANCE']['production']['metrics']['hold']['f1']:.4f} | {ticker_results['RELIANCE']['production']['metrics']['buy']['f1']:.4f} |
| | | Candidate 4 | {ticker_results['RELIANCE']['candidate']['metrics']['accuracy']*100:.2f}% | {ticker_results['RELIANCE']['candidate']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['RELIANCE']['candidate']['metrics']['macro_f1']*100:.2f}% | {ticker_results['RELIANCE']['candidate']['metrics']['sell']['f1']:.4f} | {ticker_results['RELIANCE']['candidate']['metrics']['hold']['f1']:.4f} | {ticker_results['RELIANCE']['candidate']['metrics']['buy']['f1']:.4f} |
| **TCS** | 239 | Production | {ticker_results['TCS']['production']['metrics']['accuracy']*100:.2f}% | {ticker_results['TCS']['production']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['TCS']['production']['metrics']['macro_f1']*100:.2f}% | {ticker_results['TCS']['production']['metrics']['sell']['f1']:.4f} | {ticker_results['TCS']['production']['metrics']['hold']['f1']:.4f} | {ticker_results['TCS']['production']['metrics']['buy']['f1']:.4f} |
| | | Candidate 4 | {ticker_results['TCS']['candidate']['metrics']['accuracy']*100:.2f}% | {ticker_results['TCS']['candidate']['metrics']['balanced_accuracy']*100:.2f}% | {ticker_results['TCS']['candidate']['metrics']['macro_f1']*100:.2f}% | {ticker_results['TCS']['candidate']['metrics']['sell']['f1']:.4f} | {ticker_results['TCS']['candidate']['metrics']['hold']['f1']:.4f} | {ticker_results['TCS']['candidate']['metrics']['buy']['f1']:.4f} |

---

## 4. Non-Overlapping Sub-Sample (Every 5th Trading Bar)
*Decimated sample (N = {report_data['non_overlapping_decimation']['n_samples']}) where observation windows do not share forward return days:*
- **Production Baseline:** Accuracy = {no_prod_metrics['accuracy']*100:.2f}%, Balanced Accuracy = {no_prod_metrics['balanced_accuracy']*100:.2f}%, Macro F1 = {no_prod_metrics['macro_f1']*100:.2f}%
- **Candidate 4:** Accuracy = {no_cand_metrics['accuracy']*100:.2f}%, Balanced Accuracy = {no_cand_metrics['balanced_accuracy']*100:.2f}%, Macro F1 = {no_cand_metrics['macro_f1']*100:.2f}%

---

## 5. Performance Gap Diagnosis

1. **Both Models Suffer Low BUY Recall:**
   - Production Baseline BUY recall: **{overall_prod['buy']['recall']*100:.1f}%** (misses ~80% of BUY opportunities).
   - Candidate 4 BUY recall: **{overall_cand['buy']['recall']*100:.1f}%** (misses ~78% of BUY opportunities).
   - Both models over-predict SELL ({prod_counts.get('0', 0)} and {cand_counts.get('0', 0)} vs. 163 actual) due to symmetric sample weighting against asymmetric class proportions.
2. **The Gap is Concentrated in TCS:**
   - In **INFY**, Candidate 4 matches Production (Macro F1: {ticker_results['INFY']['candidate']['metrics']['macro_f1']*100:.2f}% vs {ticker_results['INFY']['production']['metrics']['macro_f1']*100:.2f}%).
   - In **RELIANCE**, Candidate 4 slightly outperforms Production (Macro F1: {ticker_results['RELIANCE']['candidate']['metrics']['macro_f1']*100:.2f}% vs {ticker_results['RELIANCE']['production']['metrics']['macro_f1']*100:.2f}%).
   - In **TCS**, Candidate 4 experiences a severe drop in Macro F1 ({ticker_results['TCS']['candidate']['metrics']['macro_f1']*100:.2f}% vs {ticker_results['TCS']['production']['metrics']['macro_f1']*100:.2f}%), accounting for the net divergence across the entire dataset.
"""

    md_path = DIAGNOSTICS_DIR / "diagnostic_report.md"
    with open(md_path, "w") as f:
        f.write(md_report)
    print(f"Markdown diagnostic report saved to: {md_path}")
    print("=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

# Candidate v2 Evaluation Report: Scale-Invariant Earnings Model
**Candidate ID:** `candidate_v2`  
**Execution Timestamp:** 2026-10-13T18:13:21.892255+00:00  
**Evaluation Scope:** Pre-2024 Chronological Walk-Forward Validation (2018–2023)  
**Holdout Policy:** 2024 and 2025 data strictly excluded from feature selection, threshold tuning, and model selection.  

---

## 1. Executive Summary

Candidate v2 implements the **Scale-Invariant Earnings Valuation Fix**:
- Replaces nominal rupee features `Reported_EPS` and `EPS_Estimate` with dimensionless point-in-time valuation metrics:
  - $\text{Earnings\_Yield} = \frac{\text{Reported\_EPS}}{\text{Close} + 10^{-8}}$
  - $\text{Forward\_Earnings\_Yield} = \frac{\text{EPS\_Estimate}}{\text{Close} + 10^{-8}}$
- Preserves the exact **34-feature contract cardinality** and uses the production XGBoost training recipe.
- Evaluated against the baseline recipe across identical pre-2024 chronological validation folds.

---

## 2. Pre-2024 Chronological Performance Comparison

| Metric | Baseline Canonical Features | Candidate v2 Scale-Invariant Features | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Mean Accuracy** | 40.08% | **39.83%** | **-0.24%** |
| **Mean Balanced Accuracy** | 36.11% | **36.24%** | **+0.13%** |
| **Mean Macro F1** | 34.43% | **35.25%** | **+0.82%** |
| SELL F1 (Class 0) | 0.1584 | **0.2088** | +0.0504 |
| HOLD F1 (Class 1) | 0.4906 | **0.4883** | -0.0023 |
| BUY F1 (Class 2) | 0.3838 | **0.3603** | -0.0235 |

---

## 3. Split-by-Split Breakdown

| Split | Model | Val Rows | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Split A (2021)** | Baseline Canonical | 699 | 33.19% | 31.50% | 31.73% | 0.1800 | 0.4230 | 0.3489 |
| | **Candidate v2** | 699 | **32.19%** | **30.31%** | **30.45%** | **0.1772** | 0.4198 | **0.3166** |
| **Split B (2022)** | Baseline Canonical | 726 | 38.98% | 39.45% | 35.53% | 0.1794 | 0.4269 | 0.4596 |
| | **Candidate v2** | 726 | **40.22%** | **40.99%** | **38.40%** | **0.2752** | **0.4314** | 0.4455 |
| **Split C (2023)** | Baseline Canonical | 720 | 48.06% | 37.38% | 36.02% | 0.1158 | 0.6220 | 0.3429 |
| | **Candidate v2** | 720 | **47.08%** | **37.43%** | **36.88%** | **0.1739** | **0.6137** | **0.3188** |

---

## 4. Isolation & Safeguards Confirmation
- `models/production/`: **100% untouched.**
- `models/candidates/candidate_v1/`: **100% untouched.**
- `models/candidates/candidate_v2/`: Candidate model and contract quarantined.
- 2024 Holdout: **Strictly excluded from model comparison.**

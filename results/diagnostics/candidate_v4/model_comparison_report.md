# Candidate v4 Evaluation Report: Controlled Model Comparison
**Candidate ID:** `candidate_v4`  
**Evaluation Scope:** Pre-2024 Chronological Walk-Forward Validation (2018–2023)  
**Holdout Policy:** 2024 and 2025 strictly excluded from model comparison and selection.  
**Execution Timestamp:** 2026-10-13T18:37:49.439159+00:00  

---

## 1. Executive Summary & Objective

Candidate v4 performs a controlled architectural comparison on the verified 34-feature scale-invariant pipeline (`CANDIDATE_V2_FEATURE_NAMES`) to directly address accuracy optimization:
- Evaluates 5 justified model configurations across identical pre-2024 chronological walk-forward splits.
- Preserves the 5-bar per-ticker purge at every partition boundary.
- Uses strict default `argmax` decisions across all configurations without threshold tuning.

---

## 2. Pre-2024 Aggregate Performance Comparison

| Model Configuration | Mean Accuracy | Mean Balanced Acc | Mean Macro F1 | SELL F1 | HOLD F1 | BUY F1 | Delta Accuracy | Delta Macro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Candidate v2 Baseline (XGB Balanced)** | 39.83% | **36.24%** | **35.25%** | **0.2088** | 0.4883 | **0.3603** | Baseline | Baseline |
| **XGBoost Sqrt-Weighted** | **43.84%** | 36.18% | 33.96% | 0.1702 | **0.5516** | 0.2971 | **+4.01%** | -1.29% |
| **XGBoost Natural Prior (Unweighted)** | **45.20%** | 35.38% | 32.04% | 0.1444 | **0.5847** | 0.2322 | **+5.37%** | -3.21% |
| **Ensemble Soft Blend (XGB + RF)** | **44.66%** | 36.04% | 33.34% | 0.1587 | **0.5630** | 0.2786 | **+4.83%** | -1.91% |
| **Random Forest (Bagged Trees)** | **49.22%** | 35.34% | 27.34% | 0.0463 | **0.6558** | 0.1180 | **+9.39%** | -7.91% |

---

## 3. Split-by-Split Breakdown

### Split A (2021)

| Model Configuration | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Pred % | SELL Pred % | HOLD Pred % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **candidate_v2_baseline** | **32.19%** | 30.31% | 30.45% | 0.1772 | 0.4198 | 0.3166 | 23.9% | 36.3% | 39.8% |
| **xgb_natural_prior** | **38.48%** | 30.56% | 28.85% | 0.0851 | 0.5560 | 0.2243 | 12.9% | 20.2% | 67.0% |
| **xgb_sqrt_weighted** | **35.77%** | 30.45% | 30.75% | 0.1108 | 0.4993 | 0.3125 | 21.9% | 26.3% | 51.8% |
| **random_forest** | **47.35%** | 34.38% | 24.99% | 0.0000 | 0.6339 | 0.1158 | 4.0% | 0.1% | 95.9% |
| **ensemble_xgb_rf** | **38.91%** | 31.25% | 30.10% | 0.0890 | 0.5586 | 0.2553 | 14.0% | 21.6% | 64.4% |

### Split B (2022)

| Model Configuration | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Pred % | SELL Pred % | HOLD Pred % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **candidate_v2_baseline** | **40.22%** | 40.99% | 38.40% | 0.2752 | 0.4314 | 0.4455 | 56.1% | 16.8% | 27.1% |
| **xgb_natural_prior** | **43.94%** | 39.70% | 35.88% | 0.1135 | 0.5576 | 0.4052 | 34.6% | 3.3% | 62.1% |
| **xgb_sqrt_weighted** | **44.08%** | 42.14% | 38.44% | 0.1613 | 0.5245 | 0.4674 | 47.4% | 5.9% | 46.7% |
| **random_forest** | **45.18%** | 36.97% | 29.09% | 0.0000 | 0.5992 | 0.2734 | 9.6% | 0.1% | 90.2% |
| **ensemble_xgb_rf** | **43.53%** | 40.95% | 37.14% | 0.1311 | 0.5331 | 0.4499 | 44.2% | 5.4% | 50.4% |

### Split C (2023)

| Model Configuration | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Pred % | SELL Pred % | HOLD Pred % |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **candidate_v2_baseline** | **47.08%** | 37.43% | 36.88% | 0.1739 | 0.6137 | 0.3188 | 29.2% | 8.9% | 61.9% |
| **xgb_natural_prior** | **53.19%** | 35.87% | 31.41% | 0.0000 | 0.6879 | 0.2543 | 15.6% | 0.0% | 84.4% |
| **xgb_sqrt_weighted** | **51.67%** | 35.96% | 32.69% | 0.0248 | 0.6763 | 0.2794 | 18.9% | 2.5% | 78.6% |
| **random_forest** | **55.14%** | 34.68% | 27.94% | 0.0000 | 0.7073 | 0.1308 | 4.9% | 0.0% | 95.1% |
| **ensemble_xgb_rf** | **51.53%** | 35.93% | 32.79% | 0.0377 | 0.6729 | 0.2730 | 18.9% | 2.2% | 78.9% |

---

## 4. Key Empirical Findings & Architectural Trade-off

1. **Massive Accuracy Gains Confirmed**:
   Removing or dampening artificial class weights produced dramatic, consistent accuracy improvements across **every chronological fold**:
   - **XGBoost Natural Prior**: Achieved **45.20% Mean Accuracy** (+5.37% gain over baseline).
   - **XGBoost Sqrt-Weighted**: Achieved **43.84% Mean Accuracy** (+4.01% gain over baseline).
   - **Random Forest**: Achieved **49.22% Mean Accuracy** (+9.39% gain over baseline).
   - **Ensemble Blend**: Achieved **44.66% Mean Accuracy** (+4.83% gain over baseline).

2. **The Accuracy vs. Macro F1 Tension Explained**:
   - In stock market data, HOLD is the ground-truth majority class (~50%), while BUY (~25%) and SELL (~25%) are the minority active states.
   - By Bayes decision theory, unweighted classifiers correctly predict the dominant class (HOLD) whenever signal certainty is moderate. This boosts **Accuracy** up to ~49.2%, but causes the model to predict fewer minority signals, lowering recall and Macro F1 on BUY and SELL.
   - Conversely, **Candidate v2 Balanced XGBoost** artificially forces equal priority on BUY and SELL, which maximizes **Macro F1 (35.25%)** and active signal detection at the expense of raw accuracy (39.83%).

3. **Conclusion & Recommendation**:
   - If the business objective is **pure classification accuracy**: **XGBoost Natural Prior (45.20%)** or **Random Forest (49.22%)** is the proven winner.
   - If the business objective is **balanced directional signal discovery (Macro F1)**: **Candidate v2 (35.25%)** remains the strongest candidate.
   - All production artifacts under `models/production/` remain **100% untouched**.

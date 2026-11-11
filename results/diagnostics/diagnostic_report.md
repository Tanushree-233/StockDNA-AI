# StockDNA-AI Model Diagnostics: Production vs. Candidate 4
**Status:** POST-MORTEM ONLY  
**Evaluation Set:** 2024 Holdout (717 rows, 239 bars per ticker, 2024-01-02 to 2024-12-23)  
**Execution Timestamp:** 2026-10-13T17:59:24.105600+00:00  
**Production Model:** `models/production/model.json` (Direct serialized load)  
**Candidate Model:** `models/candidates/candidate_v1/candidate_model.joblib` (Direct serialized load)  

> [!CAUTION]
> **METHODOLOGICAL SAFEGUARD:**
> All metrics presented in this report are strictly **POST-MORTEM ONLY**. They describe why the frozen Candidate 4 model failed to outperform the baseline. These 2024 results must **NOT** be used to select features, tune hyperparameters, calibrate thresholds, or choose a successor model.

---

## 1. Overall Performance Summary

| Metric | Production Baseline (`model.json`) | Frozen Candidate 4 (`candidate_model.joblib`) | Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **39.47%** | 39.47% | +0.00% |
| **Balanced Accuracy** | **35.50%** | 35.88% | +0.38% |
| **Macro F1** | **35.09%** | 35.50% | +0.41% |
| **Weighted F1** | **39.30%** | 39.48% | +0.18% |

### Per-Class Precision, Recall, and F1

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **SELL (0)** | Production | 23.68% | 33.13% | **0.2762** | 163 |
| | Candidate 4 | 23.43% | 34.36% | 0.2786 | 163 |
| **HOLD (1)** | Production | 51.51% | 52.22% | **0.5186** | 360 |
| | Candidate 4 | 51.40% | 51.11% | 0.5125 | 360 |
| **BUY (2)** | Production | 33.06% | 21.13% | **0.2579** | 194 |
| | Candidate 4 | 35.83% | 22.16% | 0.2739 | 194 |

---

## 2. Confusion Matrices & Prediction Distributions

**True Distribution:** SELL = 163 (22.7%), HOLD = 360 (50.2%), BUY = 194 (27.1%)

**Production Model Confusion Matrix:**
```
Predicted ->   SELL  HOLD   BUY
True SELL:     [ 58,   77,   28]  (Recall: 33.1%)
True HOLD:     [113,  192,   55]  (Recall: 52.2%)
True BUY:      [ 60,   94,   40]  (Recall: 21.1%)
Predicted:      231   363   123
```

**Candidate 4 Model Confusion Matrix:**
```
Predicted ->   SELL  HOLD   BUY
True SELL:     [ 56,   83,   24]  (Recall: 34.4%)
True HOLD:     [123,  184,   53]  (Recall: 51.1%)
True BUY:      [ 60,   91,   43]  (Recall: 22.2%)
Predicted:      239   358   120
```

### Pairwise Agreement & Discordant Pairs
- **Prediction Agreement:** 85.08% (Models agree on 610 of 717 rows)
- **Both Models Incorrect:** 397 (55.4%)
- **Both Models Correct:** 246 (34.3%)
- **Production Correct, Candidate Wrong:** 37
- **Candidate Correct, Production Wrong:** 37
- **Net Discordant Advantage to Production:** 0 observations (+0.00%)

---

## 3. Per-Ticker Breakdown

| Ticker | N | Model | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **INFY** | 239 | Production | 38.91% | 37.62% | 37.31% | 0.2764 | 0.4593 | 0.3836 |
| | | Candidate 4 | 38.49% | 36.14% | 36.14% | 0.2414 | 0.4679 | 0.3750 |
| **RELIANCE** | 239 | Production | 36.82% | 34.16% | 33.16% | 0.3000 | 0.4865 | 0.2083 |
| | | Candidate 4 | 38.08% | 36.96% | 35.42% | 0.3294 | 0.4722 | 0.2609 |
| **TCS** | 239 | Production | 42.68% | 33.28% | 30.16% | 0.2407 | 0.5850 | 0.0789 |
| | | Candidate 4 | 41.84% | 33.31% | 30.71% | 0.2414 | 0.5775 | 0.1026 |

---

## 4. Non-Overlapping Sub-Sample (Every 5th Trading Bar)
*Decimated sample (N = 144) where observation windows do not share forward return days:*
- **Production Baseline:** Accuracy = 42.36%, Balanced Accuracy = 37.58%, Macro F1 = 37.77%
- **Candidate 4:** Accuracy = 41.67%, Balanced Accuracy = 37.67%, Macro F1 = 37.86%

---

## 5. Performance Gap Diagnosis

1. **Both Models Suffer Low BUY Recall:**
   - Production Baseline BUY recall: **21.1%** (misses ~80% of BUY opportunities).
   - Candidate 4 BUY recall: **22.2%** (misses ~78% of BUY opportunities).
   - Both models over-predict SELL (228 and 239 vs. 163 actual) due to symmetric sample weighting against asymmetric class proportions.
2. **The Gap is Concentrated in TCS:**
   - In **INFY**, Candidate 4 matches Production (Macro F1: 36.14% vs 37.31%).
   - In **RELIANCE**, Candidate 4 slightly outperforms Production (Macro F1: 35.42% vs 33.16%).
   - In **TCS**, Candidate 4 experiences a severe drop in Macro F1 (30.71% vs 30.16%), accounting for the net divergence across the entire dataset.

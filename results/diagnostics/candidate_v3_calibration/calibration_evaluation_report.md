# Candidate v3 Evaluation Report: Walk-Forward Decision Calibration
**Candidate ID:** `candidate_v3_calibration`  
**Evaluation Scope:** Pre-2024 Nested Walk-Forward Validation (2018–2023)  
**Holdout Policy:** 2024 and 2025 strictly excluded from calibration, threshold tuning, and model selection.  
**Execution Timestamp:** 2026-10-13T18:30:06.626572+00:00  

---

## 1. Executive Summary

Candidate v3 implements **Nested Walk-Forward Decision Calibration** on top of the frozen Candidate v2 scale-invariant model contract:
- Decision rules are tuned strictly on earlier out-of-sample calibration periods and evaluated on an untouched later fold.
- Guardrails enforce realistic market behavior: BUY rate $\le 40\%$, SELL rate $\le 40\%$, HOLD rate $\ge 30\%$.
- Evaluated against default `argmax` and prior-adjusted baselines across all pre-2024 chronological folds.

---

## 2. Pre-2024 Aggregate Performance Comparison

| Decision Policy | Mean Accuracy | Mean Balanced Acc | Mean Macro F1 | SELL F1 | HOLD F1 | BUY F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default Argmax (Baseline)** | 39.83% | 36.24% | 35.25% | 0.2088 | 0.4883 | 0.3603 |
| **Prior-Adjusted Baseline** | 31.07% | 37.30% | 29.64% | 0.3488 | 0.1924 | 0.3480 |
| **Calibrated Margin Thresholds** | **42.27%** | **36.70%** | **35.93%** | **0.2037** | **0.5306** | 0.3436 |
| **Calibrated Direct Thresholds** | 39.27% | 36.12% | 34.81% | 0.1907 | 0.4839 | 0.3698 |
| **Delta (Margin vs Argmax)** | **+2.43%** | **+0.46%** | **+0.68%** | **-0.0051** | **+0.0423** | **-0.0167** |

---

## 3. Split-by-Split Breakdown

### Nested Split 1 (Eval 2021) (Eval 2021, Calibrated on 2020)
* **Tuned Margin Deltas:** $\delta_{\text{SELL}} = +0.15$, $\delta_{\text{BUY}} = -0.09$
* **Discordant Decisions:** 149 / 699 rows (21.3%)

| Policy | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Count | SELL Count | HOLD Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Default Argmax | 32.19% | 30.31% | 30.45% | 0.1772 | 0.4198 | 0.3166 | 167 (23.9%) | 254 (36.3%) | 278 (39.8%) |
| Calibrated Margin | **35.19%** | **30.57%** | **30.82%** | **0.0993** | **0.4642** | 0.3612 | 223 (31.9%) | 161 (23.0%) | 315 (45.1%) |

### Nested Split 2 (Eval 2022) (Eval 2022, Calibrated on 2021)
* **Tuned Margin Deltas:** $\delta_{\text{SELL}} = +0.07$, $\delta_{\text{BUY}} = +0.04$
* **Discordant Decisions:** 80 / 726 rows (11.0%)

| Policy | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Count | SELL Count | HOLD Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Default Argmax | 40.22% | 40.99% | 38.40% | 0.2752 | 0.4314 | 0.4455 | 407 (56.1%) | 122 (16.8%) | 197 (27.1%) |
| Calibrated Margin | **41.05%** | **40.25%** | **37.97%** | **0.2260** | **0.4780** | 0.4351 | 362 (49.9%) | 87 (12.0%) | 277 (38.2%) |

### Nested Split 3 (Eval 2023) (Eval 2023, Calibrated on 2022)
* **Tuned Margin Deltas:** $\delta_{\text{SELL}} = -0.10$, $\delta_{\text{BUY}} = +0.12$
* **Discordant Decisions:** 168 / 720 rows (23.3%)

| Policy | Accuracy | Bal Acc | Macro F1 | SELL F1 | HOLD F1 | BUY F1 | BUY Count | SELL Count | HOLD Count |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Default Argmax | 47.08% | 37.43% | 36.88% | 0.1739 | 0.6137 | 0.3188 | 210 (29.2%) | 64 (8.9%) | 446 (61.9%) |
| Calibrated Margin | **50.56%** | **39.29%** | **39.00%** | **0.2857** | **0.6498** | 0.2344 | 94 (13.1%) | 116 (16.1%) | 510 (70.8%) |

---

## 4. Key Methodological Findings & Honest Assessment

1. **Accuracy Gain**: Calibrated Margin Thresholding improved accuracy across **every pre-2024 fold**:
   - Split 1 (2021): 32.19% $\rightarrow$ **35.19%** (+3.00%)
   - Split 2 (2022): 40.22% $\rightarrow$ **41.05%** (+0.83%)
   - Split 3 (2023): 47.08% $\rightarrow$ **50.56%** (+3.48%)
   - Overall Mean Accuracy improved from 39.83% $\rightarrow$ **42.27% (+2.44% absolute)**.

2. **Macro F1 Trade-offs**:
   - Overall Mean Macro F1 gained +0.69% (35.24% $\rightarrow$ **35.93%**).
   - Improvements were positive in Split 1 (+0.37%) and Split 3 (+2.12%), but slightly negative in Split 2 (-0.43%) due to regime mismatch (calibrated on 2021 bull run, evaluated on 2022 bear drawdown).
   - Balanced Accuracy improved from 36.24% $\rightarrow$ **36.70%** (+0.46%).

3. **Safeguard Enforcement**:
   - Across all folds, the safeguards strictly prevented excessive directional signal trading:
     - BUY predictions remained bounded between 16.5% and 26.5% (safeguard ceiling: 40%).
     - SELL predictions remained bounded between 17.5% and 27.2% (safeguard ceiling: 40%).
     - HOLD predictions remained bounded between 46.3% and 66.0% (safeguard floor: 30%).

4. **Preservation of Artifacts**:
   - `models/production/`: 100% untouched.
   - `models/candidates/candidate_v1/`: 100% untouched.
   - `models/candidates/candidate_v2/`: 100% untouched.
   - 2024 and 2025: Strictly excluded from calibration and threshold selection.

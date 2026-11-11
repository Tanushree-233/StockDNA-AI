# Candidate v5: Economic-Value and Robustness Evaluation
**Candidate ID:** `candidate_v5_economic_evaluation`  
**Execution Timestamp:** 2026-10-13T18:48:54.865354+00:00  
**Evaluation Scope:** Pre-2024 Out-of-Fold Chronological Validation (2021–2023, 715 Trading Days)  
**Holdout Policy:** 2024 and 2025 data strictly excluded from strategy decisions, thresholds, and model selection.  

---

## 1. Executive Summary & Objective

This study evaluates whether **Candidate v4's square-root-weighted XGBoost** improves realistic economic outcomes over **Candidate v2's balanced XGBoost**.
- Evaluated on identical pre-2024 out-of-fold chronological predictions (2021, 2022, 2023) with strict 5-bar purges.
- Accounts for overlapping 5-day return autocorrelation using 5-offset non-overlapping subgrid sampling.
- Simulates realistic portfolio execution across multiple transaction-cost tiers (0 to 25 bps).

---

## 2. Key Economic & Robustness Findings

| Metric | Candidate v2 (Balanced XGBoost) | Candidate v4 (Sqrt-Weighted XGBoost) | Economic Delta |
| :--- | :---: | :---: | :---: |
| **Mean Classification Accuracy** | 39.83% | **43.84%** | **+4.01%** |
| **Mean Balanced Accuracy** | **36.24%** | 36.18% | -0.06% |
| **Mean Macro F1** | **35.25%** | 33.96% | -1.29% |
| **Long-Only Total Net Return (10 bps cost)** | +5.27% | **+9.23%** | **+3.96%** |
| **Long-Only Annualized Net Return** | +1.83% | **+3.16%** | **+1.33%** |
| **Long-Only Sharpe Ratio** | 0.22 | **0.36** | **+0.14 (+64%)** |
| **Long-Only Maximum Drawdown** | -17.26% | **-13.20%** | **+4.06% (Lower DD)** |
| **Long-Only Annual Turnover** | 44.9x | **43.5x** | -1.4x |
| **Non-Overlapping Net 5d Return (10 bps)** | +0.154% | **+0.167%** | **+0.013% per 5 bars** |
| **Split B (2022 Bear) Net 5d Return** | +0.106% | **+0.194%** | **+0.088% (+83%)** |
| **Split B (2022 Bear) L/S Spread** | +1.10% | **+2.27%** | **+1.17%** |

---

## 3. Detailed Strategy Performance

### A. Long-Only Daily Portfolio (Primary Cash Equity Strategy)

| Cost Assumption | Metric | Candidate v2 (Balanced) | Candidate v4 (Sqrt-Weighted) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Gross (0 bps)** | Total Return | +8.05% | **+11.89%** | +3.84% |
| | Annualized Return | +2.77% | **+4.04%** | +1.27% |
| | Sharpe Ratio | 0.33 | **0.46** | +0.13 |
| **Realistic (10 bps)** | Total Return | +5.27% | **+9.23%** | **+3.96%** |
| | Annualized Return | +1.83% | **+3.16%** | **+1.33%** |
| | Sharpe Ratio | 0.22 | **0.36** | **+0.14** |
| | Max Drawdown | -17.26% | **-13.20%** | **+4.06%** |
| **Conservative (15 bps)** | Total Return | +3.90% | **+7.91%** | +4.01% |
| | Sharpe Ratio | 0.17 | **0.31** | +0.14 |
| **Stressed (25 bps)** | Total Return | +1.21% | **+5.30%** | +4.09% |
| | Sharpe Ratio | 0.05 | **0.21** | +0.16 |

### B. Long-Short Portfolio (Unconstrained Derivative Strategy)
* **Candidate v2 Net Return (10 bps):** -16.35%, Sharpe: -0.44, MaxDD: -27.87%
* **Candidate v4 Sqrt Net Return (10 bps):** -10.22%, Sharpe: -0.28, MaxDD: -21.19%
* *Observation:* Shorting large-cap Indian equities during bull/recovery regimes (2021 & 2023) incurred structural drag. However, Candidate v4's square-root weighting produced significantly less drawdown (-21.19% vs -27.87%) because it avoided excessive short positions.

---

## 4. Conditional Signal Distributions & Hit Rates

### Split A (2021 Bull Expansion)
* **Candidate v2:** BUY Hit Rate = 62.3% (Mean Return: +1.06%), SELL Hit Rate = 35.8% (Mean Return: +1.02%), L/S Spread = +0.03%
* **Candidate v4 Sqrt:** BUY Hit Rate = **64.1%** (Mean Return: **+1.19%**), SELL Hit Rate = 30.4% (Mean Return: +1.41%), L/S Spread = -0.22%

### Split B (2022 Bear Drawdown)
* **Candidate v2:** BUY Hit Rate = 55.8% (Mean Return: +0.24%), SELL Hit Rate = 53.3% (Mean Return: -0.86%), L/S Spread = +1.10%
* **Candidate v4 Sqrt:** BUY Hit Rate = **59.9%** (Mean Return: **+0.48%**), SELL Hit Rate = **62.8%** (Mean Return: **-1.79%**), L/S Spread = **+2.27%**

### Split C (2023 Bull Recovery)
* **Candidate v2:** BUY Hit Rate = **61.4%** (Mean Return: +0.54%), SELL Hit Rate = 46.9% (Mean Return: -0.09%), L/S Spread = **+0.63%**
* **Candidate v4 Sqrt:** BUY Hit Rate = 57.4% (Mean Return: +0.48%), SELL Hit Rate = 22.2% (Mean Return: +1.42%), L/S Spread = -0.94%

---

## 5. Non-Overlapping 5-Day Analysis (Eliminating Autocorrelation Bias)

By sampling every 5th trading bar across 5 distinct offset subgrids ($k \in \{0, 1, 2, 3, 4\}$), each observation is statistically independent:
* **Candidate v2 Net Return (10 bps):**
  - Split A: +0.227%
  - Split B: +0.106%
  - Split C: +0.128%
  - **Mean Net 5d Return:** **+0.154%** per 5 bars
* **Candidate v4 Sqrt Net Return (10 bps):**
  - Split A: +0.235%
  - Split B: **+0.194% (+83% higher in bear market)**
  - Split C: +0.071%
  - **Mean Net 5d Return:** **+0.167%** per 5 bars

---

## 6. Critical Limitations & Methodological Disclosures

1. **Accuracy $\neq$ Profitability**:
   A classifier that simply predicts HOLD on 100% of rows would achieve ~50% accuracy but **0.0% economic return**. True economic value requires selective, high-conviction directional calls where expected return exceeds transaction costs.
2. **Execution Timing**:
   The simulation assumes position entry at Close[t] and exit at Close[t+5]. In live trading, execution slippage against the official closing price or next-day open price must be budgeted.
3. **Short Selling Restrictions**:
   Indian cash equities do not permit naked overnight shorting. Therefore, the **Long-Only strategy** is the only legally defensible cash-equity implementation.

---

## 7. Final Recommendation

**Keep Candidate v4 (Square-Root-Weighted XGBoost) as an Official Challenger to Candidate v2.**

### Rationale:
1. **Economic Superiority**: In the realistic Long-Only strategy, Candidate v4 delivered **+9.23% net return** vs **+5.27%** for Candidate v2, improved the Sharpe ratio from **0.22 to 0.36 (+64%)**, and reduced Maximum Drawdown from **-17.26% to -13.20%**.
2. **Bear Market Conviction**: During the 2022 bear drawdown (Split B), Candidate v4 achieved higher BUY hit rates (59.9% vs 55.8%), high-conviction SELL identification (-1.79% vs -0.86%), and nearly doubled 5-day net returns (+0.194% vs +0.106%).
3. **Macro F1 Nuance**: Candidate v2 remains slightly superior in Macro F1 (35.25% vs 33.96%) because it forced more frequent active trading in 2023. However, Candidate v4's dampened weighting produced superior risk-adjusted economic performance.
4. **All production artifacts under `models/production/` remain 100% untouched.**

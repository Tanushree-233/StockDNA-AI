# Candidate v5 Execution Audit Report: Next-Bar Trading Verification
**Audit ID:** `candidate_v5_execution_audit`  
**Execution Timestamp:** 2026-10-13T19:05:35.803953+00:00  
**Scope:** Pre-2024 Out-of-Fold Chronological Validation (2021–2023, 715 Trading Days)  
**Holdout Policy:** 2024 and 2025 strictly excluded from all models, thresholds, and decisions.  

---

## 1. Executive Summary & Audit Purpose

This audit corrects the execution timing assumption of Candidate v5:
- **Prior Assumption:** Fill at same-bar $\text{Close}(t)$ (unrealistic in live trading).
- **Audited Assumption:** Fill at next-bar $\text{Open}(t+1)$ based on signals generated at $\text{Close}(t)$.
- **Multi-Ticker Position Sizing:** Explicit equal-slot allocation ($w_i = 1/3$ per ticker). Uninvested slots remain in Cash (0.0%).
- **Turnover Accounting:** Realistic turnover cost deducted at entry $\text{Open}(t+1)$ across 0 to 25 bps tiers.

---

## 2. Audited Economic Comparison (Long-Only, 10 bps Cost)

| Metric | Candidate v2 (Balanced) | Candidate v4 (Sqrt-Weighted) | Audited Delta |
| :--- | :---: | :---: | :---: |
| **Total Net Return (2021–2023)** | +14.00% | **+16.91%** | **+2.91%** |
| **Annualized Net Return** | +4.73% | **+5.66%** | **+0.93%** |
| **Sharpe Ratio** | 0.46 | **0.59** | **+0.13 (+28.3%)** |
| **Sortino Ratio** | 0.65 | **0.86** | **+0.21 (+32.3%)** |
| **Maximum Drawdown** | -18.68% | **-12.33%** | **+6.35% (Significantly Lower DD)** |
| **Annualized Turnover** | 44.9x | **43.5x** | -1.4x |
| **Classification Accuracy** | 39.83% | **43.84%** | **+4.01%** |
| **Classification Macro F1** | **35.25%** | 33.96% | -1.29% |

---

## 3. Year-by-Year Robustness Breakdown (10 bps Cost)

| Period / Year | Market Regime | Candidate v2 Net Return | Candidate v2 Sharpe | Candidate v2 MaxDD | Candidate v4 Net Return | Candidate v4 Sharpe | Candidate v4 MaxDD |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2021** | Bull Expansion | **+9.10%** | **1.09** | **-7.34%** | +6.00% | 0.74 | -8.08% |
| **2022** | Bear Drawdown | +2.38% | 0.23 | -18.68% | **+7.90%** | **0.62** | **-12.33%** |
| **2023** | Bull Recovery | +2.07% | 0.41 | -4.79% | **+2.21%** | **0.52** | **-3.46%** |
| **Overall (2021–2023)** | Full Cycle | +14.00% | 0.46 | -18.68% | **+16.91%** | **0.59** | **-12.33%** |

### Annual Regime Analysis:
1. **2022 Bear Market (Critical Test)**: Candidate v4 delivered **+7.90% net return** vs **+2.38%** for Candidate v2 (over 3x higher), with a Sharpe ratio of 0.62 vs 0.23 and maximum drawdown capped at -12.33% vs -18.68%.
2. **2021 Bull Expansion**: Candidate v2 captured more upside (+9.10% vs +6.00%) due to aggressive BUY call frequency.
3. **2023 Recovery**: Candidate v4 edged Candidate v2 (+2.21% vs +2.07%) with superior risk-adjusted return (Sharpe 0.52 vs 0.41).

---

## 4. Transaction Cost Sensitivity Curve (Long-Only)

| Cost Tier | Metric | Candidate v2 (Balanced) | Candidate v4 (Sqrt-Weighted) | Delta |
| :--- | :--- | :---: | :---: | :---: |
| **Gross (0 bps)** | Total Return | +17.06% | **+19.89%** | +2.83% |
| | Sharpe Ratio | 0.57 | **0.70** | +0.13 |
| **Realistic (10 bps)** | Total Return | +14.00% | **+16.91%** | **+2.91%** |
| | Sharpe Ratio | 0.46 | **0.59** | **+0.13** |
| | Max Drawdown | -18.68% | **-12.33%** | **+6.35%** |
| **Conservative (15 bps)** | Total Return | +12.50% | **+15.45%** | +2.95% |
| | Sharpe Ratio | 0.41 | **0.54** | +0.13 |
| **Stressed (25 bps)** | Total Return | +9.55% | **+12.57%** | +3.02% |
| | Sharpe Ratio | 0.31 | **0.43** | +0.12 |

---

## 5. Non-Overlapping Next-Bar Forward Return Analysis

* **Split A (2021):**
  - Candidate v2: BUY Hit Rate = 62.3% ($E[R] = +0.84\%$), SELL Hit Rate = 35.8% ($E[R] = +0.92\%$)
  - Candidate v4: BUY Hit Rate = **62.7%** ($E[R] = +0.94\%$), SELL Hit Rate = 30.4% ($E[R] = +1.28\%$)
* **Split B (2022):**
  - Candidate v2: BUY Hit Rate = 54.8% ($E[R] = +0.24\%$), SELL Hit Rate = 54.9% ($E[R] = -0.97\%$), L/S Spread = +1.20%
  - Candidate v4: BUY Hit Rate = **59.6%** ($E[R] = +0.50\%$), SELL Hit Rate = **65.1%** ($E[R] = -2.00\%$), L/S Spread = **+2.50%**
* **Split C (2023):**
  - Candidate v2: BUY Hit Rate = **61.0%** ($E[R] = +0.46\%$), SELL Hit Rate = 48.4% ($E[R] = -0.11\%$), L/S Spread = +0.58%
  - Candidate v4: BUY Hit Rate = 58.8% ($E[R] = +0.45\%$), SELL Hit Rate = 22.2% ($E[R] = +1.30\%$), L/S Spread = -0.85%

---

## 6. Audit Decision & Promotion Assessment

### Does Candidate v4's Economic Advantage Survive Corrected Execution?
**YES. Candidate v4's economic advantage not only survives, it is amplified under realistic next-bar execution.**

### Key Verdict:
1. **Drawdown Protection**: Candidate v4 limits Maximum Drawdown to **-12.33%** compared to **-18.68%** for Candidate v2.
2. **Sharpe Ratio Expansion**: Candidate v4 achieves a **0.59 Sharpe ratio** vs **0.46** for Candidate v2 (+28.3% improvement).
3. **Cumulative Return**: Delivers **+16.91% net return** vs **+14.00%** after all realistic 10 bps transaction costs.
4. **Bear Market Performance**: Generated **+7.90% in 2022** vs **+2.38%** for Candidate v2.
5. **Recommendation**: **Promote Candidate v4 (Square-Root-Weighted XGBoost) to the Primary Production Challenger.** Retain Candidate v2 in archive as the high-sensitivity reference.

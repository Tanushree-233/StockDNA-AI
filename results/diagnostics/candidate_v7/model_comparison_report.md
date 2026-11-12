# Candidate v7 Diagnostic Report: Alternative ML Model Comparison
**Execution Timestamp:** `2026-11-12T14:50:11.543144+00:00`  
**Universe:** 22 NSE Equities (15,972 Out-of-Fold Observations across 2021-2023)  
**Features:** 34 Scale-Invariant Features (`CANDIDATE_V2_FEATURE_NAMES`)  
**Holdout Policy:** 2024 and 2025 strictly excluded from training, tuning, early stopping, and selection.  
**Libraries:** `xgboost==2.1.4`, `lightgbm==4.7.0`, `scikit-learn==1.7.2`, `catboost`: `blocked (ImportError: DLL load failed while importing _catboost: An Application Control policy has blocked this file.)`  

---

## 1. Out-of-Fold Classification & Trading Performance (2021-2023)

| Model | Accuracy | Bal Acc | Macro F1 | SELL F1 (P / R) | HOLD F1 (P / R) | BUY F1 (P / R) | Pred % (S / H / B) | 10bps Net Ret | Sharpe | Train (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority_HOLD** | 49.64% | 33.33% | 22.12% | 0.000 (0.00/0.00) | 0.663 (0.50/1.00) | 0.000 (0.00/0.00) | 0.0% / 100.0% / 0.0% | +0.00% | +0.00 | 0.00s |
| **Candidate_v2_XGB_Balanced** | 44.73% | 38.01% | 37.87% | 0.228 (0.26/0.20) | 0.590 (0.55/0.64) | 0.318 (0.34/0.30) | 16.6% / 57.8% / 25.6% | +13.66% | +0.60 | 17.83s |
| **Candidate_v4_XGB_Sqrt** | 47.61% | 36.82% | 34.81% | 0.130 (0.25/0.09) | 0.632 (0.53/0.78) | 0.282 (0.36/0.23) | 7.4% / 73.5% / 19.2% | +13.68% | +0.68 | 16.82s |
| **XGB_Natural_Prior** | 49.05% | 35.68% | 30.98% | 0.068 (0.27/0.04) | 0.650 (0.51/0.88) | 0.211 (0.37/0.15) | 3.0% / 85.4% / 11.6% | +11.17% | +0.72 | 9.05s |
| **LightGBM_Natural** | 49.54% | 33.49% | 23.08% | 0.004 (0.09/0.00) | 0.663 (0.50/0.99) | 0.026 (0.41/0.01) | 0.5% / 98.5% / 0.9% | +3.53% | +0.84 | 4.46s |
| **LightGBM_Sqrt** | 49.65% | 34.29% | 25.92% | 0.023 (0.23/0.01) | 0.661 (0.50/0.96) | 0.094 (0.41/0.05) | 1.1% / 95.1% / 3.8% | +8.98% | +1.09 | 4.29s |
| **LightGBM_Sqrt_Reg** | 49.45% | 34.20% | 26.02% | 0.012 (0.21/0.01) | 0.658 (0.50/0.96) | 0.110 (0.38/0.06) | 0.6% / 94.4% / 5.0% | +10.72% | +1.13 | 3.61s |
| **LightGBM_Balanced** | 46.66% | 36.09% | 34.15% | 0.161 (0.26/0.12) | 0.626 (0.52/0.78) | 0.238 (0.34/0.18) | 9.7% / 74.4% / 15.9% | +7.35% | +0.43 | 4.02s |
| **LightGBM_Fixed200_Sqrt** | 47.68% | 37.07% | 35.25% | 0.144 (0.26/0.10) | 0.633 (0.53/0.78) | 0.281 (0.36/0.23) | 8.1% / 73.0% / 18.9% | +13.35% | +0.68 | 9.81s |
| **LightGBM_Fixed200_Balanced** | 44.40% | 38.00% | 37.91% | 0.230 (0.26/0.21) | 0.585 (0.55/0.62) | 0.323 (0.34/0.31) | 17.1% / 56.4% / 26.5% | +10.99% | +0.50 | 9.99s |
| **HistGB_Natural** | 48.89% | 35.79% | 31.47% | 0.077 (0.26/0.04) | 0.649 (0.52/0.87) | 0.218 (0.37/0.16) | 3.6% / 84.0% / 12.5% | +14.92% | +0.91 | 30.85s |
| **HistGB_Sqrt** | 47.60% | 37.04% | 35.23% | 0.140 (0.26/0.10) | 0.630 (0.53/0.78) | 0.286 (0.36/0.24) | 7.9% / 72.8% / 19.3% | +15.70% | +0.78 | 28.85s |
| **HistGB_Sqrt_Reg** | 48.23% | 36.69% | 34.11% | 0.123 (0.26/0.08) | 0.641 (0.53/0.82) | 0.259 (0.36/0.20) | 6.4% / 77.0% / 16.6% | +14.83% | +0.77 | 17.11s |
| **HistGB_Sqrt_D5** | 46.45% | 36.72% | 35.42% | 0.164 (0.26/0.12) | 0.618 (0.53/0.74) | 0.281 (0.34/0.24) | 9.5% / 70.0% / 20.5% | +14.34% | +0.72 | 39.51s |
| **HistGB_Balanced** | 43.75% | 37.49% | 37.41% | 0.222 (0.24/0.20) | 0.577 (0.55/0.61) | 0.322 (0.34/0.31) | 17.6% / 55.8% / 26.6% | +13.62% | +0.60 | 31.74s |

## 2. Accuracy & Macro F1 by Chronological Evaluation Year

| Model | 2021 Acc / MacroF1 | 2022 Acc / MacroF1 | 2023 Acc / MacroF1 | LogLoss | Brier | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority_HOLD | 47.03% / 21.32% | 44.33% / 20.48% | 57.67% / 24.38% | 8.1166 | 1.0071 | 0.5036 |
| Candidate_v2_XGB_Balanced | 41.53% / 36.85% | 40.87% / 37.81% | 51.89% / 33.13% | 1.0691 | 0.6455 | 0.0209 |
| Candidate_v4_XGB_Sqrt | 45.08% / 35.80% | 42.95% / 34.60% | 54.91% / 29.24% | 1.0474 | 0.6296 | 0.0182 |
| XGB_Natural_Prior | 46.41% / 32.85% | 44.07% / 30.47% | 56.78% / 26.07% | 1.0379 | 0.6220 | 0.0106 |
| LightGBM_Natural | 46.82% / 23.88% | 44.31% / 20.47% | 57.59% / 24.48% | 1.0308 | 0.6194 | 0.0239 |
| LightGBM_Sqrt | 46.63% / 27.29% | 45.21% / 23.83% | 57.20% / 25.40% | 1.0433 | 0.6284 | 0.0772 |
| LightGBM_Sqrt_Reg | 46.02% / 27.99% | 44.99% / 22.99% | 57.44% / 25.32% | 1.0444 | 0.6290 | 0.0747 |
| LightGBM_Balanced | 42.93% / 34.58% | 41.98% / 33.71% | 55.17% / 29.88% | 1.0779 | 0.6526 | 0.1073 |
| LightGBM_Fixed200_Sqrt | 44.99% / 36.29% | 43.43% / 35.43% | 54.72% / 29.07% | 1.0485 | 0.6299 | 0.0140 |
| LightGBM_Fixed200_Balanced | 41.08% / 36.84% | 40.67% / 37.90% | 51.53% / 33.80% | 1.0710 | 0.6466 | 0.0140 |
| HistGB_Natural | 46.75% / 33.40% | 43.79% / 31.10% | 56.21% / 26.36% | 1.0439 | 0.6256 | 0.0195 |
| HistGB_Sqrt | 44.99% / 36.44% | 43.21% / 34.91% | 54.68% / 29.78% | 1.0527 | 0.6329 | 0.0174 |
| HistGB_Sqrt_Reg | 45.66% / 35.09% | 43.30% / 34.19% | 55.83% / 28.18% | 1.0481 | 0.6299 | 0.0297 |
| HistGB_Sqrt_D5 | 43.88% / 36.34% | 42.22% / 35.26% | 53.33% / 30.65% | 1.0577 | 0.6359 | 0.0134 |
| HistGB_Balanced | 40.93% / 36.74% | 39.67% / 36.82% | 50.74% / 34.86% | 1.0745 | 0.6489 | 0.0153 |

## 3. Phase 4 Acceptance Verdict
* **Winning Replacement Candidate:** `LightGBM_Fixed200_Balanced`
* **Best Alternative by Macro F1:** `LightGBM_Fixed200_Balanced` (37.91%)
* **Best Alternative by Accuracy:** `LightGBM_Sqrt` (49.65%)

> **Environment Notice on CatBoost:** `catboost==1.2.10` was installed in `.venv`, but loading `_catboost.pyd` was blocked by Windows Application Control policy (`ImportError: DLL load failed while importing _catboost: An Application Control policy has blocked this file.`).

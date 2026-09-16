# StockDNA-AI: Project Status & Submission Audit

**Classification**: **Production-Ready Academic Prototype**  
*(Not a live automated broker-execution trading system)*

**Date of Audit**: September 16, 2026  
**Repository State**: Feature-Complete & Frozen (Phase 7 Final Gate)

---

## 1. System Specifications & Contract Summary

| Metric / Parameter | Value / Implementation |
|---|---|
| **Production Model** | Gradient Boosted Trees (`XGBClassifier`) |
| **Model Version** | `v1.0.0` (Locked in `models/production/model.json`) |
| **Feature Dimension** | **34 Scale-Invariant Features** (0 nominal price levels) |
| **Feature Breakdown** | **10 INTERNAL** (company-specific micro) / **24 EXTERNAL** (macro/technical) |
| **Prediction Target** | 5-trading-day forward return: $\text{Close}[t+5] / \text{Close}[t] - 1$ |
| **Class Formulations** | SELL ($\le -2\%$), HOLD ($-2\% \text{ to } +2\%$), BUY ($\ge +2\%$) |
| **XAI Engine** | Exact tree-kernel Shapley values (`shap.TreeExplainer`) |
| **Data Tool Decoupling** | Standalone service (`data_tool/`), zero ML coupling |
| **Backend Framework** | FastAPI (Python 3.10) with JWT Auth & SQLite persistence |
| **Frontend Framework** | React 18 + Vite SPA, Vanilla CSS Design System |

---

## 2. Scientific Evaluation Results

### A. Chronological Split Design
- **TRAIN**: 2018-10-23 to 2023-12-21 (3,786 rows; 1,262 bars/ticker)
- **VALIDATION**: 2024-01-02 to 2024-12-23 (717 rows; 239 bars/ticker)
- **TEST (Untouched Out-of-Sample)**: 2025-01-02 to 2025-12-19 (717 rows; 239 bars/ticker)
- **Temporal Embargo**: Verified zero overlap (TRAIN final bar realizes Dec 29, 2023; VAL starts Jan 2, 2024).

### B. Out-of-Sample Performance Metrics (2025 Test Set)
- **Accuracy**: 34.45%
- **Balanced Accuracy**: 34.18%
- **Macro Precision**: 31.62%
- **Macro Recall**: 34.18%
- **Macro F1 Score**: 31.40%
- **Weighted F1 Score**: 34.03%

### C. Population SHAP Attribution ($N=1,434$)
- **Mean INTERNAL Attribution**: 22.27% (Median: 19.91%)
- **Mean EXTERNAL Attribution**: 77.73% (Median: 80.09%)
- **Normalization Validity**: 100.0% of instances satisfy $\text{Internal}\% + \text{External}\% = 100.0\%$.

### D. Audited Daily Portfolio Backtest (2025 Test Set)
- **Daily Strategy Cumulative Return (Net)**: **-1.96%** (1-day execution lag, 10 bps costs)
- **Buy-and-Hold Universe Benchmark**: **-2.66%**
- **Always HOLD / Cash Benchmark**: 0.00%
- **Strategy Maximum Drawdown**: -21.71% (Benchmark: -19.85%)
- **Annualized Sharpe Ratio**: -0.02 (Benchmark: -0.08)

---

## 3. Engineering & Architecture Status

### A. Backend Services (`backend/`)
- [x] CORS middleware configured for secure local origins.
- [x] JWT bearer authentication with bcrypt password hashing and token expiry.
- [x] Relational schema with user isolation (`User`, `PredictionHistory`, `DataProvenanceLog`).
- [x] Read-only `/research/` router dynamically serving audited Phase 5 artifacts.
- [x] 100% test coverage across core contracts (51/51 tests passing).

### B. Independent Data Studio (`data_tool/`)
- [x] Abstract `BaseDataProvider` interface ensuring loose coupling.
- [x] Multi-source ingestion (CSV, OHLCV, news headlines).
- [x] Header normalization and corrupted record rejection.
- [x] Chronological composite-key deduplication (`['Ticker', 'Date']`).
- [x] Automated factor taxonomy classifier (`INTERNAL` vs `EXTERNAL`).
- [x] Financial sentiment NLP heuristic scorer.
- [x] Cryptographic point-in-time provenance stamping (`ProcessedDataBundle`).

### C. Frontend Interface (`frontend/`)
- [x] Executive Dashboard with immediate showcase of the Three Deliverables.
- [x] Prediction page with decision cards, confidence scores, and probability gauges.
- [x] SHAP visualizer with attribution donut, top factors, and horizontal waterfall chart.
- [x] Independent Data Studio with 4 operational tabs.
- [x] User-isolated history audit trail with filtering and deletion.
- [x] Scientific methodology documentation with math, embargo tables, and backtests.
- [x] Responsive layout tested for desktop, tablet, and mobile views.
- [x] Built to production static bundle (`dist/`) and mounted in FastAPI.

---

## 4. Verification & Testing Evidence

```
Total Test Count: 51
Pass Rate: 100% (51/51 passed)
Failures: 0
Errors: 0
Execution Time: ~5.5 seconds
```

### End-to-End API Acceptance Pipeline:
1. User registration & JWT login: **PASS**
2. Inference execution & probability distribution: **PASS**
3. TreeExplainer SHAP attribution decomposition: **PASS**
4. Prediction history persistence under user isolation: **PASS**
5. Data tool universe retrieval & factor taxonomy: **PASS**
6. News sentiment scoring: **PASS**
7. Data bundle cleaning and provenance generation: **PASS**
8. Read-only research metrics, baselines, and backtest queries: **PASS**

---

## 5. Deployment & Operating Instructions

### A. Local Execution Mode (Development & Auditing):
```bash
# 1. Compile frontend
cd frontend && npm run build && cd ..

# 2. Launch FastAPI server (serves both API and SPA)
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Application accessible at: **`http://127.0.0.1:8000/`**

### B. Public Cloud Deployment (Zero-Cost on Render):
- **Hosting Provider**: Render (Web Service Free Tier)
- **Blueprint Manifest**: [`render.yaml`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/render.yaml) & [`Dockerfile`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/Dockerfile)
- **Deployment URL**: `https://stockdna-ai.onrender.com` (upon GitHub blueprint sync)
- **Cold-Start Availability**: Inactivity for >15 minutes causes instance down-scaling; first wake-up request takes ~30–50s.
- **Persistence**: Ephemeral SQLite by default for evaluation; permanent persistence via managed PostgreSQL `DATABASE_URL`.
- **Classification**: **Production-Ready Academic Prototype** (explicitly NOT a live trading execution system).

---

## 6. Known Boundaries & Limitations

1. **Academic Decision-Support Prototype**: The system is designed to demonstrate ML methodology and XAI attribution for research and evaluation; it does not execute live market orders.
2. **Equity Universe**: Audited on 3 large-cap Indian equities (`TCS`, `INFY`, `RELIANCE`).
3. **Low Financial Signal-to-Noise Ratio**: Directional return prediction operates in a naturally noisy environment where a balanced accuracy of ~34% to 36% represents realistic bounds without lookahead leakage.
4. **Attribution, Not Causality**: SHAP values quantify feature contributions to the model's mathematical output log-odds, not macroeconomic causality.

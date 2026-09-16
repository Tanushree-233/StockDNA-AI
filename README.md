# StockDNA-AI: Explainable AI Decision-Support System for Equities

StockDNA-AI is an institutional machine learning and Explainable AI (XAI) decision-support system designed to classify short-term stock return regimes while providing transparent, mathematically grounded attribution of model decisions into company-specific **INTERNAL** versus market/macro **EXTERNAL** factors.

---

## 1. Problem Statement

Predicting equity returns is notorious for low signal-to-noise ratios, non-stationarity, and severe data leakage risks (lookahead bias, overlapping forward return windows, and scaling leakage). Traditional machine learning applications in finance often suffer from three critical flaws:
1. **The "Black Box" Dilemma**: Practitioners receive predictions without understanding whether signals originate from company quarterly fundamentals or broad market momentum.
2. **Temporal & Survivorship Leakage**: Training models with un-embargoed multi-day forward return labels or nominal price levels introduces hidden lookahead bias.
3. **Monolithic Tight Coupling**: Data ingestion, normalization, and feature transformations are entangled with specific model architectures, preventing reusable data governance.

StockDNA-AI addresses these challenges by establishing an end-to-end audited pipeline with strict chronological embargoes, scale-invariant feature math, a logically decoupled Data Tool, and SHAP-based feature decomposition.

---

## Live Demo & Public Deployment

- **Deployment Platform**: [Render](https://render.com) (Web Service Free Tier)
- **Declarative Blueprint**: [`render.yaml`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/render.yaml) & [`Dockerfile`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/Dockerfile)
- **Deployment Architecture**: Unified single-container architecture running the compiled React 18 frontend mounted statically inside FastAPI at `/`, backed by the locked production XGBoost engine and TreeExplainer cache.
- **Availability & Cold Starts**: Hosted on Render's free tier. Inactivity for >15 minutes causes the container to spin down to save compute; the first request after an idle period incurs a **30 to 50 second cold start** while the runtime spins up. Subsequent requests respond with standard low latency.
- **Persistence Scope**: Default deployment utilizes SQLite ephemeral container storage suitable for live demonstrations and audit evaluation. Persistent storage can be bound by providing a managed PostgreSQL `DATABASE_URL`.

---

## 2. Three Core Contributions

StockDNA-AI delivers three distinct, independently verifiable capabilities:

### Deliverable 1: BUY / HOLD / SELL Stock Movement Prediction
A production-grade multi-class gradient boosting model that predicts whether an equity's forward 5-trading-day return will experience upward momentum, sideways consolidation, or downward movement, accompanied by calibrated class probability distributions.

### Deliverable 2: SHAP-based INTERNAL vs EXTERNAL Explainability
An XAI layer powered by TreeExplainer that decomposes every individual inference into:
- **INTERNAL Attribution**: Company-specific micro factors (reported quarterly EPS, consensus earnings surprises, earnings event proximity).
- **EXTERNAL Attribution**: Broader market, macro, and technical momentum factors (India VIX volatility benchmark, NIFTY 50 index trends, Normalized ATR, moving average spreads).

### Deliverable 3: Independent Data Studio & Factor Classifier
A standalone, decoupled financial data aggregation, cleaning, deduplication, and automated factor taxonomy service. It operates independently of the ML model and can be consumed by external analytical pipelines via standard contracts (`BaseDataProvider` and `ProcessedDataBundle`).

---

## 3. System Architecture

```
                       Raw Multi-Source Data Feeds
          (Daily OHLCV, Quarterly Earnings Events, India VIX, NIFTY 50)
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │             INDEPENDENT DATA TOOL SERVICE                │
       │  - Data Cleaning & corrupted record rejection             │
       │  - Deduplication across composite keys (Ticker + Date)    │
       │  - Point-in-time timestamp & cryptographic provenance     │
       │  - Automated Factor Classification (INTERNAL vs EXTERNAL) │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Emits ProcessedDataBundle
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │             CANONICAL FEATURE ENGINEERING                │
       │  - 34 Scale-Invariant Features (zero nominal prices)     │
       │  - 10 INTERNAL Features | 24 EXTERNAL Features           │
       │  - Stationarity and cross-asset scale comparability      │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Fixed 34-Feature Vector
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │           PRODUCTION XGBOOST PREDICTION ENGINE           │
       │  - Multiclass Softmax: P(SELL), P(HOLD), P(BUY)          │
       │  - 5-trading-day horizon return classification           │
       └────────────────────────────┬─────────────────────────────┘
                                    │ Probability Distribution + Prediction
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │              SHAP EXPLAINABILITY LAYER (XAI)             │
       │  - Exact TreeExplainer tree-kernel margins               │
       │  - Log-odds attribution decomposed by factor origin      │
       │  - Aggregated Attribution: Internal % vs External %      │
       └────────────────────────────┬─────────────────────────────┘
                                    │ JSON Response
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │        FASTAPI REST BACKEND & PERSISTENCE LAYER          │
       │  - JWT Bearer Authentication & User Isolation            │
       │  - Relational Prediction Audit History (SQLite)          │
       │  - Read-Only Research & Backtest Artifact Endpoints      │
       └────────────────────────────┬─────────────────────────────┘
                                    │
                                    ▼
       ┌──────────────────────────────────────────────────────────┐
       │             REACT 18 + VITE WEB INTERFACE                │
       │  - Dashboard & Ticker Universe Visualizer                │
       │  - Decision Card with Confidence & Probability Gauge     │
       │  - Attribution Donut & Bi-directional SHAP Waterfall     │
       │  - Standalone Data Studio & Provenance Inspector         │
       │  - Audited Methodology & Scientific Performance Tables   │
       └──────────────────────────────────────────────────────────┘
```

---

## 4. Machine Learning Methodology

### A. Target Formulation
The model predicts a forward-looking 5-trading-day return ($h=5$) relative to observation bar $t$:

$$\text{Forward\_Return\_5d}[t] = \frac{\text{Close}[t+5]}{\text{Close}[t]} - 1$$

Continuous returns are categorized into three mutually exclusive directional classes:
- **SELL** (Index 0): $\text{Forward\_Return\_5d}[t] \le -2.0\%$
- **HOLD** (Index 1): $-2.0\% < \text{Forward\_Return\_5d}[t] < +2.0\%$
- **BUY** (Index 2): $\text{Forward\_Return\_5d}[t] \ge +2.0\%$

*Note: In the user interface, HOLD is rendered as **HOLD (KEEP)** to reflect portfolio decision support without altering underlying class labels.*

### B. Strict Chronological Splits & Embargo Proof
To guarantee zero lookahead leakage, datasets are split strictly chronologically. Overlapping 5-day return labels require an embargo buffer to prevent split boundary contagion:

| Split | Date Range | Observations | Realization Date of Final Bar ($t+5$) | Next Split Start | Embargo Status |
|---|---|---|---|---|---|
| **TRAIN** | 2018-10-23 to 2023-12-21 | 3,786 rows | 2023-12-29 | 2024-01-02 | **PASS** (1 trading day buffer) |
| **VALIDATION** | 2024-01-02 to 2024-12-23 | 717 rows | 2024-12-31 | 2025-01-02 | **PASS** (1 trading day buffer) |
| **TEST** | 2025-01-02 to 2025-12-19 | 717 rows | Final 5 rows dropped ($t+5$ unobserved) | — | **Untouched Out-of-Sample** |

The **2025 Test Set** was kept completely untouched during model selection, hyperparameter tuning, and probability thresholding.

### C. 34 Scale-Invariant Features
All nominal prices (e.g. `Close`, `SMA_50`, `EMA_20`) are strictly prohibited in the production input matrix. Every indicator is normalized into percentage spreads, ratios, or bounded oscillators:
- **10 INTERNAL Features**: Reported EPS, Consensus EPS Surprise, Surprise Percentile, Earnings Event Proximity, Normalized Volatility Regimes during Earnings Windows.
- **24 EXTERNAL Features**: India VIX Level, VIX 5d/20d Ratios, NIFTY 50 Relative Performance, Normalized ATR (ATR/Close), MACD Spread/Close, Distances to SMA (SMA20, SMA50, SMA200), RSI-14, Volume-to-MA20 Ratio, Historical Volatility (20d, 60d).

---

## 5. Explainable AI: SHAP Methodology

StockDNA-AI utilizes **TreeExplainer** to extract exact Shapley values for tree ensembles:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f_x(S \cup \{i\}) - f_x(S) \right]$$

### Attribution vs. Causality
- **Attribution Definition**: SHAP values explain how much each feature shifted the model's output log-odds away from the expected base value for a specific instance.
- **No Causal Claim**: SHAP percentages represent *model decision attribution*, NOT macroeconomic or corporate causality.

### Aggregation Formula
Individual absolute SHAP contributions are grouped into **INTERNAL** ($\mathcal{I}$) and **EXTERNAL** ($\mathcal{E}$) factor sets:

$$\text{Internal Attribution \%} = \frac{\sum_{j \in \mathcal{I}} |\phi_j|}{\sum_{k \in \mathcal{I} \cup \mathcal{E}} |\phi_k|} \times 100\%$$

Across the entire validation and test population ($N=1,434$), external market forces account for an average of **77.73%** of model attribution, while company-specific quarterly fundamentals account for **22.27%**, aligning with empirical asset pricing theory regarding short-term returns.

---

## 6. Scientific Validation & Model Benchmarking

Models were evaluated across the validation set (2024) and the untouched out-of-sample test set (2025):

| Model Architecture | Split | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| **Always HOLD (Zero-Rule)** | Validation / Test | 50.21% / 50.91% | 33.33% / 33.33% | 22.28% / 22.49% | 33.57% / 34.35% |
| **Majority Class Predictor** | Validation / Test | 50.21% / 50.91% | 33.33% / 33.33% | 22.28% / 22.49% | 33.57% / 34.35% |
| **Stratified Random Baseline** | Validation / Test | 34.45% / 40.73% | 30.02% / 37.40% | 30.08% / 37.20% | 34.85% / 41.06% |
| **Balanced Logistic Regression** | Validation / Test | 39.61% / 42.68% | 33.32% / 37.47% | 33.11% / 36.83% | 38.55% / 41.46% |
| **Balanced Random Forest** | Validation / Test | 36.12% / 43.93% | 31.14% / 36.86% | 30.18% / 35.61% | 35.12% / 41.45% |
| **Production XGBoost** | **Validation (2024)** | **40.45%** | **36.51%** | **35.93%** | **40.19%** |
| **Production XGBoost** | **Untouched Test (2025)**| **34.45%** | **34.18%** | **31.40%** | **34.03%** |

*Academic Discussion on Modest Accuracy*:
In equity markets, the Always-HOLD baseline achieves ~50% raw accuracy simply because approximately half of 5-day intervals experience sideways drift between $-2\%$ and $+2\%$. However, Always-HOLD produces 0 recall on directional trading opportunities. Production XGBoost achieved the highest validation Balanced Accuracy (36.51%) and Macro F1 (35.93%), preserving positive information coefficient without data leakage.

---

## 7. Audited Financial Backtest (2025 Test Period)

To prevent reporting artifacts, we audited the naive arithmetic trade return sum against an institutional daily mark-to-market portfolio simulation:

| Metric | Stacked Trade Sum (Naive) | Daily Portfolio Simulation (Audited) | Buy & Hold Universe Benchmark |
|---|---|---|---|
| **Return Methodology** | Arithmetic sum of trade returns | Mark-to-market daily rebalanced with 1-day lag | Equal-weighted basket of TCS, INFY, RELIANCE |
| **Transaction Costs** | 0 bps | 10 bps per trade | 0 bps |
| **Cumulative Return** | +35.75% | **-1.96%** | **-2.66%** |
| **Annualized Sharpe** | 0.16 | **-0.02** | **-0.08** |
| **Maximum Drawdown** | -70.36% | **-21.71%** | **-19.85%** |
| **Always HOLD / Cash** | 0.00% | **0.00%** | 0.00% |

*Conclusion*: The naive arithmetic return (+35.75%) is a common reporting distortion caused by summing overlapping trade returns. In realistic daily mark-to-market simulation, the model produced **-1.96%**, outperforming the underlying benchmark basket (**-2.66%**) during a consolidation market year.

---

## 8. API Specifications

The FastAPI backend exposes structured endpoints with Pydantic validation:

### Authentication
- `POST /auth/register`: Create analyst account (`username`, `email`, `password`).
- `POST /auth/login`: Authenticate and receive signed JWT bearer token.

### Prediction & XAI
- `POST /predict`: Generate 5-day movement prediction, class probabilities, and full TreeExplainer SHAP decomposition (`ticker: "TCS"`).
- `GET /history/`: Retrieve user-isolated chronological prediction history.
- `DELETE /history/{id}`: Delete an individual history entry.

### Independent Data Studio
- `GET /datatool/universe`: Return tracked equity universe metadata.
- `POST /datatool/factors/classify`: Categorize feature names into INTERNAL vs EXTERNAL taxonomy.
- `POST /datatool/sentiment`: Classify financial news headlines into sentiment label and score.
- `POST /datatool/process`: Ingest, clean, deduplicate, and normalize raw records into a `ProcessedDataBundle`.

### Research & Methodology
- `GET /research/metrics`: Serve locked 2025 out-of-sample test evaluation metrics.
- `GET /research/baselines`: Serve baseline model comparison across splits.
- `GET /research/backtest`: Serve audited daily mark-to-market backtest results.
- `GET /research/regimes`: Serve post-earnings observation window analysis.

---

## 9. Running Locally vs Public Deployment

### A. LOCAL DEVELOPMENT
```bash
# 1. Backend Server (Local development)
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload

# 2. Frontend Dev Server (Separate terminal with HMR & proxy)
cd frontend
npm install
npm run dev
# Accessible at http://localhost:5173

# 3. Unified Local Production Test (Serves compiled React bundle from FastAPI root)
cd frontend && npm run build && cd ..
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
# Accessible at http://127.0.0.1:8000/
```

### B. PUBLIC CLOUD DEPLOYMENT (Zero-Cost on Render)
StockDNA-AI includes declarative infrastructure blueprints for instant public cloud hosting:
1. **GitHub Connection**: Push code to `main` branch of `https://github.com/Tanushree-233/StockDNA-AI`.
2. **Render Blueprint**: On [Render.com](https://dashboard.render.com), click **New +** &rarr; **Blueprint** and select the repository.
3. **Automated Provisioning**: Render reads [`render.yaml`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/render.yaml), builds the Vite frontend, installs Python dependencies, and deploys to a free HTTPS URL (e.g. `https://stockdna-ai.onrender.com`).
4. **Alternative Container Deploy**: Use the production [`Dockerfile`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/Dockerfile) to deploy to Google Cloud Run, AWS ECS, or Fly.io.

### C. Automated Verification Suites
```bash
# Run all 51 backend regression tests:
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Run end-to-end API acceptance test:
.\.venv\Scripts\python.exe scripts/test_api_endpoints.py
```

---

## 10. Research Limitations & Academic Boundary Notes

1. **Noisy Return Dynamics**: Predicting 5-day directional returns in equity markets is constrained by low signal-to-noise ratios. A balanced accuracy of ~34% to 36% represents realistic predictive bounds without lookahead leakage.
2. **Universe Scope**: The test universe is focused on 3 high-liquidity Indian equities (`TCS`, `INFY`, `RELIANCE`) across 2018–2025.
3. **Sparse Earnings Frequency**: Indian corporations report financial statements quarterly (4 events per year per ticker). Consequently, market technical factors naturally dominate the high-frequency daily SHAP attribution (~78% vs ~22%).
4. **Decision Support, Not Execution**: The system produces decision-support classifications; it is not connected to a live broker execution order router.
5. **Attribution, Not Causation**: SHAP values indicate how mathematical inputs shifted the model's probability distribution, not underlying macroeconomic causality.

---

## 11. Project Team

- **Om More**
- **Garv Hemnani**
- **Tamanna Harchandani**
- **Tanushree Rathod**
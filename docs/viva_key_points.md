# StockDNA-AI: Academic Viva & Defense Key Points

This document provides rigorous, scientifically grounded answers to the 16 primary questions commonly raised by academic evaluators, reviewers, and professors.

---

### 1. Why BUY/HOLD/SELL classification instead of exact price regression?
**Answer**:
Exact price regression in equity markets violates stationarity and suffers from extreme noise. Financial time series are non-stationary in prices ($I(1)$ processes) and exhibit heteroskedasticity. Predicting an exact rupee price (e.g. ₹4,123.50) creates false precision and encourages over-fitting to recent nominal levels. In contrast, classifying expected directional movement into **BUY**, **HOLD (sideways/neutral)**, and **SELL** regimes directly aligns with portfolio risk management and asset allocation decisions, converting non-stationary price dynamics into stationary return categories.

---

### 2. Why a 5-trading-day horizon ($h=5$)?
**Answer**:
A 1-day horizon is dominated by market microstructure noise, bid-ask bounce, and intraday order flow imbalance. Horizons longer than 20 days (e.g. quarterly) introduce severe multi-month macro confounders and reduce the number of non-overlapping training samples. The **5-trading-day horizon (one business week)** represents the standard sweet spot for swing trading and post-earnings drift analysis: it is long enough for quarterly fundamental surprises and momentum indicators to diffuse into market prices, yet short enough to maintain hundreds of distinct chronological observations.

---

### 3. Why ±2.0% thresholds for class separation?
**Answer**:
In Indian large-cap equities (NIFTY 50 components like TCS, INFY, and RELIANCE), the annualized daily volatility typically ranges between 18% and 24%, corresponding to a typical 5-day standard deviation of approximately $1.5\% \text{ to } 2.2\%$. 
- Setting thresholds too narrow (e.g. $\pm 0.5\%$) causes bid-ask noise and broker transaction costs (10–20 bps) to trigger false directional signals.
- Setting thresholds too wide (e.g. $\pm 5.0\%$) causes extreme class imbalance where >90% of samples become HOLD.
A $\pm 2.0\%$ threshold yields balanced, economically meaningful movement boundaries while accounting for slippage and execution costs.

---

### 4. How was data leakage prevented?
**Answer**:
We implemented a multi-layered leakage prevention architecture:
1. **Zero Nominal Prices**: All nominal price levels (`Close`, `Open`, `High`, `Low`, `SMA`, `EMA`) were strictly barred from the feature set. Only scale-invariant ratios, percentage returns, and bounded oscillators are permitted.
2. **Point-in-Time Availability**: Fundamental indicators (e.g. reported EPS, surprises) are strictly stamped with public release dates, ensuring data from an earnings report announced on Day $t$ is not accessible to bar $t-1$.
3. **Temporal Purge & Embargo**: We proved mathematically that the 5-day forward return of the last bar in TRAIN (2023-12-21) realizes on 2023-12-29—leaving a clean 1-trading-day buffer before the VALIDATION split begins (2024-01-02).
4. **Independent Imputation**: Imputation medians were computed strictly on the TRAIN partition and applied forward to VALIDATION and TEST.

---

### 5. Why was the 2025 Test Set kept untouched?
**Answer**:
In scientific machine learning, repeatedly evaluating models on a test set during feature engineering or hyperparameter tuning causes **data snooping bias** (overfitting to the test set). To ensure institutional validity, the 2025 test partition was frozen. All feature selection, baseline comparisons, and hyperparameter tuning were conducted strictly on the 2018–2023 TRAIN and 2024 VALIDATION sets. The final model was evaluated on the 2025 TEST set exactly once during the final validation gate.

---

### 6. Why choose XGBoost over Deep Learning / LSTMs?
**Answer**:
Tabular financial datasets with heterogeneous features (continuous technical ratios, discrete event flags, and quarterly surprises) consistently favor gradient-boosted decision trees over neural networks in empirical literature (e.g. Grinsztajn et al., *Why do tree-based models still outperform deep learning on typical tabular data?*, NeurIPS 2022). XGBoost handles feature correlation, missing values via default split directions, and non-linear interactions without requiring scale-distorting normalization layers. Furthermore, tree architectures support exact polynomial-time Shapley value computation via **TreeExplainer**.

---

### 7. Why use SHAP (SHapley Additive exPlanations)?
**Answer**:
Traditional feature importance methods (such as Gini impurity or gain) suffer from consistency and attribution bias: a feature can increase in real predictive power while its measured importance decreases depending on tree depth. SHAP is uniquely grounded in cooperative game theory and satisfies the four fundamental axioms of fair attribution: **Efficiency**, **Symmetry**, **Dummy (Null Player)**, and **Additivity**. Furthermore, SHAP provides **local explanations** for each individual trade recommendation rather than just global dataset averages.

---

### 8. What is the distinction between INTERNAL and EXTERNAL factors?
**Answer**:
- **INTERNAL Factors (Company-Specific Micro)**: Variables originating directly from corporate operations and quarterly regulatory disclosures—such as Reported EPS, Consensus Surprise percentages, surprise percentiles, and earnings event calendar flags.
- **EXTERNAL Factors (Market & Macro)**: Variables originating from broad-market sentiment and technical dynamics—including India VIX volatility levels, NIFTY 50 benchmark returns, Normalized ATR, and moving-average distance spreads.

---

### 9. Does SHAP prove causality?
**Answer**:
**No. SHAP measures mathematical attribution, not economic causation.** SHAP quantifies how each input variable shifted the model's output log-odds relative to the baseline expectation across the training dataset. For example, if a high VIX level contributes a $+0.21$ SHAP value toward a SELL prediction, it proves that the model used VIX to reach its decision; it does not prove that VIX caused the underlying company's stock to decline. We enforce this distinction across all UI and documentation.

---

### 10. Why is EXTERNAL attribution significantly higher (~78%) than INTERNAL (~22%)?
**Answer**:
This asymmetry reflects market reality:
1. **Update Frequency**: Indian companies report earnings disclosures once every 90 days (4 quarterly events per year). Consequently, on ~95% of trading days, company fundamentals remain static while market prices fluctuate.
2. **Systematic Beta**: In short-term (5-day) equity returns, macro liquidity, sector momentum, and market-wide volatility shocks dominate idiosyncratic corporate drift. Company fundamentals dictate long-term valuation anchors, but short-term price discovery is largely driven by systematic market risk.

---

### 11. Why is the model's test accuracy ~34.45%? Is that too low?
**Answer**:
No, ~34% is standard for an un-leaky 3-class financial forecasting model. 
- In a 3-class problem, pure random guessing yields 33.33%.
- The naive Always-HOLD baseline achieves ~50% raw accuracy purely because large-cap stocks consolidate between $-2\%$ and $+2\%$ half the time; however, Always-HOLD achieves **0% recall** on BUY and SELL opportunities.
- Our model achieved **34.18% Balanced Accuracy** and **31.40% Macro F1** on an untouched out-of-sample year (2025). In financial time series, an information coefficient that consistently beats random walk while identifying directional tails represents genuine predictive signal. Models claiming 80%+ accuracy in 5-day stock return classification invariably suffer from lookahead leakage or target overlap contamination.

---

### 12. Why is the audited backtest return near zero (-1.96%)?
**Answer**:
During Phase 3, naive summation of overlapping trade returns produced an arithmetic artifact of $+35.75\%$. When we applied rigorous portfolio math—daily mark-to-market accounting, concurrent capital allocation across multiple equities, a 1-day execution lag, and 10 bps transaction costs—the realistic return was **-1.96%**. 
Importantly, the underlying equity universe benchmark (equal-weighted TCS, INFY, and RELIANCE) lost **-2.66%** over the same period. The model preserved capital and outperformed the benchmark during a sideways/bearish consolidation year, demonstrating the value of academic auditing over naive metric reporting.

---

### 13. Why is the Data Tool completely decoupled from the prediction model?
**Answer**:
Decoupling fulfills three software engineering and ML principles:
1. **Reusability**: The Data Tool can ingest, clean, deduplicate, and classify financial feeds for other analytical services or alternative models (e.g. Random Forest, deep learning) without modifying data code.
2. **Independent Testability**: The data engineering pipeline has its own comprehensive test suite (`test_data_tool.py`) that executes in milliseconds without loading heavy ML dependencies.
3. **Point-in-Time Data Contracts**: The tool emits an immutable `ProcessedDataBundle` with cryptographic provenance, ensuring clean separation between data preparation and inference.

---

### 14. What are the key limitations of the system?
**Answer**:
1. **Small Universe**: Audited on 3 large-cap Indian equities (`TCS`, `INFY`, `RELIANCE`).
2. **Sparse Quarterly Fundamentals**: Only 4 earnings reports per ticker per year limit fundamental feature diversity.
3. **No Direct Broker Routing**: StockDNA-AI is an analytical decision-support tool, not an automated high-frequency order execution engine.
4. **Regime Vulnerability**: Severe black swan market shocks (e.g. sudden geopolitical crises) can disrupt statistical momentum patterns.

---

### 15. What would you improve with more data and compute?
**Answer**:
1. **Universe Expansion**: Broaden the universe to the entire NIFTY 100 or NIFTY 500 universe.
2. **Alternative Data Feeds**: Integrate real-time conference call audio transcripts, SEC/NSE corporate filing sentiment via fine-tuned financial LLMs (e.g. FinBERT), and supply-chain graph networks.
3. **Conformal Prediction**: Generate calibrated prediction intervals and set-valued predictions to quantify instance-level epistemic uncertainty.

---

### 16. How could this system be deployed into production?
**Answer**:
1. **Data Ingestion Cron**: Schedule nightly automated market data pulls at market close (3:30 PM IST) via the Data Tool.
2. **Containerization**: Deploy the FastAPI backend and compiled React SPA as Docker containers on Kubernetes or AWS ECS / Google Cloud Run.
3. **Database Scaling**: Migrate from local SQLite to managed PostgreSQL with connection pooling.
4. **Drift Monitoring**: Implement continuous model monitoring (Evidently AI / MLflow) tracking population stability index (PSI) and SHAP feature attribution drift to trigger automated re-training alerts.

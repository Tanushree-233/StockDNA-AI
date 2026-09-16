# StockDNA-AI: 5–7 Minute Viva & Demonstration Script

This script provides an exact, step-by-step presentation flow for presenting **StockDNA-AI** to evaluators, professors, or an examination committee.

---

## Timing Overview
- **Minute 0:00 – 1:00**: System Overview & Three Deliverables (Dashboard)
- **Minute 1:00 – 2:30**: Stock Prediction & Probability Distribution (Predict)
- **Minute 2:30 – 4:00**: SHAP Explainability & Factor Decomposition (XAI)
- **Minute 4:00 – 5:00**: Independent Data Studio & Factor Taxonomy (Data Tool)
- **Minute 5:00 – 6:15**: Scientific Methodology, Embargo Proof & Audited Backtest
- **Minute 6:15 – 7:00**: Limitations, Academic Honesty & Conclusion

---

## Step 1: Open Dashboard (Minute 0:00 – 1:00)
**Action**: Navigate to `http://127.0.0.1:8000/` (Dashboard / Home).

**Speaking Script**:
> "Good morning/afternoon. Today I am presenting **StockDNA-AI**, an institutional explainable machine learning decision-support system for equity markets.
> 
> Rather than treating stock prediction as an uninterpretable black box, StockDNA-AI is structured around **three distinct core contributions**, visible here on the dashboard:
> 1. **Directional Return Movement Prediction**: A multi-class model classifying forward 5-trading-day returns into BUY, HOLD (KEEP), and SELL regimes.
> 2. **Decomposed Explainable AI (XAI)**: A TreeExplainer SHAP layer that separates feature attribution into company-specific **INTERNAL** fundamentals versus broad-market **EXTERNAL** momentum.
> 3. **Independent Data Studio**: A completely decoupled data engineering layer that cleans, deduplicates, and classifies financial data independently of any machine learning model.
>
> Let's start by analyzing an equity."

---

## Step 2: Stock Prediction Workflow (Minute 1:00 – 2:30)
**Action**: Click **"Analyze a Stock"** or navigate to **Prediction**. Log in/register as an analyst, select **TCS**, and click **"Run Prediction"**.

**Speaking Script**:
> "Here on the Prediction page, we select an equity from our audited NIFTY universe—in this case, Tata Consultancy Services (`TCS`).
> 
> When we run the analysis, the locked production XGBoost model evaluates 34 scale-invariant features. Notice what is returned:
> - **Predicted Class**: The system predicts a directional regime (e.g. `BUY` or `HOLD`).
> - **Model Confidence & Probability Distribution**: We don't just provide a label; we display the calibrated probability distribution across SELL, HOLD, and BUY summing to 100%.
> - **Academic Disclaimer**: Notice our explicit disclaimer. This is decision support, not financial advice. The model predicts a statistical 5-day trajectory, not a guaranteed future price.
> - **Feature Contract**: The model strictly evaluates 34 scale-invariant indicators—nominal rupee stock prices like ₹4,000 are completely barred to prevent price-level bias."

---

## Step 3: SHAP Explainability & Factor Decomposition (Minute 2:30 – 4:00)
**Action**: Click **"Inspect SHAP Decomposition"** or navigate to **Explainability**.

**Speaking Script**:
> "This brings us to our second core contribution: **Explainability**.
> 
> Standard feature importance outputs a long list of variables without context. StockDNA-AI groups SHAP attribution into two economic categories:
> - **INTERNAL Factors**: Reported quarterly EPS, consensus earnings surprises, and earnings announcement event windows.
> - **EXTERNAL Factors**: India VIX volatility, NIFTY 50 index momentum, Normalized ATR, and moving average spreads.
>
> On the **Attribution Donut**, we see the exact percentage split: for example, 19.5% Internal versus 80.5% External. External forces typically dominate short-term 5-day horizons because quarterly earnings only occur 4 times a year, whereas market volatility trades daily.
>
> Below, our **Bi-directional Waterfall Chart** displays both **magnitude** and **direction**:
> - Features extending to the right in emerald pushed the model toward the predicted class.
> - Features extending to the left in crimson pushed against it.
>
> **Critical Academic Point**: SHAP values represent *model decision attribution*—how mathematical inputs shifted the model's log-odds—not real-world macroeconomic causes."

---

## Step 4: Independent Data Studio (Minute 4:00 – 5:00)
**Action**: Navigate to **Data Tool** in the top navbar.

**Speaking Script**:
> "Our third deliverable is the **Independent Data Studio**. 
> 
> In many ML projects, data preprocessing is entangled with model training code. In StockDNA-AI, the Data Tool is a standalone service operating via a clean contract.
> - In **Tab 1**, we view our authoritative tracked universe.
> - In **Tab 2**, our automated **Factor Taxonomy Classifier** categorizes any arbitrary list of features into `INTERNAL` or `EXTERNAL`.
> - In **Tab 3**, our **Financial Sentiment Classifier** analyzes news headlines using domain-tailored heuristics.
> - In **Tab 4**, we can run our sample cleaning pipeline. It ingests dirty records, rejects corrupted null rows, drops negative volumes, removes duplicate date entries, and generates a structured `ProcessedDataBundle` with cryptographic provenance."

---

## Step 5: Scientific Methodology, Embargo Proof & Audited Backtest (Minute 5:00 – 6:15)
**Action**: Navigate to **Methodology** in the top navbar.

**Speaking Script**:
> "Now let's examine our scientific validation gate. StockDNA-AI was engineered under zero-leakage constraints:
> 
> 1. **Target Formulation**:
>    $$\text{Forward\_Return\_5d}[t] = \frac{\text{Close}[t+5]}{\text{Close}[t]} - 1$$
>    Categorized with $\pm 2.0\%$ thresholds.
> 
> 2. **Programmatic Embargo Proof**:
>    In the embargo table, notice that the 5-day forward return of the last bar in the Training split (Dec 21, 2023) realizes on Dec 29, 2023—leaving a clean 1-trading-day buffer before the Validation split begins on Jan 2, 2024. The 2025 Test set was kept completely untouched.
>
> 3. **Honest Out-of-Sample Results**:
>    We compare against 6 baselines. While an Always-HOLD naive strategy achieves ~50% accuracy due to frequent sideways markets, it has 0 recall on trading opportunities. Our model achieved **36.51% Balanced Accuracy** on Validation and **34.18% Balanced Accuracy** on the untouched 2025 Test set. In financial NLP and equities forecasting, this reflects genuine out-of-sample signal without overfitting.
>
> 4. **Audited Financial Backtest**:
>    In early trials, summing individual trade returns showed a naive +35.75% return. However, we audited this using a realistic daily mark-to-market portfolio simulation with a 1-day execution lag and 10 bps transaction costs. The strategy generated **-1.96%**, outperforming the buy-and-hold universe benchmark of **-2.66%** during a market pullback year."

---

## Step 6: Limitations, Academic Rigor & Conclusion (Minute 6:15 – 7:00)
**Action**: Scroll to the concluding notes on Methodology or return to Dashboard.

**Speaking Script**:
> "To conclude, StockDNA-AI is presented with full academic honesty:
> - It is designed as a **decision-support prototype**, not an automated execution black box.
> - It explicitly avoids false claims of guaranteed profitability or market certainty.
> - It demonstrates that transparent XAI factor decomposition and decoupled data engineering can be combined into a robust, leakage-free machine learning architecture.
> 
> All 51 backend unit tests pass with zero regressions, and the entire system is live and operational. Thank you, and I welcome any questions."

# StockDNA-AI Data Engineering Pipeline

## 1. Purpose

The data engineering pipeline collects raw financial and market data,
cleans and standardizes the data, performs feature preparation,
validates the resulting dataset, and stores the clean dataset in the
StockDNA-AI database.

---

## 2. Data Flow

Raw Sources
↓
Data Collectors
↓
Raw Data
↓
Preprocessing
↓
Feature Engineering
↓
Master Dataset
↓
Data Validation
↓
SQLite Database
↓
Machine Learning / Prediction

---

## 3. Data Sources

The project collects the following categories of data:

- Stock prices
- Market indices
- India VIX
- Macro-economic market data
- Company fundamentals
- Earnings data
- Earnings events
- Internal financial data

---

## 4. Data Collection

Collectors are located in:

`scripts/collectors/`

Main collectors include:

- `download_prices.py`
- `download_market.py`
- `download_macro.py`
- `download_fundamentals.py`
- `download_earnings.py`
- `download_earnings_events.py`
- `download_internal.py`

Collected data is stored under:

`data/raw/`

---

## 5. Data Preprocessing

Preprocessing scripts are located in:

`scripts/processors/`

The preprocessing stage performs operations including:

- Date normalization
- Ticker normalization
- Duplicate removal
- Invalid-date removal
- Empty-data handling
- Column standardization
- Data sorting

Processed data is stored under:

`data/processed/`

---

## 6. Feature Engineering

Technical and financial features are prepared for machine learning.

Examples include:

- Daily Return
- SMA20
- SMA50
- SMA100
- SMA200
- EMA20
- EMA50
- EMA100
- RSI
- MACD
- Bollinger Bands
- ATR
- Volatility
- Momentum
- ROC
- Volume indicators
- Price spread indicators

The resulting features are combined into the master dataset.

---

## 7. Master Dataset

The final master dataset is:

`data/final/master_dataset.csv`

Current validated dataset:

- Rows: 5,322
- Columns: 38
- Companies: INFY, RELIANCE, TCS
- Minimum date: 2018-10-23
- Maximum date: 2025-12-29

---

## 8. Data Quality Validation

The master dataset is validated using:

`scripts/validators/validate_master_dataset.py`

The latest validation confirmed:

- Missing values: 0
- Duplicate rows: 0
- Duplicate Ticker + Date records: 0
- Invalid dates: 0
- Blank tickers: 0
- Infinite values: 0
- Invalid price values: 0
- Invalid target values: 0

Overall validation result:

**PASS**

---

## 9. Database Storage

The validated master dataset is stored in SQLite.

Database model:

`backend/database/models.py`

Database table:

`stock_data`

Database ingestion script:

`scripts/database/ingest_master_dataset.py`

Database verification script:

`scripts/database/verify_database.py`

The latest database verification confirmed:

- Database rows: 5,322
- Expected rows: 5,322
- Tickers match
- Date range matches
- Duplicate Ticker + Date records: 0
- Required NULL values: 0

Overall database verification result:

**PASS**

---

## 10. Machine Learning Integration

The clean dataset is used by the machine learning pipeline.

The prediction system uses an XGBoost model to classify stocks into:

- BUY
- HOLD
- SELL

The live prediction system also retrieves current:

- Stock market prices
- NIFTY 50 data
- India VIX data

Latest predictions are stored in:

`data/processed/internal/live_predictions.csv`

---

## 11. Production Pipeline

The complete pipeline is executed using:

`run_stockdna.py`

Pipeline sequence:

1. Download earnings events
2. Build internal dataset
3. Build earnings-event dataset
4. Build final ML dataset
5. Download current market data
6. Download NIFTY 50 data
7. Download India VIX data
8. Generate live predictions
9. Save live predictions

---

## 12. Data Engineering Deliverable

The completed data engineering workflow provides:

**Raw Sources**

↓

**Clean Standardized Dataset**

↓

**Validated Dataset**

↓

**Database**

↓

**Machine Learning / Prediction**

This establishes the data collection, cleaning, validation,
standardization, and database-ingestion layer for StockDNA-AI.
# Machine Learning Model Artifacts

> [!IMPORTANT]
> **Production Model Location**: The authoritative, leakage-free production model is housed exclusively in:
> [`models/production/`](file:///c:/Users/OM%20MORE/Documents/GitHub/StockDNA-AI/models/production/)
> - `model.json`: Production XGBoost multiclass model (locked v1.0.0).
> - `model_contract.json`: Official 34-feature production schema, classes (`SELL: 0`, `HOLD: 1`, `BUY: 2`), imputation defaults, and training split boundaries.
> - `metadata.json`: Full training provenance, hyperparameter specifications, and validation metrics.

### Root-Level Artifacts Notice
All `.pkl` files and `.csv` files in the root of `models/` represent:
1. Baseline architectures (`LogisticRegression.pkl`, `RandomForest.pkl`) used for academic comparative benchmarking.
2. Historical early-phase trial artifacts preserved for research reproducibility.

The production backend (`backend/predictor.py`) and inference pipeline load exclusively from `models/production/`.

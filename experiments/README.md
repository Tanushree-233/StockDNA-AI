# Historical Experiments & Exploratory Research

> [!NOTE]
> **Academic Archival Notice**: The scripts in this directory represent early experimental explorations (e.g. separate binary classifiers, un-embargoed trial models) conducted prior to the Phase 3–6 production unification.

These files are preserved solely for research reproducibility and audit trails. **They are NOT part of the active production system.**

### Authoritative Production Pipeline Locations:
- **Production Model**: `models/production/` (`model.json`, `model_contract.json`, `label_encoder.json`)
- **Feature Registry**: `config/feature_registry.py` (34 locked scale-invariant features)
- **ML Training & Validation**: `scripts/ml/` (`train_production_model.py`, `prove_temporal_leakage.py`, `evaluate_test_set.py`, `run_audited_backtest.py`)
- **Decoupled Data Service**: `data_tool/`
- **FastAPI REST Service**: `backend/`
- **React 18 + Vite Interface**: `frontend/`

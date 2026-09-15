"""
Research Router for StockDNA-AI.

Provides read-only access to audited Phase 5 scientific results, baselines,
earnings regime studies, and backtest evaluations.
"""

import os
import json
import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/research", tags=["Research & Methodology"])

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


@router.get("/metrics")
def get_metrics():
    """Returns final audited test metrics and model performance summary."""
    test_metrics_path = os.path.join(RESULTS_DIR, "test_metrics.json")
    if not os.path.exists(test_metrics_path):
        raise HTTPException(status_code=404, detail="Test metrics artifact not found.")
    
    with open(test_metrics_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/baselines")
def get_baselines():
    """Returns baseline model comparison table across validation and test sets."""
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Baseline comparison artifact not found.")
    
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")


@router.get("/backtest")
def get_backtest():
    """Returns audited financial backtest and benchmark comparison."""
    backtest_path = os.path.join(RESULTS_DIR, "backtest_results.json")
    if not os.path.exists(backtest_path):
        raise HTTPException(status_code=404, detail="Backtest results artifact not found.")
    
    with open(backtest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/regimes")
def get_regimes():
    """Returns post-earnings regime analysis across observation horizons."""
    csv_path = os.path.join(RESULTS_DIR, "earnings_regime_analysis.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail="Earnings regime artifact not found.")
    
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

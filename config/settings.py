from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import pandas as pd
import yaml

# Authoritative Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
COMPANIES_PATH = PROJECT_ROOT / "data" / "reference" / "companies.csv"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def load_raw_config() -> Dict[str, Any]:
    """Load raw YAML config dictionary."""
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def get_company_universe() -> pd.DataFrame:
    """
    Retrieve the authoritative list of equities tracked by StockDNA-AI.
    Reads from data/reference/companies.csv.
    """
    if not COMPANIES_PATH.exists():
        # Fallback default universe if reference file is missing
        return pd.DataFrame([
            {"ticker": "TCS.NS", "company": "Tata Consultancy Services", "sector": "IT"},
            {"ticker": "INFY.NS", "company": "Infosys", "sector": "IT"},
            {"ticker": "RELIANCE.NS", "company": "Reliance Industries", "sector": "Energy"},
        ])
    
    df = pd.read_csv(COMPANIES_PATH)
    df.columns = df.columns.str.strip().str.lower()
    if "ticker" not in df.columns:
        raise ValueError(f"Expected 'ticker' column in {COMPANIES_PATH}")
    
    # Provide clean ticker column (e.g. TCS instead of TCS.NS)
    df["clean_ticker"] = df["ticker"].astype(str).str.replace(".NS", "", regex=False).str.strip().str.upper()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    return df


def get_tickers(clean: bool = False) -> List[str]:
    """
    Get list of tickers in the tracked universe.
    
    Args:
        clean: If True, returns tickers without exchange suffix (e.g. 'TCS').
               If False, returns full exchange symbol (e.g. 'TCS.NS').
    """
    df = get_company_universe()
    col = "clean_ticker" if clean else "ticker"
    return df[col].tolist()


def get_ticker_mapping() -> Dict[str, str]:
    """
    Returns mapping from clean ticker to Yahoo exchange ticker.
    Example: {'TCS': 'TCS.NS', 'INFY': 'INFY.NS', 'RELIANCE': 'RELIANCE.NS'}
    """
    df = get_company_universe()
    return dict(zip(df["clean_ticker"], df["ticker"]))


def get_data_path(*subpaths: str) -> Path:
    """Return an absolute Path inside the project root data directory."""
    return PROJECT_ROOT.joinpath("data", *subpaths)


def get_model_path(*subpaths: str) -> Path:
    """Return an absolute Path inside the project root models directory."""
    return PROJECT_ROOT.joinpath("models", *subpaths)

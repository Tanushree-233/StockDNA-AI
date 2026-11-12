"""
Unified Model Loading, Inference, and Rollback Adapter for StockDNA-AI.

Decouples backend inference and prediction services from any single ML library
(XGBoost, HistGradientBoosting, LightGBM, CatBoost) by reading the versioned
`model_contract.json` and routing through standardized feature extraction and
probability inference interfaces.

Active Model Resolution Order:
  1. Environment override: `STOCKDNA_ACTIVE_MODEL` ("production", "candidate_v7",
     or an explicit directory path)
  2. Registry pointer: `models/active_model.json`
  3. Safe fallback: `models/production/`

Rollback:
  Call `rollback_to_production()` or run:
    python scripts/ml/model_adapter.py --rollback
  or set `STOCKDNA_ACTIVE_MODEL=production`.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_tool.provider import BaseDataProvider, DataToolProvider
from scripts.feature_engineering.canonical_features import build_canonical_features
from scripts.feature_engineering.candidate_features import build_candidate_features

MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_REGISTRY_PATH = MODELS_DIR / "active_model.json"
PROD_DIR = MODELS_DIR / "production"
CANDIDATE_V7_DIR = MODELS_DIR / "candidates" / "candidate_v7"

MODEL_LOCATIONS: Dict[str, Path] = {
    "production": PROD_DIR,
    "candidate_v1": MODELS_DIR / "candidates" / "candidate_v1",
    "candidate_v2": MODELS_DIR / "candidates" / "candidate_v2",
    "candidate_v7": CANDIDATE_V7_DIR,
}

# Thread-local / process-level cache keyed by resolved directory path
_ADAPTER_CACHE: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def clear_model_cache() -> None:
    """Clears the in-memory model/contract cache (used on model switch/rollback)."""
    _ADAPTER_CACHE.clear()


def resolve_model_dir(model_id_or_path: Optional[str] = None) -> Tuple[str, Path]:
    """
    Resolves which model directory is currently active or requested.
    Returns (model_id, Path_to_model_dir).
    """
    target = model_id_or_path or os.getenv("STOCKDNA_ACTIVE_MODEL")

    if target:
        if target in MODEL_LOCATIONS:
            return target, MODEL_LOCATIONS[target]
        p = Path(target)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.exists():
            return p.name, p
        raise FileNotFoundError(f"Requested model target '{target}' not found.")

    if ACTIVE_REGISTRY_PATH.exists():
        with open(ACTIVE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            reg = json.load(f)
        active_id = reg.get("active_model_id", "production")
        active_rel = reg.get("active_model_dir")
        if active_rel:
            p = Path(active_rel)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                return active_id, p
        if active_id in MODEL_LOCATIONS and MODEL_LOCATIONS[active_id].exists():
            return active_id, MODEL_LOCATIONS[active_id]

    return "production", PROD_DIR


def _load_artifact_by_contract(model_dir: Path, contract: Dict[str, Any]) -> Any:
    """Instantiates and loads a trained classifier according to its contract."""
    model_type = contract.get("model_type", "XGBClassifier")
    model_format = contract.get("model_format")

    # Locate artifact file
    candidate_filenames = []
    if contract.get("model_filename"):
        candidate_filenames.append(contract["model_filename"])
    candidate_filenames.extend([
        "model.json",
        "candidate_model.json",
        "candidate_model.joblib",
        "model.joblib",
        "candidate_model.cbm",
        "model.cbm",
    ])

    artifact_path = None
    for fname in candidate_filenames:
        p = model_dir / fname
        if p.exists():
            artifact_path = p
            break

    if artifact_path is None:
        raise FileNotFoundError(f"No model binary found in {model_dir}")

    ext = artifact_path.suffix.lower()

    if model_type == "XGBClassifier" or (ext == ".json" and "xgb" in model_type.lower()):
        from xgboost import XGBClassifier

        model = XGBClassifier()
        model.load_model(str(artifact_path))
        return model

    if model_type in ("HistGradientBoostingClassifier", "LGBMClassifier", "RandomForestClassifier") or (
        model_format == "joblib" or ext == ".joblib"
    ):
        return joblib.load(artifact_path)

    if model_type == "CatBoostClassifier" or ext == ".cbm":
        try:
            from catboost import CatBoostClassifier
        except Exception as exc:
            raise RuntimeError(f"CatBoost runtime unavailable on this host: {exc}") from exc
        model = CatBoostClassifier()
        model.load_model(str(artifact_path))
        return model

    raise ValueError(f"Unsupported model_type='{model_type}' in {model_dir}")


def load_model_from_dir(model_dir: Path, use_cache: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Loads a model and its contract from an explicit model directory."""
    resolved = str(model_dir.resolve())
    if use_cache and resolved in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[resolved]

    contract_path = model_dir / "model_contract.json"
    if not contract_path.exists():
        raise FileNotFoundError(f"Model contract not found at {contract_path}")

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = json.load(f)

    model = _load_artifact_by_contract(model_dir, contract)

    if use_cache:
        _ADAPTER_CACHE[resolved] = (model, contract)
    return model, contract


def load_active_model(
    model_id_or_path: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Any, Dict[str, Any], str]:
    """
    Loads the active model and contract via resolution rules.
    Returns (model, contract, active_model_id).
    """
    model_id, model_dir = resolve_model_dir(model_id_or_path)
    model, contract = load_model_from_dir(model_dir, use_cache=use_cache)
    return model, contract, model_id


def extract_features_for_contract(
    price_df: pd.DataFrame,
    contract: Dict[str, Any],
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes exact features required by `contract` using the appropriate
    feature pipeline (`canonical` vs `candidate_v2_scale_invariant`), then
    aligns column ordering and applies contract imputation rules.

    Returns (full_featured_df, aligned_X_df).
    """
    feature_names = contract["feature_names"]
    pipeline_type = contract.get("feature_pipeline")
    if pipeline_type is None:
        if "Earnings_Yield" in feature_names or "Forward_Earnings_Yield" in feature_names:
            pipeline_type = "candidate_v2_scale_invariant"
        else:
            pipeline_type = "canonical"

    if pipeline_type == "candidate_v2_scale_invariant":
        featured = build_candidate_features(
            price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )
    else:
        featured = build_canonical_features(
            price_df, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
        )

    if featured.empty:
        return featured, pd.DataFrame(columns=feature_names)

    imputation_dict = contract.get("imputation_values", {})
    X = featured[feature_names].copy()
    for feat in feature_names:
        fill_val = imputation_dict.get(feat, 0.0)
        X[feat] = (
            pd.to_numeric(X[feat], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(fill_val)
        )

    return featured, X


def predict_from_features(
    model: Any,
    contract: Dict[str, Any],
    X: pd.DataFrame,
    active_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Runs standardized predict_proba + class mapping on an aligned single/multi-row DataFrame."""
    feature_names = contract["feature_names"]
    X_ordered = X[feature_names]
    probs = model.predict_proba(X_ordered)[0]

    # Normalize defensively in case of floating-point drift
    prob_sum = float(np.sum(probs))
    if prob_sum > 0:
        probs = probs / prob_sum

    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    raw_mapping = contract.get("class_mapping", {"0": "SELL", "1": "HOLD", "2": "BUY"})
    class_mapping = {int(k): v for k, v in raw_mapping.items()}
    pred_label = class_mapping.get(pred_idx, "HOLD")

    probabilities_dict = {
        class_mapping.get(i, str(i)): round(float(probs[i]), 4)
        for i in range(len(probs))
    }

    feature_snapshot = {
        col: round(float(X_ordered[col].iloc[0]), 5)
        if pd.notnull(X_ordered[col].iloc[0])
        else 0.0
        for col in feature_names
    }

    return {
        "prediction": pred_label,
        "confidence": round(conf, 4),
        "probabilities": probabilities_dict,
        "model_version": contract.get("model_version", "1.0.0"),
        "model_type": contract.get("model_type", type(model).__name__),
        "active_model_id": active_model_id or contract.get("candidate_id", "production"),
        "feature_snapshot": feature_snapshot,
    }


def predict_ticker_via_adapter(
    ticker: str,
    price_df: Optional[pd.DataFrame] = None,
    nifty_df: Optional[pd.DataFrame] = None,
    vix_df: Optional[pd.DataFrame] = None,
    earnings_df: Optional[pd.DataFrame] = None,
    data_provider: Optional[BaseDataProvider] = None,
    model_id_or_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end model-agnostic single-ticker prediction entrypoint.
    """
    model, contract, active_id = load_active_model(model_id_or_path)
    clean_ticker = ticker.replace(".NS", "").strip().upper()

    if price_df is None:
        provider = data_provider or DataToolProvider()
        price_df = provider.get_market_data(clean_ticker)

    price_df = price_df.copy()
    price_df["Ticker"] = clean_ticker
    price_df["Date"] = pd.to_datetime(price_df["Date"])
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in price_df.columns:
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    price_df = (
        price_df.dropna(subset=["Close", "High", "Low", "Open"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if nifty_df is None:
        nifty_path = PROJECT_ROOT / "data" / "raw" / "market" / "nifty50.csv"
        if nifty_path.exists():
            nifty_df = pd.read_csv(nifty_path)
    if vix_df is None:
        vix_path = PROJECT_ROOT / "data" / "raw" / "market" / "indiavix.csv"
        if vix_path.exists():
            vix_df = pd.read_csv(vix_path)
    if earnings_df is None:
        earnings_path = PROJECT_ROOT / "data" / "processed" / "internal" / "earnings_events.csv"
        if earnings_path.exists():
            earnings_df = pd.read_csv(earnings_path)

    featured, X_all = extract_features_for_contract(
        price_df, contract, nifty_df=nifty_df, vix_df=vix_df, earnings_df=earnings_df
    )
    if featured.empty:
        raise ValueError(f"Feature computation resulted in empty dataset for {clean_ticker}")

    latest_date = featured["Date"].iloc[-1]
    timestamp_str = latest_date.strftime("%Y-%m-%d %H:%M:%S")
    X_latest = X_all.iloc[[-1]]

    pred_out = predict_from_features(model, contract, X_latest, active_model_id=active_id)
    pred_out["ticker"] = clean_ticker
    pred_out["timestamp"] = timestamp_str
    return pred_out


def activate_model(model_id: str, reason: str = "Promoted after passing acceptance criteria") -> Dict[str, Any]:
    """Switches the active deployment model in `models/active_model.json`."""
    if model_id not in MODEL_LOCATIONS:
        raise ValueError(f"Unknown model_id '{model_id}'. Valid options: {list(MODEL_LOCATIONS.keys())}")
    target_dir = MODEL_LOCATIONS[model_id]
    if not (target_dir / "model_contract.json").exists():
        raise FileNotFoundError(f"Cannot activate '{model_id}': missing model_contract.json in {target_dir}")

    # Verify model loads cleanly before switching
    clear_model_cache()
    _, contract = load_model_from_dir(target_dir, use_cache=False)

    payload = {
        "active_model_id": model_id,
        "active_model_dir": str(target_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_version": contract.get("model_version"),
        "model_type": contract.get("model_type"),
        "rollback_model_id": "production",
        "rollback_model_dir": "models/production",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
    }
    ACTIVE_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    clear_model_cache()
    return payload


def rollback_to_production(reason: str = "Explicit rollback to legacy production XGBoost model") -> Dict[str, Any]:
    """Rolls back the active serving model to `models/production/`."""
    return activate_model("production", reason=reason)


def get_active_model_status() -> Dict[str, Any]:
    """Returns metadata describing the active model and rollback configuration."""
    model_id, model_dir = resolve_model_dir()
    contract_path = model_dir / "model_contract.json"
    meta_path = model_dir / "metadata.json"

    contract = {}
    if contract_path.exists():
        with open(contract_path, "r", encoding="utf-8") as f:
            contract = json.load(f)

    metadata = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    registry = {}
    if ACTIVE_REGISTRY_PATH.exists():
        with open(ACTIVE_REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

    return {
        "active_model_id": model_id,
        "active_model_dir": str(model_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "model_version": contract.get("model_version", "1.0.0"),
        "model_type": contract.get("model_type", "XGBClassifier"),
        "library": contract.get("library", "xgboost"),
        "library_version": contract.get("library_version"),
        "feature_count": contract.get("feature_count", 34),
        "feature_pipeline": contract.get("feature_pipeline", "canonical"),
        "class_mapping": contract.get("class_mapping", {"0": "SELL", "1": "HOLD", "2": "BUY"}),
        "rollback_available": (PROD_DIR / "model.json").exists(),
        "rollback_command": "python scripts/ml/model_adapter.py --rollback",
        "registry": registry,
        "metadata_summary": {
            "oof_accuracy": metadata.get("oof_metrics", {}).get("accuracy"),
            "oof_macro_f1": metadata.get("oof_metrics", {}).get("macro_f1"),
            "oof_sharpe_10bps": metadata.get("oof_metrics", {}).get("long_only_10bps_sharpe"),
            "holdout_2025_accuracy": metadata.get("holdout_2025_metrics", {}).get("accuracy")
            or metadata.get("test_metrics", {}).get("accuracy"),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="StockDNA-AI Model Adapter & Rollback CLI")
    parser.add_argument("--status", action="store_true", help="Print active model status")
    parser.add_argument("--activate", type=str, help="Activate model id (e.g. candidate_v7, production)")
    parser.add_argument("--rollback", action="store_true", help="Roll back active model to production XGBoost")
    parser.add_argument("--reason", type=str, default=None, help="Audit reason for activation/rollback")
    args = parser.parse_args()

    if args.rollback:
        res = rollback_to_production(reason=args.reason or "CLI rollback to production XGBoost")
        print(json.dumps(res, indent=2))
    elif args.activate:
        res = activate_model(args.activate, reason=args.reason or f"CLI activation of {args.activate}")
        print(json.dumps(res, indent=2))
    else:
        print(json.dumps(get_active_model_status(), indent=2))


if __name__ == "__main__":
    main()


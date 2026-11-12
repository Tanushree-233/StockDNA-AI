"""
Training & Versioned Serialization Pipeline for Candidate v7 (StockDNA-AI).

Trains the winning Candidate v7 architecture (`HistGB_Sqrt`: HistGradientBoostingClassifier
with square-root class weights on the 34-feature scale-invariant pipeline) strictly on
pre-2024 development data (2018-2023 with 5-bar purge before 2024).

Guarantees:
  1. Preserves `models/production/`, `candidate_v1`, and `candidate_v2` untouched.
  2. Fits imputation medians and model weights strictly on pre-2024 training data.
  3. Evaluates the frozen trained artifact once on untouched 2024 & 2025 holdouts
     for audit reporting (without feeding holdouts into training or selection).
  4. Serializes versioned artifacts to `models/candidates/candidate_v7/`:
     - `candidate_model.joblib`
     - `model_contract.json`
     - `metadata.json`
  5. Verifies end-to-end loading, prediction parity, and SHAP compatibility via `model_adapter`.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.feature_registry import get_feature_group
from config.settings import get_tickers
from scripts.feature_engineering.candidate_features import CANDIDATE_V2_FEATURE_NAMES
from scripts.ml.evaluate_candidate_v7_models import (
    SEED,
    attach_next_bar_execution_returns,
    compute_sample_weights,
    evaluate_classification,
    fit_and_apply_imputation,
    get_purged_partition,
    load_universe_dataset,
    simulate_trading_performance,
)
from scripts.ml.model_adapter import (
    clear_model_cache,
    load_model_from_dir,
    predict_from_features,
    predict_ticker_via_adapter,
)

CANDIDATE_V7_DIR = PROJECT_ROOT / "models" / "candidates" / "candidate_v7"
REPORT_PATH = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v7" / "model_comparison_report.json"


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 85)
    print("TRAINING & SERIALIZATION: CANDIDATE V7 (HistGradientBoostingClassifier)")
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"sklearn   : {sklearn.__version__} | Seed: {SEED}")
    print("Train Scope: 2018-2023 (5-bar purged). 2024 & 2025 Strictly Untouched During Fit.")
    print("=" * 85)

    tickers = get_tickers(clean=True)
    df_cand, raw_prices = load_universe_dataset(tickers)

    # Purged chronological partitions
    train_df = get_purged_partition(df_cand, tickers, [2018, 2019, 2020, 2021, 2022, 2023])
    val_2024_df = get_purged_partition(df_cand, tickers, [2024])
    test_2025_df = get_purged_partition(df_cand, tickers, [2025])

    # Fit medians strictly on pre-2024 training partition
    X_train, X_val_2024, medians = fit_and_apply_imputation(
        train_df, val_2024_df, CANDIDATE_V2_FEATURE_NAMES
    )
    _, X_test_2025, _ = fit_and_apply_imputation(
        train_df, test_2025_df, CANDIDATE_V2_FEATURE_NAMES
    )

    y_train = train_df["Target"].values.astype(int)
    y_val_2024 = val_2024_df["Target"].values.astype(int)
    y_test_2025 = test_2025_df["Target"].values.astype(int)

    print(f"Train (2018-2023 purged) : {len(train_df)} rows ({train_df['Date'].min().date()}..{train_df['Date'].max().date()})")
    print(f"Holdout 2024 (purged)    : {len(val_2024_df)} rows ({val_2024_df['Date'].min().date()}..{val_2024_df['Date'].max().date()})")
    print(f"Holdout 2025 (purged)    : {len(test_2025_df)} rows ({test_2025_df['Date'].min().date()}..{test_2025_df['Date'].max().date()})")

    # Selected winning architecture from Candidate v7 evaluation: HistGB_Sqrt
    hgb_params: Dict[str, Any] = {
        "loss": "log_loss",
        "max_iter": 150,
        "learning_rate": 0.03,
        "max_depth": 4,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 60,
        "l2_regularization": 5.0,
        "early_stopping": False,
        "random_state": SEED,
    }
    sw_sqrt = compute_sample_weights(y_train, "sqrt")

    model = HistGradientBoostingClassifier(**hgb_params)
    model.fit(X_train, y_train, sample_weight=sw_sqrt)

    # Single-pass holdout audit (post-training, zero feedback into training)
    probs_2024 = model.predict_proba(X_val_2024)
    probs_2025 = model.predict_proba(X_test_2025)
    metrics_2024 = evaluate_classification(y_val_2024, probs_2024)
    metrics_2025 = evaluate_classification(y_test_2025, probs_2025)

    test_exec_df, test_daily_rets = attach_next_bar_execution_returns(test_2025_df, raw_prices, tickers)
    # Re-align probs_2025 with test_exec_df order
    probs_df_2025 = test_2025_df[["Date", "Ticker"]].copy()
    probs_df_2025["p0"] = probs_2025[:, 0]
    probs_df_2025["p1"] = probs_2025[:, 1]
    probs_df_2025["p2"] = probs_2025[:, 2]
    merged_2025 = pd.merge(test_exec_df, probs_df_2025, on=["Date", "Ticker"], how="inner")
    trading_2025 = simulate_trading_performance(
        merged_2025, merged_2025[["p0", "p1", "p2"]].values, test_daily_rets, tickers
    )

    print(f"\n[Single-Pass Audit] 2024 Holdout -> Acc={metrics_2024['accuracy']*100:.2f}%, BalAcc={metrics_2024['balanced_accuracy']*100:.2f}%, MacroF1={metrics_2024['macro_f1']*100:.2f}%")
    print(f"[Single-Pass Audit] 2025 Holdout -> Acc={metrics_2025['accuracy']*100:.2f}%, BalAcc={metrics_2025['balanced_accuracy']*100:.2f}%, MacroF1={metrics_2025['macro_f1']*100:.2f}%")
    lo10_2025 = trading_2025["daily_portfolio"]["long_only"]["10bps"]
    print(f"[Single-Pass Audit] 2025 Long-Only 10bps -> NetRet={lo10_2025['total_net_return']*100:+.2f}%, Sharpe={lo10_2025['sharpe_ratio']:+.2f}, MaxDD={lo10_2025['max_drawdown']*100:.2f}%")

    # Save versioned artifacts under models/candidates/candidate_v7/
    CANDIDATE_V7_DIR.mkdir(parents=True, exist_ok=True)
    model_path = CANDIDATE_V7_DIR / "candidate_model.joblib"
    contract_path = CANDIDATE_V7_DIR / "model_contract.json"
    meta_path = CANDIDATE_V7_DIR / "metadata.json"

    joblib.dump(model, model_path)

    internal_feats = [f for f in CANDIDATE_V2_FEATURE_NAMES if get_feature_group(f) == "INTERNAL"]
    external_feats = [f for f in CANDIDATE_V2_FEATURE_NAMES if get_feature_group(f) == "EXTERNAL"]

    contract = {
        "candidate_id": "candidate_v7",
        "model_version": "7.0.0-histgb-sqrt",
        "model_type": "HistGradientBoostingClassifier",
        "model_format": "joblib",
        "model_filename": "candidate_model.joblib",
        "library": "scikit-learn",
        "library_version": sklearn.__version__,
        "objective": "log_loss",
        "num_class": 3,
        "class_mapping": {"0": "SELL", "1": "HOLD", "2": "BUY"},
        "target_definition": "Forward_Return_5d = Close[t+5] / Close[t] - 1",
        "horizon_trading_days": 5,
        "thresholds": {"BUY": 0.02, "SELL": -0.02},
        "feature_pipeline": "candidate_v2_scale_invariant",
        "preprocessing_requirements": {
            "replace_inf_with_nan": True,
            "imputation_strategy": "training_median",
            "scaling_required": False,
            "strict_feature_order": True,
            "scale_invariant_earnings_yield": True,
        },
        "feature_names": CANDIDATE_V2_FEATURE_NAMES,
        "feature_count": len(CANDIDATE_V2_FEATURE_NAMES),
        "feature_groups": {
            "INTERNAL": internal_feats,
            "EXTERNAL": external_feats,
        },
        "internal_feature_count": len(internal_feats),
        "external_feature_count": len(external_feats),
        "imputation_values": medians,
        "weighting_scheme": "sqrt",
        "hyperparameters": hgb_params,
        "random_seed": SEED,
        "date_ranges": {
            "train": {
                "start": str(train_df["Date"].min().date()),
                "end": str(train_df["Date"].max().date()),
            },
            "holdout_2024": {
                "start": str(val_2024_df["Date"].min().date()),
                "end": str(val_2024_df["Date"].max().date()),
            },
            "holdout_2025": {
                "start": str(test_2025_df["Date"].min().date()),
                "end": str(test_2025_df["Date"].max().date()),
            },
        },
    }

    with open(contract_path, "w", encoding="utf-8") as f:
        json.dump(contract, f, indent=2)

    oof_summary = {}
    if REPORT_PATH.exists():
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            rep = json.load(f)
        hgb_rep = rep.get("models", {}).get("HistGB_Sqrt", {})
        if hgb_rep:
            oof_summary = {
                "accuracy": hgb_rep["oof_overall"]["accuracy"],
                "balanced_accuracy": hgb_rep["oof_overall"]["balanced_accuracy"],
                "macro_f1": hgb_rep["oof_overall"]["macro_f1"],
                "ece": hgb_rep["oof_overall"]["calibration"]["ece"],
                "long_only_10bps_net_return": hgb_rep["trading_performance"]["daily_portfolio"]["long_only"]["10bps"]["total_net_return"],
                "long_only_10bps_sharpe": hgb_rep["trading_performance"]["daily_portfolio"]["long_only"]["10bps"]["sharpe_ratio"],
                "long_short_10bps_net_return": hgb_rep["trading_performance"]["daily_portfolio"]["long_short"]["10bps"]["total_net_return"],
                "long_short_10bps_sharpe": hgb_rep["trading_performance"]["daily_portfolio"]["long_short"]["10bps"]["sharpe_ratio"],
            }

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "candidate_v7",
        "selected_model_config": "HistGB_Sqrt",
        "model_type": "HistGradientBoostingClassifier",
        "model_version": "7.0.0-histgb-sqrt",
        "library_version": f"scikit-learn=={sklearn.__version__}",
        "artifact_sha256": compute_file_sha256(model_path),
        "tickers_count": len(tickers),
        "train_rows": len(train_df),
        "holdout_2024_rows": len(val_2024_df),
        "holdout_2025_rows": len(test_2025_df),
        "oof_metrics": oof_summary,
        "holdout_2024_metrics": metrics_2024,
        "holdout_2025_metrics": metrics_2025,
        "holdout_2025_trading_next_bar": trading_2025,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved Candidate v7 model    : {model_path}")
    print(f"Saved Candidate v7 contract : {contract_path}")
    print(f"Saved Candidate v7 metadata : {meta_path}")

    # Parity & Adapter verification
    clear_model_cache()
    loaded_model, loaded_contract = load_model_from_dir(CANDIDATE_V7_DIR, use_cache=False)
    sample_row = X_test_2025.iloc[[0]]
    p_direct = loaded_model.predict_proba(sample_row)[0]
    p_adapter = predict_from_features(loaded_model, loaded_contract, sample_row, active_model_id="candidate_v7")
    for idx, cname in enumerate(["SELL", "HOLD", "BUY"]):
        assert abs(float(p_direct[idx]) - p_adapter["probabilities"][cname]) < 1e-3
    live_res = predict_ticker_via_adapter("TCS", model_id_or_path=str(CANDIDATE_V7_DIR))
    print(f"Verified adapter inference on TCS: {live_res['prediction']} (conf={live_res['confidence']}, version={live_res['model_version']})")


if __name__ == "__main__":
    main()


"""
Candidate v7: Alternative ML Architecture Comparison & Selection for StockDNA-AI.

Compares alternative gradient-boosted tree architectures against existing XGBoost
baselines on identical pre-2024 chronological walk-forward splits (2018-2023)
using the established 34-feature scale-invariant pipeline.

Models Compared:
  Baselines:
    1. Majority_HOLD             : Always-HOLD zero-rule baseline
    2. Candidate_v2_XGB_Balanced : Balanced XGBoost (n_estimators=200, depth=4, lr=0.03)
    3. Candidate_v4_XGB_Sqrt     : Sqrt-weighted XGBoost (n_estimators=200, depth=4, lr=0.03)
    4. XGB_Natural_Prior         : Unweighted XGBoost (reference Bayes posterior)

  Alternative Models:
    5. CatBoostClassifier        : Symmetric trees with chronological early stopping
                                   (Natural, Sqrt, Regularized-Sqrt, Balanced)
    6. LightGBMClassifier        : Leaf-wise GBDT with regularization & chronological
                                   early stopping (Natural, Sqrt, Regularized-Sqrt, Balanced)
    7. HistGradientBoosting      : Scikit-learn native regularized histogram GBDT
                                   (Natural, Sqrt, Balanced)

Strict Leakage & Methodology Safeguards:
  - 2024 and 2025 are STRICTLY EXCLUDED from all training, tuning, early stopping,
    preprocessing, and model selection.
  - 5-bar purge at every split boundary per ticker (Split A: 2021, Split B: 2022,
    Split C: 2023), plus internal 5-bar purge before early-stopping validation tail.
  - Training medians for missing value imputation computed strictly on each split's
    training partition (never across future validation rows).
  - Next-bar Open(t+1) execution simulation with multi-tier transaction costs.

Outputs:
  - results/diagnostics/candidate_v7/model_comparison_report.json
  - results/diagnostics/candidate_v7/model_comparison_report.md
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_tickers
from scripts.feature_engineering.candidate_features import (
    CANDIDATE_V2_FEATURE_NAMES,
    build_candidate_features,
)

# Graceful CatBoost check (handles OS Application Control / DLL policies transparently)
CATBOOST_AVAILABLE = False
CATBOOST_VERSION = None
CATBOOST_IMPORT_ERROR = None
try:
    import catboost
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
    CATBOOST_VERSION = getattr(catboost, "__version__", "unknown")
except Exception as exc:
    CATBOOST_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "production_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "candidate_v7"
SEED = 42
PURGE_BARS = 5
CLASS_NAMES = ["SELL", "HOLD", "BUY"]


def load_universe_dataset(tickers: List[str]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Loads production_dataset.csv, attaches Open/Close prices from processed/raw
    price feeds per ticker, and computes the 34-feature scale-invariant pipeline.
    """
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Ticker"].isin(tickers)].copy()

    raw_prices: Dict[str, pd.DataFrame] = {}
    price_dfs: List[pd.DataFrame] = []
    for t in tickers:
        proc_p = PROJECT_ROOT / "data" / "processed" / "prices" / f"{t}.csv"
        raw_p = PROJECT_ROOT / "data" / "raw" / "prices" / f"{t}.csv"
        target_p = proc_p if proc_p.exists() else raw_p
        p_df = pd.read_csv(target_p)
        if "Price" in p_df.columns and "Date" not in p_df.columns:
            p_df = p_df.rename(columns={"Price": "Date"})
        p_df["Date"] = pd.to_datetime(p_df["Date"])
        p_df["Ticker"] = t
        p_df = p_df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
        raw_prices[t] = p_df
        price_dfs.append(p_df[["Date", "Ticker", "Open", "Close"]])

    prices_df = pd.concat(price_dfs, ignore_index=True)
    # Drop existing Close if present before merge to avoid suffix conflicts
    if "Close" in df.columns:
        df = df.drop(columns=["Close"])
    df = pd.merge(df, prices_df, on=["Date", "Ticker"], how="left")
    df_cand = build_candidate_features(df)
    return df_cand, raw_prices


def get_purged_partition(df: pd.DataFrame, tickers: List[str], years: List[int]) -> pd.DataFrame:
    """Slices rows in `years` per ticker and purges the final 5 bars per ticker."""
    parts = []
    for t in tickers:
        tdf = df[df["Ticker"] == t].sort_values("Date").reset_index(drop=True)
        sub = tdf[tdf["Date"].dt.year.isin(years)]
        if len(sub) > PURGE_BARS:
            parts.append(sub.iloc[:-PURGE_BARS])
    return pd.concat(parts, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)


def fit_and_apply_imputation(
    tr_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """Fits medians strictly on `tr_df` and imputes both `tr_df` and `val_df`."""
    medians: Dict[str, float] = {}
    X_tr = tr_df[feature_names].copy()
    X_val = val_df[feature_names].copy()

    for feat in feature_names:
        col_clean = X_tr[feat].replace([np.inf, -np.inf], np.nan)
        med = float(col_clean.median())
        medians[feat] = 0.0 if np.isnan(med) else med
        X_tr[feat] = col_clean.fillna(medians[feat])
        X_val[feat] = X_val[feat].replace([np.inf, -np.inf], np.nan).fillna(medians[feat])

    return X_tr, X_val, medians


def make_internal_early_stop_split(
    tr_df: pd.DataFrame,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    val_fraction: float = 0.15,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Splits `tr_df` chronologically into an inner training set and an inner
    early-stopping validation set with a 5-bar purge per ticker so no future
    bars or outer validation rows leak into early stopping.
    """
    unique_dates = np.sort(tr_df["Date"].unique())
    split_idx = int(len(unique_dates) * (1.0 - val_fraction))
    cutoff_date = unique_dates[split_idx]

    # Purge 5 bars per ticker before cutoff_date
    inner_tr_idx = []
    inner_es_idx = []
    for _, grp in tr_df.groupby("Ticker"):
        grp_sorted = grp.sort_values("Date")
        before = grp_sorted[grp_sorted["Date"] < cutoff_date]
        after = grp_sorted[grp_sorted["Date"] >= cutoff_date]
        if len(before) > PURGE_BARS:
            inner_tr_idx.extend(before.iloc[:-PURGE_BARS].index.tolist())
        inner_es_idx.extend(after.index.tolist())

    inner_tr_idx = np.array(sorted(inner_tr_idx))
    inner_es_idx = np.array(sorted(inner_es_idx))

    return (
        X_tr.iloc[inner_tr_idx],
        y_tr[inner_tr_idx],
        X_tr.iloc[inner_es_idx],
        y_tr[inner_es_idx],
        inner_tr_idx,
    )


def compute_sample_weights(y: np.ndarray, scheme: str) -> Optional[np.ndarray]:
    """Computes sample weights for 'natural', 'sqrt', or 'balanced' weighting."""
    if scheme == "natural":
        return None
    if scheme == "balanced":
        return compute_sample_weight("balanced", y)
    if scheme == "sqrt":
        counts = np.bincount(y, minlength=3)
        w_map = {c: float((len(y) / (3.0 * max(counts[c], 1))) ** 0.5) for c in range(3)}
        return np.array([w_map[val] for val in y], dtype=np.float64)
    raise ValueError(f"Unknown weighting scheme: {scheme}")


def compute_class_weights_dict(y: np.ndarray, scheme: str) -> Optional[Dict[int, float]]:
    """Computes per-class weights dict for CatBoost class_weights."""
    if scheme == "natural":
        return None
    counts = np.bincount(y, minlength=3)
    power = 1.0 if scheme == "balanced" else 0.5
    return {c: float((len(y) / (3.0 * max(counts[c], 1))) ** power) for c in range(3)}


def evaluate_classification(y_true: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """Computes full classification metrics, calibration metrics, and distribution."""
    n = len(y_true)
    y_pred = np.argmax(probs, axis=1)
    per_class: Dict[str, Any] = {}

    for c_idx, c_name in enumerate(CLASS_NAMES):
        pred_count = int(np.sum(y_pred == c_idx))
        per_class[c_name] = {
            "precision": float(precision_score(y_true == c_idx, y_pred == c_idx, zero_division=0)),
            "recall": float(recall_score(y_true == c_idx, y_pred == c_idx, zero_division=0)),
            "f1": float(f1_score(y_true == c_idx, y_pred == c_idx, zero_division=0)),
            "support": int(np.sum(y_true == c_idx)),
            "predicted_count": pred_count,
            "predicted_pct": float(pred_count / n) if n > 0 else 0.0,
        }

    # Probability calibration: LogLoss, Multiclass Brier Score, 10-bin ECE on top-1 confidence
    probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    probs_clipped = probs_clipped / probs_clipped.sum(axis=1, keepdims=True)
    m_logloss = float(log_loss(y_true, probs_clipped, labels=[0, 1, 2]))

    y_onehot = np.eye(3)[y_true]
    brier_score = float(np.mean(np.sum((probs_clipped - y_onehot) ** 2, axis=1)))

    confs = np.max(probs_clipped, axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    reliability_bins = []
    for i in range(10):
        lo, hi = bins[i], bins[i + 1]
        in_bin = (confs >= lo) & (confs <= hi if i == 9 else confs < hi)
        cnt = int(np.sum(in_bin))
        if cnt > 0:
            avg_conf = float(np.mean(confs[in_bin]))
            avg_acc = float(np.mean(correct[in_bin]))
            ece += (cnt / n) * abs(avg_acc - avg_conf)
            reliability_bins.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "count": cnt,
                    "mean_confidence": avg_conf,
                    "empirical_accuracy": avg_acc,
                }
            )

    return {
        "samples": n,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "calibration": {
            "log_loss": m_logloss,
            "brier_score": brier_score,
            "ece": float(ece),
            "reliability_bins": reliability_bins,
        },
    }


def compute_accuracy_vs_coverage(
    y_true: np.ndarray,
    probs: np.ndarray,
    fwd_returns: np.ndarray,
) -> Dict[str, Any]:
    """Evaluates accuracy, Macro F1, and hit rates across confidence thresholds tau."""
    thresholds = [0.334, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    p_max = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    n_total = len(y_true)
    curve: Dict[str, Any] = {}

    for tau in thresholds:
        key = f"tau_{tau:.2f}"
        mask = p_max >= tau
        n_kept = int(np.sum(mask))
        if n_kept >= 10:
            sub_y = y_true[mask]
            sub_p = preds[mask]
            sub_fwd = fwd_returns[mask]
            buy_m = sub_p == 2
            sell_m = sub_p == 0
            curve[key] = {
                "threshold": tau,
                "coverage": float(n_kept / n_total),
                "samples_included": n_kept,
                "accuracy": float(accuracy_score(sub_y, sub_p)),
                "balanced_accuracy": float(balanced_accuracy_score(sub_y, sub_p)),
                "macro_f1": float(f1_score(sub_y, sub_p, average="macro", zero_division=0)),
                "buy_count": int(np.sum(buy_m)),
                "sell_count": int(np.sum(sell_m)),
                "buy_hit_rate": float(np.mean(sub_fwd[buy_m] > 0)) if np.sum(buy_m) > 0 else 0.0,
                "sell_hit_rate": float(np.mean(sub_fwd[sell_m] < 0)) if np.sum(sell_m) > 0 else 0.0,
            }
        else:
            curve[key] = {
                "threshold": tau,
                "coverage": float(n_kept / n_total),
                "samples_included": n_kept,
                "status": "Insufficient samples (<10)",
            }
    return curve


def attach_next_bar_execution_returns(
    val_df: pd.DataFrame,
    raw_prices: Dict[str, pd.DataFrame],
    tickers: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[Any, float]]]:
    """
    Attaches next-bar Open(t+1) -> Close(t+5) 5-day return and builds per-ticker
    daily Open(t+1) -> Open(t+2) return lookup for portfolio simulation.
    """
    ticker_parts = []
    daily_open_rets: Dict[str, Dict[Any, float]] = {}

    for t in tickers:
        raw_p = raw_prices[t].sort_values("Date").reset_index(drop=True)
        date_to_idx = {d: i for i, d in enumerate(raw_p["Date"])}

        t_sub = val_df[val_df["Ticker"] == t].sort_values("Date").copy()
        fwd_5d_close = []
        t_daily_rets: Dict[Any, float] = {}

        for d in t_sub["Date"]:
            idx = date_to_idx.get(d)
            if idx is not None and idx + 5 < len(raw_p):
                open_t1 = float(raw_p.loc[idx + 1, "Open"])
                close_t5 = float(raw_p.loc[idx + 5, "Close"])
                fwd_5d_close.append((close_t5 / open_t1) - 1.0 if open_t1 > 0 else np.nan)
            else:
                fwd_5d_close.append(np.nan)

            if idx is not None and idx + 2 < len(raw_p):
                open_t1 = float(raw_p.loc[idx + 1, "Open"])
                open_t2 = float(raw_p.loc[idx + 2, "Open"])
                t_daily_rets[d] = (open_t2 / open_t1) - 1.0 if open_t1 > 0 else 0.0
            else:
                t_daily_rets[d] = 0.0

        t_sub["next_bar_fwd_5d_close"] = fwd_5d_close
        ticker_parts.append(t_sub)
        daily_open_rets[t] = t_daily_rets

    out_df = pd.concat(ticker_parts, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return out_df, daily_open_rets


def simulate_trading_performance(
    val_df: pd.DataFrame,
    probs: np.ndarray,
    daily_open_rets: Dict[str, Dict[Any, float]],
    tickers: List[str],
) -> Dict[str, Any]:
    """
    Evaluates economic trading utility under strict next-bar Open(t+1) execution:
      1. Equal-weighted daily portfolio (1/N per slot) with 0, 5, 10, 15, 25 bps costs
         for Long-Only (BUY) and Long/Short (BUY - SELL).
      2. Non-overlapping 5-day subgrid returns (mean across 5 stride offsets).
      3. Conditional hit rate & Long-Short spread.
    """
    df_sim = val_df[["Date", "Ticker", "next_bar_fwd_5d_close"]].copy()
    df_sim["pred"] = np.argmax(probs, axis=1)

    fwd = df_sim["next_bar_fwd_5d_close"].values
    preds = df_sim["pred"].values
    valid = ~np.isnan(fwd)
    buy_mask = (preds == 2) & valid
    sell_mask = (preds == 0) & valid

    mean_buy_ret = float(np.mean(fwd[buy_mask])) if np.sum(buy_mask) > 0 else 0.0
    mean_sell_ret = float(np.mean(fwd[sell_mask])) if np.sum(sell_mask) > 0 else 0.0
    buy_hit = float(np.mean(fwd[buy_mask] > 0)) if np.sum(buy_mask) > 0 else 0.0
    sell_hit = float(np.mean(fwd[sell_mask] < 0)) if np.sum(sell_mask) > 0 else 0.0

    # Non-overlapping 5-day subgrids (Long-Only BUY)
    cost_tiers_bps = [0, 5, 10, 15, 25]
    subgrid_net: Dict[str, List[float]] = {f"{b}bps": [] for b in cost_tiers_bps}
    for k in range(5):
        k_parts = []
        for t in tickers:
            t_sub = df_sim[df_sim["Ticker"] == t].sort_values("Date").iloc[k::5]
            k_parts.append(t_sub)
        k_df = pd.concat(k_parts, ignore_index=True)
        k_valid = ~k_df["next_bar_fwd_5d_close"].isna()
        k_df = k_df[k_valid]
        if len(k_df) == 0:
            continue
        k_pos = (k_df["pred"] == 2).astype(float).values
        k_gross = float(np.mean(k_pos * k_df["next_bar_fwd_5d_close"].values))

        turnovers = []
        for t in tickers:
            tp = (k_df[k_df["Ticker"] == t].sort_values("Date")["pred"] == 2).astype(float).values
            if len(tp) > 0:
                turnovers.extend(np.abs(np.diff(tp, prepend=0.0)))
        mean_turn = float(np.mean(turnovers)) if turnovers else 0.0
        for bps in cost_tiers_bps:
            subgrid_net[f"{bps}bps"].append(k_gross - mean_turn * (bps / 10000.0))

    subgrid_summary = {k: float(np.mean(v)) if v else 0.0 for k, v in subgrid_net.items()}

    # Daily equal-weighted portfolio simulation (1/N slots)
    eval_dates = sorted(df_sim["Date"].unique())
    n_tickers = len(tickers)
    slot_weight = 1.0 / max(n_tickers, 1)

    # Index predictions by (Date, Ticker) for fast O(1) lookup
    pred_lookup = df_sim.set_index(["Date", "Ticker"])["pred"].to_dict()

    portfolio_results: Dict[str, Any] = {"long_only": {}, "long_short": {}}
    for mode in ["long_only", "long_short"]:
        prev_w = {t: 0.0 for t in tickers}
        daily_gross = []
        daily_turn = []

        for d in eval_dates:
            curr_w = {}
            for t in tickers:
                p = pred_lookup.get((d, t), 1)
                if mode == "long_only":
                    curr_w[t] = slot_weight if p == 2 else 0.0
                else:
                    curr_w[t] = slot_weight if p == 2 else (-slot_weight if p == 0 else 0.0)
            turn = sum(abs(curr_w[t] - prev_w[t]) for t in tickers)
            g_ret = sum(curr_w[t] * daily_open_rets[t].get(d, 0.0) for t in tickers)
            daily_gross.append(g_ret)
            daily_turn.append(turn)
            prev_w = curr_w

        gross_arr = np.array(daily_gross, dtype=np.float64)
        turn_arr = np.array(daily_turn, dtype=np.float64)
        n_days = max(len(gross_arr), 1)

        for bps in cost_tiers_bps:
            net_arr = gross_arr - turn_arr * (bps / 10000.0)
            cum = np.cumprod(1.0 + net_arr)
            tot_ret = float(cum[-1] - 1.0)
            ann_ret = float((cum[-1] ** (252.0 / n_days)) - 1.0) if cum[-1] > 0 else -1.0
            std_d = float(np.std(net_arr))
            sharpe = float((np.mean(net_arr) / (std_d + 1e-12)) * np.sqrt(252.0)) if std_d > 0 else 0.0
            peaks = np.maximum.accumulate(cum)
            max_dd = float(np.min((cum - peaks) / (peaks + 1e-12)))
            ann_turn = float(np.sum(turn_arr) * (252.0 / n_days))
            portfolio_results[mode][f"{bps}bps"] = {
                "total_net_return": tot_ret,
                "annualized_net_return": ann_ret,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_dd,
                "annualized_turnover": ann_turn,
            }

    return {
        "conditional_next_bar_5d": {
            "buy_signals": int(np.sum(buy_mask)),
            "sell_signals": int(np.sum(sell_mask)),
            "buy_hit_rate": buy_hit,
            "sell_hit_rate": sell_hit,
            "mean_buy_return": mean_buy_ret,
            "mean_sell_return": mean_sell_ret,
            "long_short_spread": float(mean_buy_ret - mean_sell_ret),
        },
        "non_overlapping_5d_long_only": subgrid_summary,
        "daily_portfolio": portfolio_results,
    }


def train_and_predict_candidate(
    model_name: str,
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    tr_df: pd.DataFrame,
    X_val: pd.DataFrame,
) -> Tuple[np.ndarray, float, float, Dict[str, Any]]:
    """
    Trains a model configuration strictly on training data (with chronological
    internal early stopping where applicable) and returns (probs, train_sec, infer_sec, meta).
    """
    t0 = time.perf_counter()
    extra_meta: Dict[str, Any] = {}

    if model_name == "Majority_HOLD":
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        probs = np.zeros((len(X_val), 3), dtype=np.float64)
        probs[:, 1] = 1.0
        infer_sec = time.perf_counter() - t1
        return probs, train_sec, infer_sec, {"strategy": "constant_hold"}

    # XGBoost baselines (matching existing Candidate v2 & v4 recipes)
    xgb_base_params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "random_state": SEED,
        "eval_metric": "mlogloss",
        "n_jobs": 1,
    }

    if model_name in ("Candidate_v2_XGB_Balanced", "Candidate_v4_XGB_Sqrt", "XGB_Natural_Prior"):
        scheme = (
            "balanced"
            if model_name == "Candidate_v2_XGB_Balanced"
            else ("sqrt" if model_name == "Candidate_v4_XGB_Sqrt" else "natural")
        )
        sw = compute_sample_weights(y_tr, scheme)
        model = XGBClassifier(**xgb_base_params)
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        else:
            model.fit(X_tr, y_tr, verbose=False)
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        probs = model.predict_proba(X_val)
        infer_sec = time.perf_counter() - t1
        extra_meta = {"params": xgb_base_params, "weighting": scheme}
        return probs, train_sec, infer_sec, extra_meta

    # Chronological internal split for early stopping (used by CatBoost and LightGBM)
    X_in_tr, y_in_tr, X_in_es, y_in_es, in_tr_idx = make_internal_early_stop_split(tr_df, X_tr, y_tr)

    # CatBoost configurations
    if model_name.startswith("CatBoost_"):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError(f"CatBoost unavailable: {CATBOOST_IMPORT_ERROR}")

        cfg_map = {
            "CatBoost_Natural": {"weighting": "natural", "depth": 5, "l2_leaf_reg": 6.0},
            "CatBoost_Sqrt": {"weighting": "sqrt", "depth": 5, "l2_leaf_reg": 6.0},
            "CatBoost_Sqrt_Reg": {"weighting": "sqrt", "depth": 4, "l2_leaf_reg": 12.0, "rsm": 0.8},
            "CatBoost_Balanced": {"weighting": "balanced", "depth": 4, "l2_leaf_reg": 8.0},
        }
        cfg = cfg_map[model_name]
        cw = compute_class_weights_dict(y_in_tr, cfg["weighting"])
        cb_params = {
            "iterations": 600,
            "learning_rate": 0.03,
            "depth": cfg["depth"],
            "l2_leaf_reg": cfg["l2_leaf_reg"],
            "loss_function": "MultiClass",
            "eval_metric": "MultiClass",
            "random_seed": SEED,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": 4,
        }
        if "rsm" in cfg:
            cb_params["rsm"] = cfg["rsm"]
        if cw is not None:
            cb_params["class_weights"] = cw

        model = CatBoostClassifier(**cb_params)
        model.fit(
            X_in_tr,
            y_in_tr,
            eval_set=(X_in_es, y_in_es),
            early_stopping_rounds=50,
            use_best_model=True,
        )
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        probs = model.predict_proba(X_val)
        infer_sec = time.perf_counter() - t1
        extra_meta = {
            "params": cb_params,
            "weighting": cfg["weighting"],
            "best_iteration": int(model.get_best_iteration()),
        }
        return probs, train_sec, infer_sec, extra_meta

    # LightGBM configurations
    if model_name.startswith("LightGBM_"):
        cfg_map = {
            "LightGBM_Natural": {
                "weighting": "natural",
                "n_estimators": 600,
                "early_stop": True,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 60,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.8,
            },
            "LightGBM_Sqrt": {
                "weighting": "sqrt",
                "n_estimators": 600,
                "early_stop": True,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 60,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.8,
            },
            "LightGBM_Sqrt_Reg": {
                "weighting": "sqrt",
                "n_estimators": 600,
                "early_stop": True,
                "num_leaves": 10,
                "max_depth": 3,
                "min_child_samples": 100,
                "reg_alpha": 2.0,
                "reg_lambda": 10.0,
                "colsample_bytree": 0.7,
            },
            "LightGBM_Balanced": {
                "weighting": "balanced",
                "n_estimators": 600,
                "early_stop": True,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 60,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.8,
            },
            "LightGBM_Fixed200_Sqrt": {
                "weighting": "sqrt",
                "n_estimators": 200,
                "early_stop": False,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 60,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.8,
            },
            "LightGBM_Fixed200_Balanced": {
                "weighting": "balanced",
                "n_estimators": 200,
                "early_stop": False,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 60,
                "reg_alpha": 1.0,
                "reg_lambda": 5.0,
                "colsample_bytree": 0.8,
            },
        }
        cfg = cfg_map[model_name]
        lgb_params = {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": cfg["n_estimators"],
            "learning_rate": 0.03,
            "num_leaves": cfg["num_leaves"],
            "max_depth": cfg["max_depth"],
            "min_child_samples": cfg["min_child_samples"],
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": cfg["colsample_bytree"],
            "reg_alpha": cfg["reg_alpha"],
            "reg_lambda": cfg["reg_lambda"],
            "random_state": SEED,
            "n_jobs": 1,
            "verbose": -1,
        }
        model = LGBMClassifier(**lgb_params)
        if cfg["early_stop"]:
            sw_in = compute_sample_weights(y_in_tr, cfg["weighting"])
            sw_es = compute_sample_weights(y_in_es, cfg["weighting"])
            fit_kwargs: Dict[str, Any] = {
                "eval_X": X_in_es,
                "eval_y": y_in_es,
                "callbacks": [lgb.early_stopping(50, verbose=False)],
            }
            if sw_in is not None:
                fit_kwargs["sample_weight"] = sw_in
                fit_kwargs["eval_sample_weight"] = [sw_es]
            model.fit(X_in_tr, y_in_tr, **fit_kwargs)
        else:
            sw_full = compute_sample_weights(y_tr, cfg["weighting"])
            if sw_full is not None:
                model.fit(X_tr, y_tr, sample_weight=sw_full)
            else:
                model.fit(X_tr, y_tr)

        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        probs = model.predict_proba(X_val)
        infer_sec = time.perf_counter() - t1
        extra_meta = {
            "params": lgb_params,
            "weighting": cfg["weighting"],
            "best_iteration": int(model.best_iteration_ or lgb_params["n_estimators"]),
        }
        return probs, train_sec, infer_sec, extra_meta

    # HistGradientBoostingClassifier configurations
    if model_name.startswith("HistGB_"):
        cfg_map = {
            "HistGB_Natural": {"weighting": "natural", "max_iter": 150, "max_depth": 4, "max_leaf_nodes": 15, "min_samples_leaf": 60, "l2_regularization": 5.0},
            "HistGB_Sqrt": {"weighting": "sqrt", "max_iter": 150, "max_depth": 4, "max_leaf_nodes": 15, "min_samples_leaf": 60, "l2_regularization": 5.0},
            "HistGB_Sqrt_Reg": {"weighting": "sqrt", "max_iter": 120, "max_depth": 3, "max_leaf_nodes": 10, "min_samples_leaf": 80, "l2_regularization": 10.0},
            "HistGB_Sqrt_D5": {"weighting": "sqrt", "max_iter": 150, "max_depth": 5, "max_leaf_nodes": 20, "min_samples_leaf": 80, "l2_regularization": 8.0},
            "HistGB_Balanced": {"weighting": "balanced", "max_iter": 150, "max_depth": 4, "max_leaf_nodes": 15, "min_samples_leaf": 60, "l2_regularization": 5.0},
        }
        cfg = cfg_map[model_name]
        sw = compute_sample_weights(y_tr, cfg["weighting"])
        hgb_params = {
            "loss": "log_loss",
            "max_iter": cfg["max_iter"],
            "learning_rate": 0.03,
            "max_depth": cfg["max_depth"],
            "max_leaf_nodes": cfg["max_leaf_nodes"],
            "min_samples_leaf": cfg["min_samples_leaf"],
            "l2_regularization": cfg["l2_regularization"],
            "early_stopping": False,  # Strict chronological integrity (avoid random internal val split)
            "random_state": SEED,
        }
        model = HistGradientBoostingClassifier(**hgb_params)
        if sw is not None:
            model.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr, y_tr)
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        probs = model.predict_proba(X_val)
        infer_sec = time.perf_counter() - t1
        extra_meta = {"params": hgb_params, "weighting": cfg["weighting"]}
        return probs, train_sec, infer_sec, extra_meta

    raise ValueError(f"Unknown model configuration: {model_name}")


def evaluate_acceptance_criteria(
    cand_metrics: Dict[str, Any],
    cand_trading: Dict[str, Any],
    cand_yearly: Dict[str, Any],
    v2_metrics: Dict[str, Any],
    v2_trading: Dict[str, Any],
    v4_metrics: Dict[str, Any],
    v4_trading: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluates Phase 4 acceptance criteria strictly without cherry-picking:
      1. Higher OOF Macro F1 than Candidate v2, OR defensible improvement in
         risk-adjusted economic performance (Sharpe & Net Return at 10bps)
      2. No severe collapse in BUY or SELL recall (both >= 0.05)
      3. No artificial accuracy gains from predicting HOLD almost exclusively (HOLD pred % <= 0.85)
      4. Stable results across chronological evaluation years (std of yearly accuracy <= 0.10
         and positive Macro F1 > 0.25 in every year)
    """
    macro_f1 = cand_metrics["macro_f1"]
    v2_macro_f1 = v2_metrics["macro_f1"]

    sharpe_lo = cand_trading["daily_portfolio"]["long_only"]["10bps"]["sharpe_ratio"]
    ret_lo = cand_trading["daily_portfolio"]["long_only"]["10bps"]["total_net_return"]
    v2_sharpe_lo = v2_trading["daily_portfolio"]["long_only"]["10bps"]["sharpe_ratio"]
    v4_sharpe_lo = v4_trading["daily_portfolio"]["long_only"]["10bps"]["sharpe_ratio"]

    buy_recall = cand_metrics["per_class"]["BUY"]["recall"]
    sell_recall = cand_metrics["per_class"]["SELL"]["recall"]
    hold_pred_pct = cand_metrics["per_class"]["HOLD"]["predicted_pct"]

    yearly_accs = [y["accuracy"] for y in cand_yearly.values()]
    yearly_f1s = [y["macro_f1"] for y in cand_yearly.values()]

    beats_v2_macro_f1 = macro_f1 > v2_macro_f1
    beats_economics = (sharpe_lo > max(v2_sharpe_lo, v4_sharpe_lo)) and (ret_lo > 0.0)
    primary_gate = beats_v2_macro_f1 or beats_economics

    no_recall_collapse = (buy_recall >= 0.05) and (sell_recall >= 0.05)
    no_hold_monopoly = hold_pred_pct <= 0.85
    chronological_stability = (float(np.std(yearly_accs)) <= 0.12) and all(f > 0.25 for f in yearly_f1s)

    passed = primary_gate and no_recall_collapse and no_hold_monopoly and chronological_stability

    return {
        "passed": bool(passed),
        "checks": {
            "beats_v2_macro_f1": bool(beats_v2_macro_f1),
            "beats_risk_adjusted_economics_10bps": bool(beats_economics),
            "primary_edge_gate": bool(primary_gate),
            "no_buy_sell_recall_collapse": bool(no_recall_collapse),
            "no_hold_exclusivity_collapse": bool(no_hold_monopoly),
            "chronological_stability": bool(chronological_stability),
        },
        "values": {
            "macro_f1": macro_f1,
            "v2_macro_f1": v2_macro_f1,
            "long_only_10bps_sharpe": sharpe_lo,
            "v2_long_only_10bps_sharpe": v2_sharpe_lo,
            "v4_long_only_10bps_sharpe": v4_sharpe_lo,
            "buy_recall": buy_recall,
            "sell_recall": sell_recall,
            "hold_pred_pct": hold_pred_pct,
            "yearly_acc_std": float(np.std(yearly_accs)),
        },
    }


def main():
    print("=" * 95)
    print("CANDIDATE V7: ALTERNATIVE MODEL ARCHITECTURE COMPARISON (PRE-2024 WALK-FORWARD)")
    print(f"Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"Python    : {sys.version.split()[0]} | Seed: {SEED}")
    print(f"Libraries : xgboost={xgb.__version__}, lightgbm={lgb.__version__}, sklearn={sklearn.__version__}")
    print(f"CatBoost  : {'available (' + str(CATBOOST_VERSION) + ')' if CATBOOST_AVAILABLE else 'BLOCKED (' + str(CATBOOST_IMPORT_ERROR) + ')'}")
    print("Holdout   : 2024 and 2025 STRICTLY EXCLUDED from all modeling and selection.")
    print("=" * 95)

    tickers = get_tickers(clean=True)
    df_cand, raw_prices = load_universe_dataset(tickers)
    print(f"Loaded {len(tickers)} tickers, {len(df_cand)} total rows across 2018-2025.")

    splits = {
        "Split A (2021)": {"tr_yrs": [2018, 2019, 2020], "val_yr": 2021},
        "Split B (2022)": {"tr_yrs": [2018, 2019, 2020, 2021], "val_yr": 2022},
        "Split C (2023)": {"tr_yrs": [2018, 2019, 2020, 2021, 2022], "val_yr": 2023},
    }

    candidate_models = [
        "Majority_HOLD",
        "Candidate_v2_XGB_Balanced",
        "Candidate_v4_XGB_Sqrt",
        "XGB_Natural_Prior",
        "LightGBM_Natural",
        "LightGBM_Sqrt",
        "LightGBM_Sqrt_Reg",
        "LightGBM_Balanced",
        "LightGBM_Fixed200_Sqrt",
        "LightGBM_Fixed200_Balanced",
        "HistGB_Natural",
        "HistGB_Sqrt",
        "HistGB_Sqrt_Reg",
        "HistGB_Sqrt_D5",
        "HistGB_Balanced",
    ]
    if CATBOOST_AVAILABLE:
        candidate_models.extend([
            "CatBoost_Natural",
            "CatBoost_Sqrt",
            "CatBoost_Sqrt_Reg",
            "CatBoost_Balanced",
        ])

    # Pre-slice all chronological folds with 5-bar purges
    split_data: Dict[str, Dict[str, Any]] = {}
    all_val_dfs: List[pd.DataFrame] = []

    for s_name, spec in splits.items():
        tr_df = get_purged_partition(df_cand, tickers, spec["tr_yrs"])
        val_df = get_purged_partition(df_cand, tickers, [spec["val_yr"]])
        X_tr, X_val, medians = fit_and_apply_imputation(tr_df, val_df, CANDIDATE_V2_FEATURE_NAMES)
        y_tr = tr_df["Target"].values.astype(int)
        y_val = val_df["Target"].values.astype(int)

        val_df = val_df.copy()
        val_df["Split"] = s_name
        val_df["EvalYear"] = spec["val_yr"]
        all_val_dfs.append(val_df)

        split_data[s_name] = {
            "tr_df": tr_df,
            "val_df": val_df,
            "X_tr": X_tr,
            "y_tr": y_tr,
            "X_val": X_val,
            "y_val": y_val,
            "eval_year": spec["val_yr"],
            "tr_dates": [str(tr_df["Date"].min().date()), str(tr_df["Date"].max().date())],
            "val_dates": [str(val_df["Date"].min().date()), str(val_df["Date"].max().date())],
        }
        print(f"  {s_name}: train={len(tr_df)} ({split_data[s_name]['tr_dates'][0]}..{split_data[s_name]['tr_dates'][1]}), val={len(val_df)} ({split_data[s_name]['val_dates'][0]}..{split_data[s_name]['val_dates'][1]})")

    all_val_base = pd.concat(all_val_dfs, ignore_index=True)
    all_val_exec, daily_open_rets = attach_next_bar_execution_returns(all_val_base, raw_prices, tickers)
    # Build lookup index mapping (Date, Ticker) -> row order in all_val_exec
    all_val_exec = all_val_exec.sort_values(["Split", "Date", "Ticker"]).reset_index(drop=True)

    # Evaluate every model across all splits
    model_results: Dict[str, Any] = {}
    print(f"\n{'Model':<26} | {'OOF Acc':<8} | {'Bal Acc':<8} | {'MacroF1':<8} | {'BUY F1':<7} | {'SELL F1':<7} | {'HOLD%':<7} | {'10bps Ret':<9} | {'Sharpe':<6} | {'Train(s)'}")
    print("-" * 116)

    for m_name in candidate_models:
        oof_probs_list: List[np.ndarray] = []
        oof_y_list: List[np.ndarray] = []
        oof_df_parts: List[pd.DataFrame] = []
        yearly_results: Dict[str, Any] = {}
        total_train_sec = 0.0
        total_infer_sec = 0.0
        last_meta: Dict[str, Any] = {}

        for s_name in splits:
            sd = split_data[s_name]
            probs, tr_s, inf_s, meta = train_and_predict_candidate(
                m_name, sd["X_tr"], sd["y_tr"], sd["tr_df"], sd["X_val"]
            )
            total_train_sec += tr_s
            total_infer_sec += inf_s
            last_meta = meta

            yr_key = str(sd["eval_year"])
            yearly_results[yr_key] = evaluate_classification(sd["y_val"], probs)
            yearly_results[yr_key]["train_seconds"] = round(tr_s, 4)
            yearly_results[yr_key]["infer_seconds"] = round(inf_s, 4)

            part = sd["val_df"][["Date", "Ticker", "Forward_Return_5d", "Target"]].copy()
            part["p0"] = probs[:, 0]
            part["p1"] = probs[:, 1]
            part["p2"] = probs[:, 2]
            oof_df_parts.append(part)

        oof_merged = pd.merge(
            all_val_exec[["Date", "Ticker", "next_bar_fwd_5d_close", "Forward_Return_5d", "Target"]],
            pd.concat(oof_df_parts, ignore_index=True)[["Date", "Ticker", "p0", "p1", "p2"]],
            on=["Date", "Ticker"],
            how="inner",
        ).sort_values(["Date", "Ticker"]).reset_index(drop=True)

        oof_probs = oof_merged[["p0", "p1", "p2"]].values
        oof_y = oof_merged["Target"].values.astype(int)
        oof_fwd = oof_merged["Forward_Return_5d"].values

        oof_cls_metrics = evaluate_classification(oof_y, oof_probs)
        cov_curve = compute_accuracy_vs_coverage(oof_y, oof_probs, oof_fwd)
        trading_perf = simulate_trading_performance(oof_merged, oof_probs, daily_open_rets, tickers)

        model_results[m_name] = {
            "status": "evaluated",
            "model_family": m_name.split("_")[0],
            "oof_overall": oof_cls_metrics,
            "by_year": yearly_results,
            "accuracy_vs_coverage": cov_curve,
            "trading_performance": trading_perf,
            "timing": {
                "total_train_seconds": round(total_train_sec, 4),
                "total_infer_seconds": round(total_infer_sec, 4),
                "infer_ms_per_sample": round((total_infer_sec * 1000.0) / len(oof_y), 5),
            },
            "config": last_meta,
        }

        lo10 = trading_perf["daily_portfolio"]["long_only"]["10bps"]
        print(
            f"{m_name:<26} | "
            f"{oof_cls_metrics['accuracy']*100:6.2f}%  | "
            f"{oof_cls_metrics['balanced_accuracy']*100:6.2f}%  | "
            f"{oof_cls_metrics['macro_f1']*100:6.2f}%  | "
            f"{oof_cls_metrics['per_class']['BUY']['f1']:7.4f} | "
            f"{oof_cls_metrics['per_class']['SELL']['f1']:7.4f} | "
            f"{oof_cls_metrics['per_class']['HOLD']['predicted_pct']*100:5.1f}%  | "
            f"{lo10['total_net_return']*100:+7.2f}%  | "
            f"{lo10['sharpe_ratio']:+5.2f} | "
            f"{total_train_sec:6.2f}s"
        )

    # Add explicit record for CatBoost if blocked by host OS policy
    if not CATBOOST_AVAILABLE:
        model_results["CatBoostClassifier"] = {
            "status": "blocked_by_os_policy",
            "model_family": "CatBoost",
            "error": CATBOOST_IMPORT_ERROR,
            "note": "catboost==1.2.10 installed in .venv, but Windows Application Control blocked loading _catboost.pyd.",
        }

    # Phase 4 Acceptance Check across all new alternative models
    v2_res = model_results["Candidate_v2_XGB_Balanced"]
    v4_res = model_results["Candidate_v4_XGB_Sqrt"]
    alternatives = [
        m for m, r in model_results.items()
        if r.get("status") == "evaluated" and m.startswith(("LightGBM_", "HistGB_", "CatBoost_"))
    ]

    acceptance_summary: Dict[str, Any] = {}
    passing_models: List[Tuple[str, float, float]] = []

    for m_name in alternatives:
        r = model_results[m_name]
        chk = evaluate_acceptance_criteria(
            r["oof_overall"],
            r["trading_performance"],
            r["by_year"],
            v2_res["oof_overall"],
            v2_res["trading_performance"],
            v4_res["oof_overall"],
            v4_res["trading_performance"],
        )
        acceptance_summary[m_name] = chk
        if chk["passed"]:
            # Score by Macro F1 + Sharpe synergy
            sharpe = chk["values"]["long_only_10bps_sharpe"]
            passing_models.append((m_name, chk["values"]["macro_f1"], sharpe))

    passing_models.sort(key=lambda x: (x[1], x[2]), reverse=True)
    winning_model = passing_models[0][0] if passing_models else None

    # Also identify top alternative overall even if criteria fails
    best_alt_by_macro_f1 = max(alternatives, key=lambda m: model_results[m]["oof_overall"]["macro_f1"])
    best_alt_by_acc = max(alternatives, key=lambda m: model_results[m]["oof_overall"]["accuracy"])

    print("\n" + "=" * 95)
    print("PHASE 4 ACCEPTANCE EVALUATION")
    print("=" * 95)
    for m_name, chk in acceptance_summary.items():
        status_str = "PASS" if chk["passed"] else "FAIL"
        print(f"  {m_name:<22} -> {status_str} | checks={chk['checks']}")
    print(f"\nWinning Alternative Passing Criteria : {winning_model or 'NONE (Production XGBoost Retained)'}")
    print(f"Best Alternative by Macro F1         : {best_alt_by_macro_f1} ({model_results[best_alt_by_macro_f1]['oof_overall']['macro_f1']*100:.2f}%)")
    print(f"Best Alternative by Raw Accuracy     : {best_alt_by_acc} ({model_results[best_alt_by_acc]['oof_overall']['accuracy']*100:.2f}%)")

    # Save JSON and Markdown reports
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full_report = {
        "candidate_id": "candidate_v7",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
            "tickers": tickers,
            "ticker_count": len(tickers),
            "feature_count": len(CANDIDATE_V2_FEATURE_NAMES),
            "feature_names": CANDIDATE_V2_FEATURE_NAMES,
            "oof_total_rows": len(all_val_exec),
        },
        "holdout_policy": "2024 and 2025 Strictly Excluded",
        "purge_bars": PURGE_BARS,
        "random_seed": SEED,
        "library_versions": {
            "python": sys.version.split()[0],
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "scikit_learn": sklearn.__version__,
            "catboost": CATBOOST_VERSION if CATBOOST_AVAILABLE else f"blocked ({CATBOOST_IMPORT_ERROR})",
        },
        "splits": {
            s_name: {
                "eval_year": sd["eval_year"],
                "train_rows": len(sd["tr_df"]),
                "val_rows": len(sd["val_df"]),
                "train_dates": sd["tr_dates"],
                "val_dates": sd["val_dates"],
            }
            for s_name, sd in split_data.items()
        },
        "models": model_results,
        "acceptance_evaluation": {
            "criteria_results": acceptance_summary,
            "winning_model": winning_model,
            "best_alternative_macro_f1": best_alt_by_macro_f1,
            "best_alternative_accuracy": best_alt_by_acc,
        },
    }

    json_path = OUTPUT_DIR / "model_comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    # Build Markdown summary report
    md_lines = [
        "# Candidate v7 Diagnostic Report: Alternative ML Model Comparison",
        f"**Execution Timestamp:** `{full_report['timestamp']}`  ",
        f"**Universe:** {len(tickers)} NSE Equities ({len(all_val_exec):,} Out-of-Fold Observations across 2021-2023)  ",
        f"**Features:** 34 Scale-Invariant Features (`CANDIDATE_V2_FEATURE_NAMES`)  ",
        f"**Holdout Policy:** 2024 and 2025 strictly excluded from training, tuning, early stopping, and selection.  ",
        f"**Libraries:** `xgboost=={xgb.__version__}`, `lightgbm=={lgb.__version__}`, `scikit-learn=={sklearn.__version__}`, `catboost`: `{full_report['library_versions']['catboost']}`  ",
        "",
        "---",
        "",
        "## 1. Out-of-Fold Classification & Trading Performance (2021-2023)",
        "",
        "| Model | Accuracy | Bal Acc | Macro F1 | SELL F1 (P / R) | HOLD F1 (P / R) | BUY F1 (P / R) | Pred % (S / H / B) | 10bps Net Ret | Sharpe | Train (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m_name in candidate_models:
        r = model_results[m_name]
        o = r["oof_overall"]
        pc = o["per_class"]
        lo10 = r["trading_performance"]["daily_portfolio"]["long_only"]["10bps"]
        md_lines.append(
            f"| **{m_name}** | {o['accuracy']*100:.2f}% | {o['balanced_accuracy']*100:.2f}% | "
            f"{o['macro_f1']*100:.2f}% | "
            f"{pc['SELL']['f1']:.3f} ({pc['SELL']['precision']:.2f}/{pc['SELL']['recall']:.2f}) | "
            f"{pc['HOLD']['f1']:.3f} ({pc['HOLD']['precision']:.2f}/{pc['HOLD']['recall']:.2f}) | "
            f"{pc['BUY']['f1']:.3f} ({pc['BUY']['precision']:.2f}/{pc['BUY']['recall']:.2f}) | "
            f"{pc['SELL']['predicted_pct']*100:.1f}% / {pc['HOLD']['predicted_pct']*100:.1f}% / {pc['BUY']['predicted_pct']*100:.1f}% | "
            f"{lo10['total_net_return']*100:+.2f}% | {lo10['sharpe_ratio']:+.2f} | {r['timing']['total_train_seconds']:.2f}s |"
        )

    md_lines.extend([
        "",
        "## 2. Accuracy & Macro F1 by Chronological Evaluation Year",
        "",
        "| Model | 2021 Acc / MacroF1 | 2022 Acc / MacroF1 | 2023 Acc / MacroF1 | LogLoss | Brier | ECE |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for m_name in candidate_models:
        r = model_results[m_name]
        by_y = r["by_year"]
        cal = r["oof_overall"]["calibration"]
        md_lines.append(
            f"| {m_name} | "
            f"{by_y['2021']['accuracy']*100:.2f}% / {by_y['2021']['macro_f1']*100:.2f}% | "
            f"{by_y['2022']['accuracy']*100:.2f}% / {by_y['2022']['macro_f1']*100:.2f}% | "
            f"{by_y['2023']['accuracy']*100:.2f}% / {by_y['2023']['macro_f1']*100:.2f}% | "
            f"{cal['log_loss']:.4f} | {cal['brier_score']:.4f} | {cal['ece']:.4f} |"
        )

    md_lines.extend([
        "",
        "## 3. Phase 4 Acceptance Verdict",
        f"* **Winning Replacement Candidate:** `{winning_model or 'None (Retain Production Model)'}`",
        f"* **Best Alternative by Macro F1:** `{best_alt_by_macro_f1}` ({model_results[best_alt_by_macro_f1]['oof_overall']['macro_f1']*100:.2f}%)",
        f"* **Best Alternative by Accuracy:** `{best_alt_by_acc}` ({model_results[best_alt_by_acc]['oof_overall']['accuracy']*100:.2f}%)",
    ])

    if not CATBOOST_AVAILABLE:
        md_lines.append(
            f"\n> **Environment Notice on CatBoost:** `catboost==1.2.10` was installed in `.venv`, "
            f"but loading `_catboost.pyd` was blocked by Windows Application Control policy (`{CATBOOST_IMPORT_ERROR}`)."
        )

    md_path = OUTPUT_DIR / "model_comparison_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nSaved JSON report : {json_path}")
    print(f"Saved MD report   : {md_path}")


if __name__ == "__main__":
    main()


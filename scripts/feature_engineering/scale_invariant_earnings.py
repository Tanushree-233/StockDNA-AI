"""
Scale-Invariant Earnings Features Module for StockDNA-AI.

Transforms nominal raw earnings features (Reported_EPS, EPS_Estimate)
into scale-independent point-in-time valuation metrics:
1. Earnings_Yield = Reported_EPS / Close
2. Forward_Earnings_Yield = EPS_Estimate / Close

Guarantees:
- Zero nominal price contamination (scale-invariant under stock splits and price scale).
- Zero look-ahead (uses strictly current Close and latest point-in-time announced EPS).
- Isolated module: Does not alter production contracts or existing model artifacts.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_scale_invariant_earnings_features(
    df: pd.DataFrame,
    price_col: str = "Close",
    reported_eps_col: str = "Reported_EPS",
    eps_estimate_col: str = "EPS_Estimate",
) -> pd.DataFrame:
    """
    Computes scale-independent earnings yield ratios.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least `price_col` and the nominal EPS columns.
    price_col : str
        Name of the closing price column (default: "Close").
    reported_eps_col : str
        Name of the reported EPS column (default: "Reported_EPS").
    eps_estimate_col : str
        Name of the estimated EPS column (default: "EPS_Estimate").

    Returns
    -------
    pd.DataFrame with added 'Earnings_Yield' and 'Forward_Earnings_Yield' columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    close = data[price_col] if price_col in data.columns else pd.Series(1.0, index=data.index)
    close_safe = close.replace(0.0, np.nan).fillna(1.0).abs() + 1e-8

    # 1. Earnings Yield: Annualized/Quarterly Trailing Earnings / Current Price
    if reported_eps_col in data.columns:
        reported_eps = data[reported_eps_col].fillna(0.0)
        data["Earnings_Yield"] = reported_eps / close_safe
    else:
        data["Earnings_Yield"] = 0.0

    # 2. Forward Earnings Yield: Estimated Earnings / Current Price
    if eps_estimate_col in data.columns:
        eps_est = data[eps_estimate_col].fillna(0.0)
        data["Forward_Earnings_Yield"] = eps_est / close_safe
    else:
        data["Forward_Earnings_Yield"] = 0.0

    return data


def get_scale_invariant_feature_matrix(
    df: pd.DataFrame,
    base_feature_names: List[str],
    replace_nominal_eps: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Constructs an updated feature matrix where nominal EPS columns
    are replaced with scale-invariant earnings yields.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the base features and 'Close'.
    base_feature_names : List[str]
        The 34 canonical feature names.
    replace_nominal_eps : bool
        If True, replaces 'Reported_EPS' with 'Earnings_Yield' and
        'EPS_Estimate' with 'Forward_Earnings_Yield', preserving exactly 34 columns.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        The transformed feature DataFrame and the list of feature column names.
    """
    data = compute_scale_invariant_earnings_features(df)

    if replace_nominal_eps:
        # 1-to-1 replacement to preserve exact 34-feature contract cardinality
        new_feature_names = []
        for feat in base_feature_names:
            if feat == "Reported_EPS":
                new_feature_names.append("Earnings_Yield")
            elif feat == "EPS_Estimate":
                new_feature_names.append("Forward_Earnings_Yield")
            else:
                new_feature_names.append(feat)
    else:
        # Augment with additional features
        new_feature_names = list(base_feature_names) + ["Earnings_Yield", "Forward_Earnings_Yield"]

    X = data[new_feature_names].copy()
    return X, new_feature_names

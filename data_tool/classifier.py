from typing import Any, Dict, List, Optional, Tuple, Union
import re
import pandas as pd
import numpy as np

from data_tool.schemas import (
    DataCategory,
    DataType,
    MarketRegime,
    EarningsSurpriseCategory,
    SentimentCategory,
)


# ============================================================
# 1. FACTOR TAXONOMY (INTERNAL vs EXTERNAL)
# ============================================================

INTERNAL_FEATURE_PATTERNS = [
    r"revenue",
    r"income",
    r"eps",
    r"earnings",
    r"debt",
    r"equity",
    r"asset",
    r"cash",
    r"capex",
    r"margin",
    r"dividend",
    r"bookvalue",
    r"company",
    r"pe_ratio",
    r"trailing",
    r"forward_pe",
    r"days_since_earnings",
    r"surprise",
]

EXTERNAL_FEATURE_PATTERNS = [
    r"nifty",
    r"vix",
    r"market",
    r"sma",
    r"ema",
    r"rsi",
    r"macd",
    r"bollinger",
    r"bb_",
    r"atr",
    r"volatility",
    r"momentum",
    r"roc",
    r"volume",
    r"spread",
    r"return",
    r"macro",
    r"rate",
    r"index",
]


def classify_factor_category(feature_name: str) -> DataCategory:
    """
    Classifies a feature name into INTERNAL (company-specific financial/earnings)
    or EXTERNAL (market, macro, technical, index, broad regime).
    """
    clean_name = feature_name.lower().strip()

    # Check internal keywords first
    for pat in INTERNAL_FEATURE_PATTERNS:
        if re.search(pat, clean_name):
            return DataCategory.INTERNAL

    # Check external keywords
    for pat in EXTERNAL_FEATURE_PATTERNS:
        if re.search(pat, clean_name):
            return DataCategory.EXTERNAL

    # Default conservative classification
    return DataCategory.EXTERNAL


def categorize_features(features: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Separates a dictionary of features into 'internal' and 'external' factor subsets.
    Crucial for the Explainable AI (XAI) deliverable.
    """
    internal = {}
    external = {}

    for k, v in features.items():
        cat = classify_factor_category(k)
        if cat == DataCategory.INTERNAL:
            internal[k] = v
        else:
            external[k] = v

    return {
        "internal": internal,
        "external": external,
    }


# ============================================================
# 2. AUTOMATED DATA TYPE INFERENCE
# ============================================================

def infer_data_type(data: Union[pd.DataFrame, Dict[str, Any]]) -> DataType:
    """
    Automatically identifies the data modality based on present columns / keys.
    """
    if isinstance(data, pd.DataFrame):
        cols = {str(c).lower().strip() for c in data.columns}
    elif isinstance(data, dict):
        cols = {str(c).lower().strip() for c in data.keys()}
    else:
        return DataType.GENERIC

    # Check for earnings events
    if any(k in cols for k in ["reported eps", "reported_eps", "eps_surprise", "eps estimate"]):
        return DataType.EARNINGS_EVENT

    # Check for news text
    if any(k in cols for k in ["headline", "title", "clean_text", "article"]):
        return DataType.NEWS_SENTIMENT

    # Check for fundamental financial statements
    if any(k in cols for k in ["total assets", "totalassets", "net income", "netincome", "operatingmargin"]):
        return DataType.FUNDAMENTAL_STATEMENT

    # Check for market index
    if any(k in cols for k in ["nifty_close", "vix_close", "indiavix", "^nsei"]):
        return DataType.MARKET_INDEX

    # Check for stock price OHLCV
    if all(k in cols for k in ["open", "high", "low", "close"]):
        return DataType.MARKET_PRICE

    # Check for company reference
    if any(k in cols for k in ["sector", "industry", "marketcap", "companyname"]):
        return DataType.COMPANY_METADATA

    return DataType.GENERIC


# ============================================================
# 3. EARNINGS SURPRISE CLASSIFICATION
# ============================================================

def classify_earnings_surprise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies quarterly earnings surprise into BEAT, MISS, or INLINE.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    # Determine surprise source column
    surprise_col = None
    for candidate in ["EPS_Surprise_Pct", "EPS_Surprise", "Surprise(%)", "Surprise"]:
        if candidate in data.columns:
            surprise_col = candidate
            break

    if surprise_col and surprise_col in data.columns:
        conditions = [
            data[surprise_col] > 1.0,   # > +1% surprise
            data[surprise_col] < -1.0,  # < -1% surprise
        ]
        choices = [
            EarningsSurpriseCategory.BEAT.value,
            EarningsSurpriseCategory.MISS.value,
        ]
        data["Earnings_Surprise_Category"] = np.select(
            conditions,
            choices,
            default=EarningsSurpriseCategory.INLINE.value
        )
    elif "Reported_EPS" in data.columns and "EPS_Estimate" in data.columns:
        diff = data["Reported_EPS"] - data["EPS_Estimate"]
        conditions = [
            diff > 0.05,
            diff < -0.05,
        ]
        choices = [
            EarningsSurpriseCategory.BEAT.value,
            EarningsSurpriseCategory.MISS.value,
        ]
        data["Earnings_Surprise_Category"] = np.select(
            conditions,
            choices,
            default=EarningsSurpriseCategory.INLINE.value
        )
    else:
        data["Earnings_Surprise_Category"] = EarningsSurpriseCategory.UNKNOWN.value

    return data


# ============================================================
# 4. MARKET REGIME CLASSIFICATION
# ============================================================

def classify_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies price time series into Trend and Volatility Regimes.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return df

    data = df.copy()
    close = data["Close"]

    # Trend regime via moving averages if available, else short rolling return
    if "SMA50" in data.columns:
        sma = data["SMA50"]
    else:
        sma = close.rolling(min(50, max(5, len(close)))).mean()

    diff_ratio = (close - sma) / (sma + 1e-8)

    conditions = [
        diff_ratio >= 0.02,
        diff_ratio <= -0.02,
    ]
    choices = [
        MarketRegime.BULLISH.value,
        MarketRegime.BEARISH.value,
    ]
    data["Trend_Regime"] = np.select(
        conditions,
        choices,
        default=MarketRegime.NEUTRAL.value
    )

    # Volatility regime
    returns = close.pct_change()
    vol_20 = returns.rolling(min(20, max(5, len(returns)))).std()
    vol_median = vol_20.median() if not vol_20.dropna().empty else 0.015

    data["Volatility_Regime"] = np.where(
        vol_20 > (vol_median * 1.5),
        MarketRegime.HIGH_VOLATILITY.value,
        MarketRegime.NORMAL_VOLATILITY.value
    )

    return data


# ============================================================
# 5. FINANCIAL NLP SENTIMENT CLASSIFICATION
# ============================================================

FINANCIAL_POSITIVE_TERMS = {
    "beat", "beats", "surged", "surge", "record profit", "growth",
    "exceeded", "higher revenue", "strong demand", "expansion",
    "deal", "order win", "dividend", "upgraded", "bullish"
}

FINANCIAL_NEGATIVE_TERMS = {
    "missed", "miss", "fall", "slump", "loss", "decline",
    "downgraded", "probe", "fine", "cut", "warning", "layoffs",
    "weak outlook", "pressure", "drop", "bearish"
}


def classify_text_sentiment(text: str) -> Tuple[SentimentCategory, float]:
    """
    Fast, deterministic financial domain sentiment classifier.
    Produces a SentimentCategory and score in [-1.0, 1.0].
    Can be complemented by FinBERT when heavy inference is active.
    """
    if not text or not isinstance(text, str):
        return SentimentCategory.NEUTRAL, 0.0

    lower_text = text.lower()
    pos_matches = sum(1 for term in FINANCIAL_POSITIVE_TERMS if re.search(r"\b" + re.escape(term) + r"\b", lower_text))
    neg_matches = sum(1 for term in FINANCIAL_NEGATIVE_TERMS if re.search(r"\b" + re.escape(term) + r"\b", lower_text))

    total = pos_matches + neg_matches
    if total == 0:
        return SentimentCategory.NEUTRAL, 0.0

    score = (pos_matches - neg_matches) / total

    if score > 0.1:
        return SentimentCategory.POSITIVE, round(score, 3)
    elif score < -0.1:
        return SentimentCategory.NEGATIVE, round(score, 3)
    else:
        return SentimentCategory.NEUTRAL, 0.0
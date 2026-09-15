"""
Authoritative Feature Registry for StockDNA-AI.

Every production feature must be registered here with:
- group: "INTERNAL" (company-specific micro factors) or "EXTERNAL" (macro/market/technical factors)
- source: data origin
- dtype: expected numeric data type
- definition: mathematical or business definition
- point_in_time_rule: guarantees against lookahead bias
- imputation_rule: strategy for missing values (fitted on training data only)
"""

from typing import Dict, Any, List

FEATURE_REGISTRY: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # EXTERNAL FEATURES: Technical Momentum, Trend, Volatility, Macro Benchmarks
    # =========================================================================
    "Daily_Return": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "1-day percentage change in Close price: Close[t] / Close[t-1] - 1",
        "point_in_time_rule": "Uses only Close[t] and Close[t-1] as of market close",
        "imputation_rule": "training_median",
    },
    "Return_5d": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "5-day rolling return: Close[t] / Close[t-5] - 1",
        "point_in_time_rule": "Backward-looking 5 bars only",
        "imputation_rule": "training_median",
    },
    "Return_10d": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "10-day rolling return: Close[t] / Close[t-10] - 1",
        "point_in_time_rule": "Backward-looking 10 bars only",
        "imputation_rule": "training_median",
    },
    "Return_21d": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "21-day rolling return (~1 month): Close[t] / Close[t-21] - 1",
        "point_in_time_rule": "Backward-looking 21 bars only",
        "imputation_rule": "training_median",
    },
    "ROC_10": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "10-day Rate of Change percentage: (Close[t] - Close[t-10]) / Close[t-10] * 100",
        "point_in_time_rule": "Backward-looking 10 bars only",
        "imputation_rule": "training_median",
    },
    "Close_to_SMA20": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Normalized distance from 20-day SMA: Close[t] / SMA20[t] - 1",
        "point_in_time_rule": "Uses Close[t] and rolling 20-day mean of Close",
        "imputation_rule": "training_median",
    },
    "Close_to_SMA50": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Normalized distance from 50-day SMA: Close[t] / SMA50[t] - 1",
        "point_in_time_rule": "Uses Close[t] and rolling 50-day mean of Close",
        "imputation_rule": "training_median",
    },
    "Close_to_SMA200": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Normalized distance from 200-day SMA: Close[t] / SMA200[t] - 1",
        "point_in_time_rule": "Uses Close[t] and rolling 200-day mean of Close",
        "imputation_rule": "training_median",
    },
    "SMA20_to_SMA50": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Moving average cross ratio: SMA20[t] / SMA50[t] - 1",
        "point_in_time_rule": "Uses backward-looking SMA20 and SMA50",
        "imputation_rule": "training_median",
    },
    "Normalized_ATR": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "14-day Average True Range normalized by Close: ATR14[t] / Close[t]",
        "point_in_time_rule": "Scale-invariant volatility metric using historical OHLC bars",
        "imputation_rule": "training_median",
    },
    "High_Low_Spread_Pct": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Intraday bar range percentage: (High[t] - Low[t]) / Close[t]",
        "point_in_time_rule": "Uses only bar t OHLC data",
        "imputation_rule": "training_median",
    },
    "Open_Close_Spread_Pct": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Intraday candle body percentage: (Close[t] - Open[t]) / Open[t]",
        "point_in_time_rule": "Uses only bar t Open and Close",
        "imputation_rule": "training_median",
    },
    "Rolling_Volatility_20d": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "20-day rolling standard deviation of Daily_Return",
        "point_in_time_rule": "Uses backward-looking 20 daily returns",
        "imputation_rule": "training_median",
    },
    "Bollinger_PctB": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Bollinger %B: (Close - LowerBand) / (UpperBand - LowerBand)",
        "point_in_time_rule": "Bounded 20-day 2-sigma channel position",
        "imputation_rule": "training_median",
    },
    "RSI_14": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "14-day Relative Strength Index (0 to 100)",
        "point_in_time_rule": "Bounded momentum oscillator on past 14 bars",
        "imputation_rule": "training_median",
    },
    "Normalized_MACD": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "MACD line normalized by Close: (EMA12 - EMA26) / Close[t]",
        "point_in_time_rule": "Scale-invariant MACD line",
        "imputation_rule": "training_median",
    },
    "Normalized_MACD_Hist": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "MACD histogram normalized by Close: (MACD - Signal) / Close[t]",
        "point_in_time_rule": "Scale-invariant MACD momentum indicator",
        "imputation_rule": "training_median",
    },
    "Volume_Change": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "1-day percentage change in Volume: Volume[t] / Volume[t-1] - 1",
        "point_in_time_rule": "Uses Volume[t] and Volume[t-1]",
        "imputation_rule": "training_median",
    },
    "Volume_to_MA20": {
        "group": "EXTERNAL",
        "source": "market_price",
        "dtype": "float64",
        "definition": "Volume relative to 20-day rolling Volume MA: Volume[t] / Volume_MA20[t]",
        "point_in_time_rule": "Uses backward-looking 20 volume bars",
        "imputation_rule": "training_median",
    },
    "NIFTY_Return_1d": {
        "group": "EXTERNAL",
        "source": "macro_market",
        "dtype": "float64",
        "definition": "1-day return of NIFTY 50 index: NIFTY_Close[t] / NIFTY_Close[t-1] - 1",
        "point_in_time_rule": "Uses NIFTY index close as of date t",
        "imputation_rule": "training_median",
    },
    "NIFTY_Return_5d": {
        "group": "EXTERNAL",
        "source": "macro_market",
        "dtype": "float64",
        "definition": "5-day rolling return of NIFTY 50 index",
        "point_in_time_rule": "Uses NIFTY index close past 5 days",
        "imputation_rule": "training_median",
    },
    "Relative_Strength_5d": {
        "group": "EXTERNAL",
        "source": "cross_market",
        "dtype": "float64",
        "definition": "Stock 5-day return minus NIFTY 5-day return (alpha momentum)",
        "point_in_time_rule": "Both stock and benchmark returns strictly backward-looking",
        "imputation_rule": "training_median",
    },
    "VIX_Level": {
        "group": "EXTERNAL",
        "source": "macro_market",
        "dtype": "float64",
        "definition": "India VIX index level on date t",
        "point_in_time_rule": "Market volatility index close as of date t",
        "imputation_rule": "training_median",
    },
    "VIX_Change_5d": {
        "group": "EXTERNAL",
        "source": "macro_market",
        "dtype": "float64",
        "definition": "5-day percentage change in India VIX: VIX[t] / VIX[t-5] - 1",
        "point_in_time_rule": "Uses VIX close past 5 days",
        "imputation_rule": "training_median",
    },

    # =========================================================================
    # INTERNAL FEATURES: Company-Specific Micro Fundamentals & Earnings Events
    # =========================================================================
    "Reported_EPS": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "float64",
        "definition": "Reported quarterly EPS from the most recent prior earnings announcement",
        "point_in_time_rule": "Backward-merged on AnnouncementDate <= date t (merge_asof backward)",
        "imputation_rule": "training_median",
    },
    "EPS_Estimate": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "float64",
        "definition": "Analyst consensus EPS estimate for the most recent prior earnings announcement",
        "point_in_time_rule": "Backward-merged on AnnouncementDate <= date t",
        "imputation_rule": "training_median",
    },
    "EPS_Surprise": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "float64",
        "definition": "Reported_EPS - EPS_Estimate",
        "point_in_time_rule": "Backward-merged on AnnouncementDate <= date t",
        "imputation_rule": "training_median",
    },
    "EPS_Surprise_Pct": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "float64",
        "definition": "Percentage surprise: EPS_Surprise / (|EPS_Estimate| + 1e-5)",
        "point_in_time_rule": "Scale-invariant surprise metric; backward-merged",
        "imputation_rule": "training_median",
    },
    "Earnings_Event": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "int64",
        "definition": "Binary flag: 1 if date t is the exact day of earnings announcement, 0 otherwise",
        "point_in_time_rule": "Exact date match with verified announcement calendar",
        "imputation_rule": "constant_zero",
    },
    "Positive_Earnings_Surprise": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "int64",
        "definition": "Binary flag: 1 if most recent earnings surprise > 0, 0 otherwise",
        "point_in_time_rule": "Backward-merged on AnnouncementDate <= date t",
        "imputation_rule": "constant_zero",
    },
    "Negative_Earnings_Surprise": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "int64",
        "definition": "Binary flag: 1 if most recent earnings surprise < 0, 0 otherwise",
        "point_in_time_rule": "Backward-merged on AnnouncementDate <= date t",
        "imputation_rule": "constant_zero",
    },
    "Days_Since_Earnings": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "float64",
        "definition": "Calendar days elapsed since the most recent earnings announcement",
        "point_in_time_rule": "(date[t] - Last_Earnings_Date).dt.days; non-negative",
        "imputation_rule": "training_median",
    },
    "Post_Earnings_1_Week": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "int64",
        "definition": "Binary flag: 1 if Days_Since_Earnings between 0 and 5, 0 otherwise",
        "point_in_time_rule": "Point-in-time window evaluation",
        "imputation_rule": "constant_zero",
    },
    "Post_Earnings_1_Month": {
        "group": "INTERNAL",
        "source": "historical_earnings",
        "dtype": "int64",
        "definition": "Binary flag: 1 if Days_Since_Earnings between 0 and 21, 0 otherwise",
        "point_in_time_rule": "Point-in-time window evaluation",
        "imputation_rule": "constant_zero",
    },
}

# Ordered canonical list of feature names
FEATURE_NAMES: List[str] = list(FEATURE_REGISTRY.keys())

# Subsets by group
INTERNAL_FEATURES: List[str] = [
    f for f, meta in FEATURE_REGISTRY.items() if meta["group"] == "INTERNAL"
]

EXTERNAL_FEATURES: List[str] = [
    f for f, meta in FEATURE_REGISTRY.items() if meta["group"] == "EXTERNAL"
]


def get_feature_registry() -> Dict[str, Dict[str, Any]]:
    """Returns the full feature registry."""
    return FEATURE_REGISTRY


def get_feature_names() -> List[str]:
    """Returns the ordered list of production feature names."""
    return FEATURE_NAMES.copy()


def get_internal_features() -> List[str]:
    """Returns the list of INTERNAL features."""
    return INTERNAL_FEATURES.copy()


def get_external_features() -> List[str]:
    """Returns the list of EXTERNAL features."""
    return EXTERNAL_FEATURES.copy()


def get_feature_group(feature_name: str) -> str:
    """Returns 'INTERNAL' or 'EXTERNAL' for a given feature."""
    if feature_name in FEATURE_REGISTRY:
        return FEATURE_REGISTRY[feature_name]["group"]
    return "EXTERNAL"

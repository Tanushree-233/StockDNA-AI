import numpy as np
import pandas as pd


# Label encoding:
# 0 = Sell
# 1 = Hold
# 2 = Buy

LABEL_MAP = {
    0: "Sell",
    1: "Hold",
    2: "Buy"
}


def generate_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Automatically generates Buy / Hold / Sell labels
    using future stock returns.

    horizon:
        Number of trading days into the future.

    buy_threshold:
        Minimum future return required for Buy.

    sell_threshold:
        Maximum future return allowed for Sell.
    """

    df = df.copy()

    # Future closing price
    df["future_price"] = (
        df[price_col].shift(-horizon)
    )

    # Future percentage return
    df["future_return"] = (
        df["future_price"] / df[price_col]
    ) - 1

    # Create classification labels
    conditions = [
        df["future_return"] >= buy_threshold,
        df["future_return"] <= sell_threshold
    ]

    choices = [
        2,  # Buy
        0   # Sell
    ]

    # Anything between the thresholds = Hold
    df["target"] = np.select(
        conditions,
        choices,
        default=1
    )

    # Last few rows cannot be labelled because
    # their future price does not exist.
    df = (
        df
        .dropna(subset=["future_return"])
        .reset_index(drop=True)
    )

    return df


def label_distribution(
    df: pd.DataFrame,
    target_col: str = "target"
) -> pd.Series:
    """
    Shows the percentage distribution of
    Sell / Hold / Buy labels.
    """

    counts = (
        df[target_col]
        .value_counts(normalize=True)
        .sort_index()
    )

    counts.index = [
        LABEL_MAP[i]
        for i in counts.index
    ]

    return counts


def dynamic_threshold_labels(
    df: pd.DataFrame,
    horizon: int = 5,
    volatility_window: int = 20,
    multiplier: float = 1.0,
    price_col: str = "Close",
) -> pd.DataFrame:
    """
    Optional alternative to fixed Buy/Sell thresholds.

    Thresholds adapt according to recent volatility.
    """

    df = df.copy()

    returns = df[price_col].pct_change()

    rolling_volatility = (
        returns
        .rolling(volatility_window)
        .std()
    )

    df["future_price"] = (
        df[price_col]
        .shift(-horizon)
    )

    df["future_return"] = (
        df["future_price"] / df[price_col]
    ) - 1

    dynamic_buy = (
        multiplier * rolling_volatility
    )

    dynamic_sell = (
        -multiplier * rolling_volatility
    )

    conditions = [
        df["future_return"] >= dynamic_buy,
        df["future_return"] <= dynamic_sell
    ]

    choices = [
        2,  # Buy
        0   # Sell
    ]

    df["target"] = np.select(
        conditions,
        choices,
        default=1
    )

    df = (
        df
        .dropna(subset=["future_return"])
        .reset_index(drop=True)
    )

    return df
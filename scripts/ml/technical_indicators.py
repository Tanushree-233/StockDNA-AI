import numpy as np
import pandas as pd


def add_moving_averages(
    df: pd.DataFrame,
    windows=(5, 10, 20, 50)
) -> pd.DataFrame:

    df = df.copy()

    for w in windows:

        df[f"sma_{w}"] = (
            df["Close"]
            .rolling(w)
            .mean()
        )

        df[f"ema_{w}"] = (
            df["Close"]
            .ewm(
                span=w,
                adjust=False
            )
            .mean()
        )

        df[f"close_to_sma_{w}"] = (
            df["Close"] /
            df[f"sma_{w}"]
        ) - 1

    return df


def add_rsi(
    df: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:

    df = df.copy()

    delta = df["Close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = (
        gain
        .rolling(window)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(window)
        .mean()
    )

    rs = avg_gain / (
        avg_loss + 1e-9
    )

    df[f"rsi_{window}"] = (
        100 -
        (
            100 / (1 + rs)
        )
    )

    return df


def add_macd(
    df: pd.DataFrame,
    fast=12,
    slow=26,
    signal=9
) -> pd.DataFrame:

    df = df.copy()

    ema_fast = (
        df["Close"]
        .ewm(
            span=fast,
            adjust=False
        )
        .mean()
    )

    ema_slow = (
        df["Close"]
        .ewm(
            span=slow,
            adjust=False
        )
        .mean()
    )

    df["macd"] = (
        ema_fast - ema_slow
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(
            span=signal,
            adjust=False
        )
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    window=20,
    num_std=2
) -> pd.DataFrame:

    df = df.copy()

    sma = (
        df["Close"]
        .rolling(window)
        .mean()
    )

    std = (
        df["Close"]
        .rolling(window)
        .std()
    )

    df["bb_upper"] = (
        sma + num_std * std
    )

    df["bb_lower"] = (
        sma - num_std * std
    )

    df["bb_width"] = (
        (
            df["bb_upper"] -
            df["bb_lower"]
        )
        / (sma + 1e-9)
    )

    df["bb_pct_b"] = (
        (
            df["Close"] -
            df["bb_lower"]
        )
        /
        (
            df["bb_upper"] -
            df["bb_lower"] +
            1e-9
        )
    )

    return df


def add_atr(
    df: pd.DataFrame,
    window=14
) -> pd.DataFrame:

    df = df.copy()

    high_low = (
        df["High"] -
        df["Low"]
    )

    high_close = (
        df["High"] -
        df["Close"].shift()
    ).abs()

    low_close = (
        df["Low"] -
        df["Close"].shift()
    ).abs()

    true_range = pd.concat(
        [
            high_low,
            high_close,
            low_close
        ],
        axis=1
    ).max(axis=1)

    df[f"atr_{window}"] = (
        true_range
        .rolling(window)
        .mean()
    )

    return df


def add_volatility(
    df: pd.DataFrame,
    windows=(5, 10, 20)
) -> pd.DataFrame:

    df = df.copy()

    returns = (
        df["Close"]
        .pct_change()
    )

    for w in windows:

        df[f"volatility_{w}"] = (
            returns
            .rolling(w)
            .std()
        )

    return df


def add_volume_features(
    df: pd.DataFrame,
    windows=(5, 10, 20)
) -> pd.DataFrame:

    df = df.copy()

    for w in windows:

        volume_ma = (
            df["Volume"]
            .rolling(w)
            .mean()
        )

        df[f"volume_ma_{w}"] = volume_ma

        df[f"volume_ratio_{w}"] = (
            df["Volume"] /
            (volume_ma + 1e-9)
        )

    df["volume_change"] = (
        df["Volume"].pct_change()
    )

    return df


def add_lag_features(
    df: pd.DataFrame,
    columns=("Close", "Volume"),
    lags=(1, 2, 3, 5)
) -> pd.DataFrame:

    df = df.copy()

    for column in columns:

        for lag in lags:

            df[
                f"{column.lower()}_lag_{lag}"
            ] = (
                df[column]
                .shift(lag)
            )

    return df


def add_return_features(
    df: pd.DataFrame,
    periods=(1, 3, 5, 10)
) -> pd.DataFrame:

    df = df.copy()

    for period in periods:

        df[
            f"return_{period}d"
        ] = (
            df["Close"]
            .pct_change(period)
        )

    return df


def build_technical_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Complete technical feature engineering pipeline.
    """

    df = df.copy()

    df = add_moving_averages(df)

    df = add_rsi(df)

    df = add_macd(df)

    df = add_bollinger_bands(df)

    df = add_atr(df)

    df = add_volatility(df)

    df = add_volume_features(df)

    df = add_lag_features(df)

    df = add_return_features(df)

    return df
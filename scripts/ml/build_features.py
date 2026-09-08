import pandas as pd

from technical_indicators import build_technical_features
from target_design import generate_labels


NON_FEATURE_COLS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Adj Close",
    "future_price",
    "future_return",
    "target",
    "Ticker",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw stock data and sort it chronologically.
    """

    df = pd.read_csv(
        path,
        parse_dates=["Date"]
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df


def build_dataset(
    raw_df: pd.DataFrame,
    horizon: int = 5,
    buy_threshold: float = 0.02,
    sell_threshold: float = -0.02,
):
    """
    Converts raw stock data into an engineered,
    labelled dataset.

    Feature selection is deliberately NOT performed here.

    This prevents information from the eventual test
    period from influencing feature selection.
    """

    # --------------------------------------------------
    # 1. Technical feature engineering
    # --------------------------------------------------

    df = build_technical_features(raw_df)

    # --------------------------------------------------
    # 2. Automatic Buy / Hold / Sell target
    # --------------------------------------------------

    df = generate_labels(
        df,
        horizon=horizon,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )

    # --------------------------------------------------
    # 3. Remove rows containing NaN values
    # --------------------------------------------------

    df = (
        df
        .dropna()
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # 4. Identify feature columns
    # --------------------------------------------------

    feature_cols = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLS
    ]

    X = df[feature_cols]

    y = df["target"]

    dates = df["Date"]

    return X, y, dates, feature_cols


if __name__ == "__main__":

    import sys

    # Allow a CSV path to be supplied from the terminal
    raw_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/processed/prices/TCS.csv"
    )

    print("\n========================================")
    print("STOCKDNA FEATURE BUILDING")
    print("========================================")

    print(f"\nInput:")
    print(raw_path)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    raw_df = load_raw_data(raw_path)

    print(
        f"Loaded {len(raw_df)} rows."
    )

    # --------------------------------------------------
    # Build dataset
    # --------------------------------------------------

    X, y, dates, features = build_dataset(
        raw_df
    )

    print(
        f"\nEngineered dataset:"
        f" {X.shape[0]} rows,"
        f" {X.shape[1]} features"
    )

    # --------------------------------------------------
    # Display target distribution
    # --------------------------------------------------

    print("\nTarget distribution:")

    print(
        y.value_counts(
            normalize=True
        )
        .sort_index()
    )

    # --------------------------------------------------
    # Display features
    # --------------------------------------------------

    print("\nEngineered features:")

    for feature in features:
        print(f"  - {feature}")

    # --------------------------------------------------
    # Save model-ready dataset
    # --------------------------------------------------

    output = X.copy()

    output["target"] = y.values

    output["Date"] = dates.values

    output_path = (
        "data/processed/features_labeled.csv"
    )

    output.to_csv(
        output_path,
        index=False
    )

    print(
        f"\nSaved dataset to:"
        f"\n{output_path}"
    )

    print("\n========================================")
    print("FEATURE BUILD COMPLETE")
    print("========================================")
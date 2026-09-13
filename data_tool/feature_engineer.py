import pandas as pd

from scripts.feature_engineering.calculate_indicators import calculate_indicators


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(
            "Cannot generate features from an empty dataset."
        )

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    data = calculate_indicators(data)

    return data
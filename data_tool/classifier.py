import pandas as pd

from scripts.ml.predict import predict


def classify_data(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        raise ValueError(
            "Cannot classify an empty dataset."
        )

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = df.copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    predictions = predict(data)

    return predictions
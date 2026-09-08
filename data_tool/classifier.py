import sys
import os
import pandas as pd

ML_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "scripts",
        "ml"
    )
)

if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)


from predict_pipeline import predict

def classify_data(
    df: pd.DataFrame,
    model_name: str = "xgboost",
    version: int = None
) -> pd.DataFrame:
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

    # Make a copy so the original data is not modified
    data = df.copy()

    # Make sure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"])

    # Make sure data is chronologically ordered
    data = (
        data
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Run existing ML prediction pipeline
    predictions = predict(
        data,
        model_name=model_name,
        version=version
    )

    return predictions
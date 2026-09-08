import pandas as pd

from aggregator import aggregate_data
from cleaner import clean_data
from deduplicator import remove_duplicates
from classifier import classify_data


def run_data_tool(
    *datasets: pd.DataFrame,
    model_name: str = "xgboost",
    model_version: int = None
) -> pd.DataFrame:
    data = aggregate_data(*datasets)

    print(
        f"[Data Tool] Aggregated {len(data)} rows."
    )


    data = clean_data(data)

    print(
        f"[Data Tool] After cleaning: {len(data)} rows."
    )

    duplicate_keys = [
        "Ticker",
        "Date"
    ]

    data = remove_duplicates(
        data,
        subset=duplicate_keys
    )

    print(
        f"[Data Tool] After deduplication: "
        f"{len(data)} rows."
    )

    predictions = classify_data(
        data,
        model_name=model_name,
        version=model_version
    )

    print(
        "[Data Tool] Automatic classification complete."
    )

    return predictions
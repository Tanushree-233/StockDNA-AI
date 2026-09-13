import pandas as pd

from data_tool.aggregator import aggregate_data
from data_tool.cleaner import clean_data
from data_tool.deduplicator import remove_duplicates
from data_tool.feature_engineer import generate_features
from data_tool.classifier import classify_data


def run_data_tool(
    *datasets: pd.DataFrame
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

    data = generate_features(data)

    print(
        f"[Data Tool] Features generated: "
        f"{len(data)} rows."
    )

    predictions = classify_data(data)

    print(
        "[Data Tool] Automatic classification complete."
    )

    return predictions
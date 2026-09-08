import pandas as pd


def aggregate_data(*datasets: pd.DataFrame) -> pd.DataFrame:
    valid_datasets = [
        df for df in datasets
        if df is not None and not df.empty
    ]

    if not valid_datasets:
        raise ValueError(
            "No valid datasets were provided for aggregation."
        )

    combined = pd.concat(
        valid_datasets,
        ignore_index=True,
        sort=False
    )

    return combined.reset_index(drop=True)
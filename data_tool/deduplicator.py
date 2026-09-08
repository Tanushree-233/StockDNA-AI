import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
    subset=None
) -> pd.DataFrame:

    df = df.copy()

    if subset is None:
        return (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

    available_columns = [
        column
        for column in subset
        if column in df.columns
    ]

    if not available_columns:
        return (
            df
            .drop_duplicates()
            .reset_index(drop=True)
        )

    return (
        df
        .drop_duplicates(
            subset=available_columns
        )
        .reset_index(drop=True)
    )
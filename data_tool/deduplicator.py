from typing import List, Optional, Tuple
import pandas as pd


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "last"
) -> pd.DataFrame:
    """
    Deduplicates DataFrame rows by subset columns, defaulting to keeping the latest record.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if subset is None:
        return data.drop_duplicates(keep=keep).reset_index(drop=True)

    available_columns = [col for col in subset if col in data.columns]
    if not available_columns:
        return data.drop_duplicates(keep=keep).reset_index(drop=True)

    return data.drop_duplicates(subset=available_columns, keep=keep).reset_index(drop=True)


def deduplicate_with_stats(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = "last"
) -> Tuple[pd.DataFrame, int]:
    """
    Removes duplicate records and returns the deduplicated DataFrame along
    with the count of removed duplicate rows.
    """
    if df is None or df.empty:
        return pd.DataFrame(), 0

    initial_len = len(df)
    cleaned_df = remove_duplicates(df, subset=subset, keep=keep)
    duplicates_removed = initial_len - len(cleaned_df)
    return cleaned_df, duplicates_removed
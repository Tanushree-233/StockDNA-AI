import pandas as pd
from typing import Optional, Union

from data_tool.service import DataToolService
from data_tool.schemas import ProcessedDataBundle, DataType


def run_data_tool(
    *datasets: pd.DataFrame,
    ticker: Optional[str] = None,
    return_bundle: bool = False
) -> Union[pd.DataFrame, ProcessedDataBundle]:
    """
    Standard data processing pipeline.
    Ingests raw datasets, cleans, normalizes, deduplicates,
    generates scale-invariant indicators, and classifies market regimes/factors.

    Returns:
        pd.DataFrame (by default for backward compatibility)
        ProcessedDataBundle (if return_bundle=True)
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided to run_data_tool.")

    valid = [d for d in datasets if d is not None and not d.empty]
    if not valid:
        raise ValueError("No non-empty datasets provided.")

    service = DataToolService(default_source="data_tool_pipeline")

    if len(valid) == 1:
        combined = valid[0]
    else:
        combined = pd.concat(valid, ignore_index=True, sort=False)

    bundle = service.process(
        data=combined,
        ticker=ticker,
        generate_technical_features=True
    )

    print(f"[Data Tool] Ingested & processed {bundle.records_count} structured records.")
    print(f"[Data Tool] Categories: {bundle.get_provenance_summary()['categories']}")

    if return_bundle:
        return bundle
    return bundle.to_dataframe()
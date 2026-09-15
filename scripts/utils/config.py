from config.settings import (
    PROJECT_ROOT,
    load_raw_config,
    get_project_root,
    get_company_universe,
    get_tickers,
    get_ticker_mapping,
    get_data_path,
    get_model_path,
)


def load_config():
    """
    Backward-compatible loader that returns raw config dict
    with paths resolved against the authoritative project root.
    """
    raw = load_raw_config()
    return raw
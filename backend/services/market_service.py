import yfinance as yf


def get_market_features():
    """
    Fetch NIFTY and India VIX features.
    Compatible with both old and new versions of yfinance.
    """

    # Download NIFTY
    nifty = yf.download(
        "^NSEI",
        period="5d",
        progress=False,
        auto_adjust=False
    )

    # Download India VIX
    vix = yf.download(
        "^INDIAVIX",
        period="5d",
        progress=False,
        auto_adjust=False
    )

    if nifty.empty:
        raise ValueError("Unable to fetch NIFTY data.")

    if vix.empty:
        raise ValueError("Unable to fetch VIX data.")

    # Handle MultiIndex columns (newer yfinance versions)
    if hasattr(nifty.columns, "nlevels") and nifty.columns.nlevels > 1:
        nifty.columns = nifty.columns.get_level_values(0)

    if hasattr(vix.columns, "nlevels") and vix.columns.nlevels > 1:
        vix.columns = vix.columns.get_level_values(0)

    # Convert values to floats
    nifty_close = float(nifty["Close"].iloc[-1])

    nifty_volume = float(nifty["Volume"].iloc[-1])

    nifty_return = float(
        nifty["Close"].pct_change().iloc[-1]
    )

    vix_close = float(
        vix["Close"].iloc[-1]
    )

    vix_change = float(
        vix["Close"].pct_change().iloc[-1]
    )

    return {
        "NIFTY_Close": nifty_close,
        "NIFTY_Volume": nifty_volume,
        "VIX_Close": vix_close,
        "NIFTY_Return": nifty_return,
        "VIX_Change": vix_change
    }
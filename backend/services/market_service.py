import yfinance as yf


def get_market_features():
    """
    Fetch NIFTY and India VIX features.
    """

    nifty = yf.download(
        "^NSEI",
        period="5d",
        progress=False,
        auto_adjust=False
    )

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
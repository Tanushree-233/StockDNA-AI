import yfinance as yf


def get_fundamentals(ticker: str):
    """
    Fetch fundamental features for a stock.
    """

    if not ticker.endswith(".NS"):
        ticker += ".NS"

    stock = yf.Ticker(ticker)

    info = stock.info

    return {
        "MarketCap": info.get("marketCap", 0),
        "BookValue": info.get("bookValue", 0),
        "DividendYield": info.get("dividendYield", 0) or 0,
        "Beta": info.get("beta", 0),
        "CurrentPrice": info.get("currentPrice", 0),
        "TrailingEPS": info.get("trailingEps", 0),
        "ForwardEPS": info.get("forwardEps", 0),
        "ForwardPE": info.get("forwardPE", 0),
        "EarningsQuarterlyGrowth": info.get(
            "earningsQuarterlyGrowth",
            0
        ),
        "PE_Ratio": info.get("trailingPE", 0)
    }
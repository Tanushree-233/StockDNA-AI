import pandas as pd

STOCK_DATA = "data/final/master_dataset.csv"
EVENT_DATA = "data/processed/internal/earnings_events.csv"
OUTPUT = "data/processed/internal/earnings_event_stock_dataset.csv"

# --------------------------------------------------
# LOAD
# --------------------------------------------------

stock = pd.read_csv(STOCK_DATA)
events = pd.read_csv(EVENT_DATA)

stock["Date"] = pd.to_datetime(stock["Date"])
events["Date"] = pd.to_datetime(events["Date"])

stock["Ticker"] = stock["Ticker"].str.replace(".NS", "", regex=False)
events["Ticker"] = events["Ticker"].str.replace(".NS", "", regex=False)

# --------------------------------------------------
# ALIGN EACH STOCK WITH MOST RECENT EARNINGS EVENT
# --------------------------------------------------

result = []

for ticker in stock["Ticker"].unique():

    print(f"Aligning {ticker}...")

    stock_ticker = (
        stock[stock["Ticker"] == ticker]
        .copy()
        .sort_values("Date")
    )

    events_ticker = (
        events[events["Ticker"] == ticker]
        .copy()
        .sort_values("Date")
    )

    # Preserve the actual earnings date
    events_ticker["Last_Earnings_Date"] = events_ticker["Date"]

    merged = pd.merge_asof(
        stock_ticker,
        events_ticker,
        on="Date",
        direction="backward",
        suffixes=("", "_event")
    )

    result.append(merged)


final = pd.concat(result, ignore_index=True)

# --------------------------------------------------
# DATE CLEANUP
# --------------------------------------------------

final["Last_Earnings_Date"] = pd.to_datetime(
    final["Last_Earnings_Date"],
    errors="coerce"
)

# --------------------------------------------------
# IMPORTANT:
# EARNINGS_EVENT MUST ONLY BE 1 ON THE ACTUAL
# EARNINGS ANNOUNCEMENT DATE
# --------------------------------------------------

final["Earnings_Event"] = (
    final["Date"] == final["Last_Earnings_Date"]
).astype(int)

# --------------------------------------------------
# EPS DATA
# --------------------------------------------------

final["EPS_Estimate"] = final["EPS_Estimate"].fillna(0)

final["Reported_EPS"] = final["Reported_EPS"].fillna(0)

final["EPS_Surprise"] = final["EPS_Surprise"].fillna(0)

# --------------------------------------------------
# SURPRISE FLAGS
# --------------------------------------------------

final["Positive_Earnings_Surprise"] = (
    final["EPS_Surprise"] > 0
).astype(int)

final["Negative_Earnings_Surprise"] = (
    final["EPS_Surprise"] < 0
).astype(int)

# --------------------------------------------------
# DAYS SINCE MOST RECENT EARNINGS
# --------------------------------------------------

final["Days_Since_Earnings"] = (
    final["Date"] - final["Last_Earnings_Date"]
).dt.days

final["Days_Since_Earnings"] = (
    final["Days_Since_Earnings"]
    .fillna(9999)
)

# --------------------------------------------------
# POST-EARNINGS WINDOWS
# --------------------------------------------------

final["Post_Earnings_1_Week"] = (
    final["Days_Since_Earnings"].between(0, 5)
).astype(int)

final["Post_Earnings_1_Month"] = (
    final["Days_Since_Earnings"].between(0, 20)
).astype(int)

# --------------------------------------------------
# SORT
# --------------------------------------------------

final = final.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# --------------------------------------------------
# SAVE
# --------------------------------------------------

final.to_csv(
    OUTPUT,
    index=False
)

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("\n" + "=" * 60)
print("EARNINGS EVENT STOCK DATASET")
print("=" * 60)

print("\nShape:")
print(final.shape)

print("\nRows by company:")
print(final["Ticker"].value_counts())

print("\nActual earnings events:")
print(
    final["Earnings_Event"].value_counts()
)

print("\nEvents by ticker:")
print(
    final[final["Earnings_Event"] == 1]
    .groupby("Ticker")
    .size()
)

print("\nActual event rows:")
print(
    final[
        final["Earnings_Event"] == 1
    ][
        [
            "Ticker",
            "Date",
            "Last_Earnings_Date",
            "EPS_Estimate",
            "Reported_EPS",
            "EPS_Surprise",
            "Days_Since_Earnings"
        ]
    ].to_string(index=False)
)

print("\nSaved:")
print(OUTPUT)
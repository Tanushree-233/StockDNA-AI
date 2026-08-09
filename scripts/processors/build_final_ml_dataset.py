import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

INTERNAL_PATH = (
    "data/processed/internal/internal_stock_dataset.csv"
)

EARNINGS_PATH = (
    "data/processed/internal/earnings_event_stock_dataset.csv"
)

OUTPUT_PATH = (
    "data/processed/internal/final_ml_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING FINAL ML DATASET")
print("=" * 70)

internal = pd.read_csv(INTERNAL_PATH)
earnings = pd.read_csv(EARNINGS_PATH)


# ============================================================
# CONVERT DATES
# ============================================================

internal["Date"] = pd.to_datetime(
    internal["Date"]
)

earnings["Date"] = pd.to_datetime(
    earnings["Date"]
)

if "AnnouncementDate" in internal.columns:
    internal["AnnouncementDate"] = pd.to_datetime(
        internal["AnnouncementDate"],
        errors="coerce"
    )

if "Last_Earnings_Date" in earnings.columns:
    earnings["Last_Earnings_Date"] = pd.to_datetime(
        earnings["Last_Earnings_Date"],
        errors="coerce"
    )


# ============================================================
# NORMALIZE TICKERS
# ============================================================

internal["Ticker"] = (
    internal["Ticker"]
    .astype(str)
    .str.replace(".NS", "", regex=False)
)

earnings["Ticker"] = (
    earnings["Ticker"]
    .astype(str)
    .str.replace(".NS", "", regex=False)
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nInternal dataset:")
print("Shape:", internal.shape)

print("\nEarnings dataset:")
print("Shape:", earnings.shape)

print("\nInternal companies:")
print(internal["Ticker"].value_counts())

print("\nEarnings companies:")
print(earnings["Ticker"].value_counts())


# ============================================================
# FINANCIAL FEATURES
# ============================================================

financial_columns = [
    "FinancialDate",
    "AnnouncementDate",

    "Revenue",
    "OperatingIncome",
    "NetIncome",
    "DilutedEPS",

    "TotalAssets",
    "TotalDebt",
    "StockholdersEquity",
    "Cash",

    "OperatingCashFlow",
    "FreeCashFlow",
    "CapitalExpenditure",

    "OperatingMargin",
    "NetMargin",
    "DebtToAssets",
    "DebtToEquity",

    "RevenueGrowth",
    "NetIncomeGrowth",
    "EPSGrowth",
    "FCFGrowth"
]


financial_columns = [
    col
    for col in financial_columns
    if col in internal.columns
]


# ============================================================
# EARNINGS FEATURES
# ============================================================

earnings_columns = [
    "EPS_Estimate",
    "Reported_EPS",
    "EPS_Surprise",
    "Earnings_Event",
    "Positive_Earnings_Surprise",
    "Negative_Earnings_Surprise",

    "Last_Earnings_Date",
    "Days_Since_Earnings",
    "Post_Earnings_1_Week",
    "Post_Earnings_1_Month"
]


earnings_columns = [
    col
    for col in earnings_columns
    if col in earnings.columns
]


# ============================================================
# CREATE CLEAN EARNINGS EVENT TABLE
# ============================================================

# We only need one copy of each event.
#
# IMPORTANT:
# We do NOT directly use the current day's
# Reported_EPS / EPS_Surprise.
#
# The current day's earnings announcement may not
# have been known when the trading-day prediction
# was made.
#
# Therefore, the event information will be shifted
# to the next trading day for conservative leakage
# prevention.

event_columns = [
    "Ticker",
    "Date"
] + earnings_columns


events = earnings[event_columns].copy()


# Remove duplicate event rows.

events = events.drop_duplicates(
    subset=["Ticker", "Date"],
    keep="last"
)


events = events.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ============================================================
# SHIFT EARNINGS INFORMATION TO NEXT TRADING DAY
# ============================================================

shift_columns = [
    col
    for col in earnings_columns
    if col not in [
        "Last_Earnings_Date",
        "Days_Since_Earnings",
        "Post_Earnings_1_Week",
        "Post_Earnings_1_Month"
    ]
]


# Rename original event date.

events["EarningsAnnouncementDate"] = events["Date"]


# The event becomes usable from the NEXT stock trading row.

shifted_events = []

for ticker in events["Ticker"].unique():

    ticker_events = events[
        events["Ticker"] == ticker
    ].copy()

    ticker_events = ticker_events.sort_values(
        "Date"
    )

    # Event features themselves are associated with
    # the next trading day through merge_asof below.

    shifted_events.append(ticker_events)


events = pd.concat(
    shifted_events,
    ignore_index=True
)


# ============================================================
# MERGE EARNINGS INFORMATION
# ============================================================

# Remove columns that are calculated relative to the
# original event date. We will calculate them again
# after the merge.

events_for_merge = events.drop(
    columns=[
        "Days_Since_Earnings",
        "Post_Earnings_1_Week",
        "Post_Earnings_1_Month"
    ],
    errors="ignore"
)


events_for_merge = events_for_merge.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# Create stock dataframe.

base = internal.copy()

base = base.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# ------------------------------------------------------------
# IMPORTANT:
# We want the PREVIOUS earnings announcement, not the
# current day's announcement.
#
# To enforce this, use allow_exact_matches=False.
# ------------------------------------------------------------

final = pd.merge_asof(
    base,
    events_for_merge,
    left_on="Date",
    right_on="Date",
    by="Ticker",
    direction="backward",
    allow_exact_matches=False,
    suffixes=("", "_earnings")
)


# ============================================================
# RESTORE / CLEAN EARNINGS ANNOUNCEMENT DATE
# ============================================================

if "EarningsAnnouncementDate" in final.columns:

    final["EarningsAnnouncementDate"] = pd.to_datetime(
        final["EarningsAnnouncementDate"],
        errors="coerce"
    )


# ============================================================
# CALCULATE DAYS SINCE LAST KNOWN EARNINGS
# ============================================================

if "EarningsAnnouncementDate" in final.columns:

    final["Days_Since_Earnings"] = (
        final["Date"]
        - final["EarningsAnnouncementDate"]
    ).dt.days

else:

    final["Days_Since_Earnings"] = 9999


final["Days_Since_Earnings"] = (
    final["Days_Since_Earnings"]
    .fillna(9999)
)


# ============================================================
# POST-EARNINGS WINDOWS
# ============================================================

final["Post_Earnings_1_Week"] = (
    final["Days_Since_Earnings"]
    .between(0, 5)
).astype(int)


final["Post_Earnings_1_Month"] = (
    final["Days_Since_Earnings"]
    .between(0, 20)
).astype(int)


# ============================================================
# REMOVE DUPLICATE / UNNECESSARY COLUMNS
# ============================================================

drop_columns = [
    "Ticker_earnings",
    "Date_earnings",
    "Ticker_event"
]

final.drop(
    columns=drop_columns,
    errors="ignore",
    inplace=True
)


# ============================================================
# ENSURE ONE TARGET
# ============================================================

if "Target" not in final.columns:

    raise ValueError(
        "Target column is missing."
    )


if "Target_Return" not in final.columns:

    raise ValueError(
        "Target_Return column is missing."
    )


# ============================================================
# REMOVE EXACT DUPLICATE COLUMNS
# ============================================================

final = final.loc[
    :,
    ~final.columns.duplicated()
]


# ============================================================
# SORT
# ============================================================

final = final.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ============================================================
# CHECK FOR DUPLICATE STOCK-DATE ROWS
# ============================================================

duplicates = final.duplicated(
    subset=["Ticker", "Date"]
).sum()

print("\nDuplicate Ticker-Date rows:", duplicates)

if duplicates > 0:

    raise ValueError(
        "Duplicate Ticker-Date rows detected."
    )


# ============================================================
# CHECK TARGET DISTRIBUTION
# ============================================================

print("\nTarget distribution:")

print(
    final["Target"].value_counts()
)

print("\nTarget percentages:")

print(
    (
        final["Target"]
        .value_counts(normalize=True)
        * 100
    ).round(2)
)


# ============================================================
# CHECK EARNINGS FEATURE AVAILABILITY
# ============================================================

earnings_feature_columns = [
    "EPS_Estimate",
    "Reported_EPS",
    "EPS_Surprise",
    "Earnings_Event",
    "Positive_Earnings_Surprise",
    "Negative_Earnings_Surprise",
    "EarningsAnnouncementDate",
    "Days_Since_Earnings",
    "Post_Earnings_1_Week",
    "Post_Earnings_1_Month"
]


print("\nEarnings feature availability:")

for col in earnings_feature_columns:

    if col in final.columns:

        available = final[col].notna().sum()

        print(
            f"{col}: "
            f"{available} / {len(final)}"
        )


# ============================================================
# CHECK FINANCIAL FEATURE AVAILABILITY
# ============================================================

print("\nFinancial feature availability:")

for col in financial_columns:

    if col in final.columns:

        available = final[col].notna().sum()

        print(
            f"{col}: "
            f"{available} / {len(final)}"
        )


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL ML DATASET")
print("=" * 70)

print("\nShape:")
print(final.shape)

print("\nColumns:")
print(final.columns.tolist())

print("\nDate range:")
print(
    final["Date"].min(),
    "→",
    final["Date"].max()
)

print("\nCompanies:")
print(
    final["Ticker"].value_counts()
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

final.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\nSaved:")
print(OUTPUT_PATH)

print("\nFINAL ML DATASET BUILD COMPLETE.")
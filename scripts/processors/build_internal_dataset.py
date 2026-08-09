import pandas as pd


STOCK_DATA = "data/final/master_dataset.csv"
INTERNAL_DATA = "data/processed/internal/internal_quarterly.csv"
EARNINGS_DATA = "data/processed/internal/earnings_events.csv"
OUTPUT = "data/processed/internal/internal_stock_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

stock = pd.read_csv(STOCK_DATA)
internal = pd.read_csv(INTERNAL_DATA)
earnings = pd.read_csv(EARNINGS_DATA)


# ============================================================
# CONVERT DATES
# ============================================================

stock["Date"] = pd.to_datetime(stock["Date"])
internal["FinancialDate"] = pd.to_datetime(
    internal["FinancialDate"]
)
earnings["Date"] = pd.to_datetime(
    earnings["Date"]
)


# ============================================================
# NORMALIZE TICKER NAMES
# ============================================================

stock["Ticker"] = (
    stock["Ticker"]
    .str.replace(".NS", "", regex=False)
)

internal["Ticker"] = (
    internal["Ticker"]
    .str.replace(".NS", "", regex=False)
)

earnings["Ticker"] = (
    earnings["Ticker"]
    .str.replace(".NS", "", regex=False)
)


# ============================================================
# FIND ACTUAL ANNOUNCEMENT DATE
# ============================================================

print("\n" + "=" * 60)
print("ALIGNING QUARTERLY FINANCIAL DATA")
print("=" * 60)

internal = internal.sort_values(
    ["FinancialDate", "Ticker"]
).reset_index(drop=True)

earnings = earnings.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# Keep only the information required to determine
# when the quarterly financial information became public.

announcement_data = earnings[
    ["Ticker", "Date"]
].copy()

announcement_data.rename(
    columns={
        "Date": "AnnouncementDate"
    },
    inplace=True
)

# Match every financial quarter with the NEXT
# earnings announcement for the same company.

internal = pd.merge_asof(
    internal,
    announcement_data,
    left_on="FinancialDate",
    right_on="AnnouncementDate",
    by="Ticker",
    direction="forward"
)


print("\nQuarterly financial data with announcement dates:")

print(
    internal[
        [
            "Ticker",
            "FinancialDate",
            "AnnouncementDate"
        ]
    ].to_string(index=False)
)


# ============================================================
# VALIDATE ANNOUNCEMENT DATES
# ============================================================

invalid_dates = internal[
    internal["AnnouncementDate"]
    < internal["FinancialDate"]
]

if not invalid_dates.empty:

    print("\nWARNING:")
    print(
        "Some announcement dates occur before "
        "their financial dates."
    )

    print(
        invalid_dates[
            [
                "Ticker",
                "FinancialDate",
                "AnnouncementDate"
            ]
        ].to_string(index=False)
    )

else:

    print(
        "\nAnnouncement-date validation passed."
    )


# ============================================================
# MERGE FINANCIAL DATA INTO DAILY STOCK DATA
# ============================================================

result = []


for ticker in stock["Ticker"].unique():

    print(f"\nAligning {ticker}...")

    stock_ticker = stock[
        stock["Ticker"] == ticker
    ].copy()

    internal_ticker = internal[
        internal["Ticker"] == ticker
    ].copy()

    if internal_ticker.empty:

        print(
            f"No internal data available for {ticker}"
        )

        continue

    # The stock dataframe already contains the
    # authoritative Ticker column.

    internal_ticker = internal_ticker.drop(
        columns=["Ticker"]
    )

    # IMPORTANT:
    # Stock dates are sorted by Date.
    # Financial information is sorted by
    # AnnouncementDate because that is when
    # the information became available.

    stock_ticker = stock_ticker.sort_values(
        "Date"
    ).reset_index(drop=True)

    internal_ticker = internal_ticker.sort_values(
        "AnnouncementDate"
    ).reset_index(drop=True)

    merged = pd.merge_asof(
        stock_ticker,
        internal_ticker,
        left_on="Date",
        right_on="AnnouncementDate",
        direction="backward"
    )

    result.append(merged)


# ============================================================
# COMBINE ALL COMPANIES
# ============================================================

if not result:

    raise ValueError(
        "No companies could be merged."
    )


final = pd.concat(
    result,
    ignore_index=True
)


# ============================================================
# SORT FINAL DATASET
# ============================================================

final = final.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ============================================================
# SAVE DATASET
# ============================================================

final.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# VALIDATION / REPORT
# ============================================================

print("\n" + "=" * 60)
print("INTERNAL STOCK DATASET")
print("=" * 60)

print("\nShape:")
print(final.shape)


print("\nCompanies:")
print(
    final["Ticker"].value_counts()
)


print("\nFinancial dates used:")

print(
    final[
        [
            "Ticker",
            "Date",
            "FinancialDate",
            "AnnouncementDate"
        ]
    ].head(30).to_string(index=False)
)


# ============================================================
# CHECK FOR LOOK-AHEAD BIAS
# ============================================================

print("\n" + "=" * 60)
print("LOOK-AHEAD BIAS CHECK")
print("=" * 60)

available_before_announcement = final[
    (
        final["Date"]
        < final["AnnouncementDate"]
    )
    & final["Revenue"].notna()
]

print(
    "Rows containing financial data before announcement:",
    len(available_before_announcement)
)

if len(available_before_announcement) == 0:

    print(
        "PASS: No financial information is available "
        "before its announcement date."
    )

else:

    print(
        "WARNING: Potential look-ahead bias detected."
    )

    print(
        available_before_announcement[
            [
                "Ticker",
                "Date",
                "FinancialDate",
                "AnnouncementDate",
                "Revenue"
            ]
        ].head(20).to_string(index=False)
    )


# ============================================================
# MISSING INTERNAL DATA
# ============================================================

print("\n" + "=" * 60)
print("MISSING INTERNAL DATA")
print("=" * 60)

internal_columns = [
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
    "CapitalExpenditure"
]

print(
    final[internal_columns].isna().sum()
)


# ============================================================
# DATA AVAILABILITY BY COMPANY
# ============================================================

print("\n" + "=" * 60)
print("FINANCIAL DATA AVAILABILITY")
print("=" * 60)

for ticker in final["Ticker"].unique():

    ticker_data = final[
        final["Ticker"] == ticker
    ]

    available_rows = ticker_data[
        ticker_data["Revenue"].notna()
    ]

    print(
        f"{ticker}: "
        f"{len(available_rows)} / "
        f"{len(ticker_data)} rows with financial data"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("INTERNAL DATASET BUILD COMPLETE")
print("=" * 60)

print("\nSaved:")
print(OUTPUT)
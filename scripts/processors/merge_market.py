import pandas as pd


MASTER_PATH = "data/final/master_dataset.csv"
NIFTY_PATH = "data/processed/market/nifty50.csv"
VIX_PATH = "data/processed/market/indiavix.csv"


# ==================================================
# LOAD MASTER DATASET
# ==================================================

master = pd.read_csv(MASTER_PATH)

master["Date"] = pd.to_datetime(master["Date"])


# ==================================================
# REMOVE OLD MARKET COLUMNS
# ==================================================

market_columns = [
    "NIFTY_Close",
    "NIFTY_Volume",
    "NIFTY_Return",
    "VIX_Close",
    "VIX_Change",
    "NIFTY_Close_x",
    "NIFTY_Close_y",
    "NIFTY_Volume_x",
    "NIFTY_Volume_y",
    "VIX_Close_x",
    "VIX_Close_y"
]

master.drop(
    columns=market_columns,
    errors="ignore",
    inplace=True
)


# ==================================================
# LOAD NIFTY
# ==================================================

# The CSV contains 3 header rows:
#
# Price,...
# Ticker,...
# Date,...
#
# Actual data starts from row 4.
#
# Therefore skip rows 1, 2, and 3.

nifty = pd.read_csv(
    NIFTY_PATH,
    skiprows=[1, 2, 3]
)

# The first column is the actual date column.
nifty.rename(
    columns={
        nifty.columns[0]: "Date"
    },
    inplace=True
)

nifty["Date"] = pd.to_datetime(
    nifty["Date"],
    errors="coerce"
)

nifty["NIFTY_Close"] = pd.to_numeric(
    nifty["Close"],
    errors="coerce"
)

nifty["NIFTY_Volume"] = pd.to_numeric(
    nifty["Volume"],
    errors="coerce"
)

nifty = nifty[
    [
        "Date",
        "NIFTY_Close",
        "NIFTY_Volume"
    ]
].copy()


# ==================================================
# CLEAN NIFTY
# ==================================================

nifty.dropna(
    subset=[
        "Date",
        "NIFTY_Close",
        "NIFTY_Volume"
    ],
    inplace=True
)

nifty = nifty.sort_values(
    "Date"
).reset_index(drop=True)


# ==================================================
# CALCULATE NIFTY RETURN
# BEFORE MERGING WITH STOCK DATA
# ==================================================

nifty["NIFTY_Return"] = (
    nifty["NIFTY_Close"]
    .pct_change(fill_method=None)
)


# ==================================================
# LOAD INDIA VIX
# ==================================================

vix = pd.read_csv(
    VIX_PATH,
    skiprows=[1, 2, 3]
)

vix.rename(
    columns={
        vix.columns[0]: "Date"
    },
    inplace=True
)

vix["Date"] = pd.to_datetime(
    vix["Date"],
    errors="coerce"
)

vix["VIX_Close"] = pd.to_numeric(
    vix["Close"],
    errors="coerce"
)

vix = vix[
    [
        "Date",
        "VIX_Close"
    ]
].copy()


# ==================================================
# CLEAN VIX
# ==================================================

vix.dropna(
    subset=[
        "Date",
        "VIX_Close"
    ],
    inplace=True
)

vix = vix.sort_values(
    "Date"
).reset_index(drop=True)


# ==================================================
# CALCULATE VIX CHANGE
# BEFORE MERGING WITH STOCK DATA
# ==================================================

vix["VIX_Change"] = (
    vix["VIX_Close"]
    .pct_change(fill_method=None)
)


# ==================================================
# CHECK FOR DUPLICATE MARKET DATES
# ==================================================

if nifty["Date"].duplicated().any():
    raise ValueError(
        "Duplicate dates found in NIFTY data."
    )

if vix["Date"].duplicated().any():
    raise ValueError(
        "Duplicate dates found in VIX data."
    )


# ==================================================
# MERGE NIFTY INTO MASTER
# ==================================================

master = master.merge(
    nifty[
        [
            "Date",
            "NIFTY_Close",
            "NIFTY_Volume",
            "NIFTY_Return"
        ]
    ],
    on="Date",
    how="left",
    validate="many_to_one"
)


# ==================================================
# MERGE VIX INTO MASTER
# ==================================================

master = master.merge(
    vix[
        [
            "Date",
            "VIX_Close",
            "VIX_Change"
        ]
    ],
    on="Date",
    how="left",
    validate="many_to_one"
)


# ==================================================
# REMOVE ROWS WITHOUT MARKET DATA
# ==================================================

master.dropna(
    subset=[
        "NIFTY_Close",
        "VIX_Close"
    ],
    inplace=True
)


# ==================================================
# SORT MASTER DATASET
# ==================================================

master["Date"] = pd.to_datetime(
    master["Date"]
)

master = master.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# ==================================================
# SAVE
# ==================================================

master.to_csv(
    MASTER_PATH,
    index=False
)


# ==================================================
# VALIDATION
# ==================================================

print("=" * 60)
print("MARKET DATA MERGED SUCCESSFULLY")
print("=" * 60)

print("\nRows:", len(master))

print("\nMarket columns:")
print(
    [
        "NIFTY_Close",
        "NIFTY_Volume",
        "NIFTY_Return",
        "VIX_Close",
        "VIX_Change"
    ]
)

print("\nMissing market values:")

print(
    master[
        [
            "NIFTY_Close",
            "NIFTY_Volume",
            "NIFTY_Return",
            "VIX_Close",
            "VIX_Change"
        ]
    ].isnull().sum()
)

print("\nDate range:")
print(
    master["Date"].min(),
    "→",
    master["Date"].max()
)

print("\nStock distribution:")
print(
    master["Ticker"].value_counts()
)

print("\nNIFTY return uniqueness per date:")

print(
    master.groupby("Date")["NIFTY_Return"]
    .nunique()
    .max()
)

print("\nVIX change uniqueness per date:")

print(
    master.groupby("Date")["VIX_Change"]
    .nunique()
    .max()
)

print("\nMarket data merge complete!")
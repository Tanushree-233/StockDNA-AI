import pandas as pd

MASTER_PATH = "data/final/master_dataset.csv"
NIFTY_PATH = "data/processed/market/nifty50.csv"
VIX_PATH = "data/processed/market/indiavix.csv"

master = pd.read_csv(MASTER_PATH)
market_columns = [
    "NIFTY_Close",
    "NIFTY_Volume",
    "VIX_Close",
    "NIFTY_Return",
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

nifty = pd.read_csv(NIFTY_PATH, skiprows=[1])
vix = pd.read_csv(VIX_PATH, skiprows=[1])

# Rename first column to Date
nifty.rename(columns={"Price": "Date"}, inplace=True)
vix.rename(columns={"Price": "Date"}, inplace=True)

for col in ["Close", "High", "Low", "Open", "Volume"]:
    nifty[col] = pd.to_numeric(nifty[col], errors="coerce")
    vix[col] = pd.to_numeric(vix[col], errors="coerce")

nifty = nifty.rename(columns={
    "Close": "NIFTY_Close",
    "Volume": "NIFTY_Volume"
})

vix = vix.rename(columns={
    "Close": "VIX_Close"
})

nifty = nifty[
    [
        "Date",
        "NIFTY_Close",
        "NIFTY_Volume"
    ]
]

vix = vix[
    [
        "Date",
        "VIX_Close"
    ]
]

master = master.merge(
    nifty,
    on="Date",
    how="left"
)

master = master.merge(
    vix,
    on="Date",
    how="left"
)

print("\nMaster Columns:")
print(master.columns.tolist())

print("\nNifty Columns:")
print(nifty.columns.tolist())

print("\nVIX Columns:")
print(vix.columns.tolist())

master["NIFTY_Return"] = master["NIFTY_Close"].pct_change(fill_method=None)

master["VIX_Change"] = master["VIX_Close"].pct_change(fill_method=None)

master.dropna(
    subset=["NIFTY_Close", "VIX_Close"],
    inplace=True
)

master.reset_index(drop=True, inplace=True)

master.to_csv(
    MASTER_PATH,
    index=False
)

print("Market data merged successfully!")
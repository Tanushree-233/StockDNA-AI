import pandas as pd

MASTER_PATH = "data/final/master_dataset.csv"
FUNDAMENTALS_PATH = "data/processed/fundamentals/fundamentals.csv"

master = pd.read_csv(MASTER_PATH)
fundamentals = pd.read_csv(FUNDAMENTALS_PATH)

fundamentals["Ticker"] = (
    fundamentals["Ticker"]
    .str.replace(".NS", "", regex=False)
)

merged = master.merge(
    fundamentals,
    on="Ticker",
    how="left"
)

print(merged.head())

print("\nRows:", len(merged))

print("\nColumns:", len(merged.columns))

print(
    merged[
        [
            "MarketCap",
            "PE_Ratio",
            "BookValue"
        ]
    ].isnull().sum()
)

merged.to_csv(
    "data/final/master_dataset.csv",
    index=False
)

print("Master dataset updated.")
import pandas as pd

MASTER_PATH = "data/final/master_dataset.csv"
EARNINGS_PATH = "data/processed/earnings/earnings.csv"

master = pd.read_csv(MASTER_PATH)
earnings = pd.read_csv(EARNINGS_PATH)

earnings["Ticker"] = (
    earnings["Ticker"]
    .str.replace(".NS", "", regex=False)
)

merged = master.merge(
    earnings,
    on="Ticker",
    how="left"
)

print("\nMerged Columns:\n")
print(merged.columns.tolist())

print("=" * 50)
print("Merged Dataset")
print("=" * 50)

print("Rows:", len(merged))
print("Columns:", len(merged.columns))

print()

print(
    merged[
        [
            "TrailingEPS",
            "ForwardEPS",
            "ForwardPE",
            "EarningsQuarterlyGrowth"
        ]
    ].isnull().sum()
)

merged["Company"] = merged["Company_x"]

merged.drop(
    columns=["Company_x", "Company_y"],
    inplace=True
)

merged["PE_Ratio"] = merged["PE_Ratio_x"]

merged.drop(
    columns=["PE_Ratio_x", "PE_Ratio_y"],
    inplace=True
)

merged.to_csv(
    MASTER_PATH,
    index=False
)

print("Master dataset updated with earnings.")
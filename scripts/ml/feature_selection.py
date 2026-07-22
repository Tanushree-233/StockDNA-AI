import pandas as pd

DATASET = "data/final/master_dataset.csv"

df = pd.read_csv(DATASET)

print("=" * 50)
print("Original Shape")
print("=" * 50)
print(df.shape)

# Target column
TARGET = "Target"

# Columns not useful for training
DROP_COLUMNS = [
    "Date",
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "Close",
    "Target_Return"
]

X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
y = df[TARGET]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

print("\nRemaining Features:\n")

for col in X.columns:
    print(col)
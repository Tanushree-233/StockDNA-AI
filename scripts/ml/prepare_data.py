import os
import joblib
import pandas as pd


# -----------------------
# Configuration
# -----------------------

DATASET = "data/final/master_dataset.csv"

DROP_COLUMNS = [
    "Date",
    "Ticker",
    "Company",
    "Sector",
    "Industry",
    "Close",
    "Target_Return",
]

TARGET = "Target"


# -----------------------
# Load Dataset
# -----------------------

df = pd.read_csv(DATASET)

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Sort by ticker and date
df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


# -----------------------
# Convert Target Labels
# -----------------------

label_map = {
    0: 0,
    1: 1,
    2: 2,
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2,
}

df[TARGET] = df[TARGET].map(label_map)


# -----------------------
# Time-Based Train/Test Split
# -----------------------

train_parts = []
test_parts = []

for ticker, stock_df in df.groupby("Ticker", sort=False):

    stock_df = stock_df.sort_values("Date").reset_index(drop=True)

    split_index = int(len(stock_df) * 0.8)

    train_stock = stock_df.iloc[:split_index].copy()
    test_stock = stock_df.iloc[split_index:].copy()

    train_parts.append(train_stock)
    test_parts.append(test_stock)


# Combine all stocks

train_df = pd.concat(
    train_parts,
    ignore_index=True
)

test_df = pd.concat(
    test_parts,
    ignore_index=True
)


# -----------------------
# Sort Combined Data
# -----------------------

train_df = train_df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

test_df = test_df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# -----------------------
# Create X and y
# -----------------------

X_train = train_df.drop(
    columns=[TARGET] + DROP_COLUMNS,
    errors="ignore"
)

y_train = train_df[TARGET]

X_test = test_df.drop(
    columns=[TARGET] + DROP_COLUMNS,
    errors="ignore"
)

y_test = test_df[TARGET]


# -----------------------
# Print Dataset Information
# -----------------------

print("=" * 50)
print("TRAINING SET")
print("=" * 50)

print("Shape:", X_train.shape)

print("\nClass Distribution:")

print(
    y_train.value_counts(
        normalize=True
    ).sort_index() * 100
)


print("\n")

print("=" * 50)
print("TESTING SET")
print("=" * 50)

print("Shape:", X_test.shape)

print("\nClass Distribution:")

print(
    y_test.value_counts(
        normalize=True
    ).sort_index() * 100
)


# -----------------------
# Print Date Ranges
# -----------------------

print("\n")

print("=" * 50)
print("DATE RANGES")
print("=" * 50)

print(
    "Training:",
    train_df["Date"].min(),
    "→",
    train_df["Date"].max()
)

print(
    "Testing:",
    test_df["Date"].min(),
    "→",
    test_df["Date"].max()
)


print("\n")

print("=" * 50)
print("TEST STOCK DISTRIBUTION")
print("=" * 50)

print(
    test_df["Ticker"].value_counts()
)


# -----------------------
# Save Prepared Data
# -----------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    X_train,
    "models/X_train.pkl"
)

joblib.dump(
    X_test,
    "models/X_test.pkl"
)

joblib.dump(
    y_train,
    "models/y_train.pkl"
)

joblib.dump(
    y_test,
    "models/y_test.pkl"
)


print("\nPrepared datasets saved successfully!")
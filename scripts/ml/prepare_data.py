import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------
# Load Dataset
# -----------------------

DATASET = "data/final/master_dataset.csv"

df = pd.read_csv(DATASET)

# -----------------------
# Drop Unwanted Columns
# -----------------------

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

X = df.drop(columns=[TARGET] + DROP_COLUMNS, errors="ignore")
y = df[TARGET]

# -----------------------
# Convert Target Labels
# -----------------------

label_map = {
    0: 0,
    1: 1,
    2: 2,

    "SELL": 0,
    "HOLD": 1,
    "BUY": 2
}

y = y.map(label_map)

# -----------------------
# Time-based Train/Test Split
# -----------------------

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("=" * 50)
print("Training Set")
print("=" * 50)
print(X_train.shape)

print("\n")

print("=" * 50)
print("Testing Set")
print("=" * 50)
print(X_test.shape)

# -----------------------
# Save Processed Data
# -----------------------

os.makedirs("models", exist_ok=True)

joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nPrepared datasets saved successfully!")

print(df["Target"].value_counts(normalize=True) * 100)
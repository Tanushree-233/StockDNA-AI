import os
import joblib
import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

DATASET = "data/final/master_dataset.csv"
MODEL_FOLDER = "models"

TARGET = "Target"

DROP_COLUMNS = [
    "Date",
    "Ticker",
    "Close",
    "Target_Return",
    "Target"
]


# ==================================================
# LOAD DATASET
# ==================================================

df = pd.read_csv(DATASET)

df["Date"] = pd.to_datetime(df["Date"])

# Explicit chronological ordering
df = df.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)


# ==================================================
# FEATURES / TARGET
# ==================================================

X = df.drop(
    columns=DROP_COLUMNS,
    errors="ignore"
)

y = df[TARGET]


# ==================================================
# VALIDATE FEATURES
# ==================================================

print("=" * 60)
print("EXTERNAL DATASET")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nFeature shape:", X.shape)

print("\nNumber of features:", len(X.columns))

print("\nFeatures:")

for col in X.columns:
    print(col)


# ==================================================
# TARGET ENCODING
# ==================================================

label_map = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2
}

y = y.map(label_map)


if y.isnull().any():
    raise ValueError(
        "Target contains unknown or missing labels."
    )


# ==================================================
# TIME-BASED TRAIN / TEST SPLIT
# ==================================================

# IMPORTANT:
# Split by DATE, not by arbitrary row position.
#
# We use the first 80% of unique dates for training
# and the remaining 20% for testing.

unique_dates = sorted(
    df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.8
)

train_dates = unique_dates[:split_index]
test_dates = unique_dates[split_index:]


train_mask = df["Date"].isin(train_dates)
test_mask = df["Date"].isin(test_dates)


X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()


# ==================================================
# VALIDATION
# ==================================================

print("\n" + "=" * 60)
print("TRAINING SET")
print("=" * 60)

print("Shape:", X_train.shape)

print("\nClass Distribution:")

print(
    (
        y_train
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


print("\n" + "=" * 60)
print("TESTING SET")
print("=" * 60)

print("Shape:", X_test.shape)

print("\nClass Distribution:")

print(
    (
        y_test
        .value_counts(normalize=True)
        .sort_index()
        * 100
    ).round(2)
)


print("\n" + "=" * 60)
print("DATE RANGES")
print("=" * 60)

print(
    "Training:",
    df.loc[train_mask, "Date"].min(),
    "→",
    df.loc[train_mask, "Date"].max()
)

print(
    "Testing:",
    df.loc[test_mask, "Date"].min(),
    "→",
    df.loc[test_mask, "Date"].max()
)


# ==================================================
# STOCK DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("TEST STOCK DISTRIBUTION")
print("=" * 60)

print(
    df.loc[test_mask, "Ticker"].value_counts()
)


# ==================================================
# CHECK FOR MISSING VALUES
# ==================================================

if X_train.isnull().any().any():
    raise ValueError(
        "Missing values found in X_train."
    )

if X_test.isnull().any().any():
    raise ValueError(
        "Missing values found in X_test."
    )


# ==================================================
# CHECK TRAIN / TEST DATE OVERLAP
# ==================================================

overlap = set(train_dates).intersection(
    set(test_dates)
)

if overlap:
    raise ValueError(
        "Train/test date overlap detected."
    )


# ==================================================
# SAVE
# ==================================================

os.makedirs(
    MODEL_FOLDER,
    exist_ok=True
)


joblib.dump(
    X_train,
    "models/X_external_train.pkl"
)

joblib.dump(
    X_test,
    "models/X_external_test.pkl"
)

joblib.dump(
    y_train,
    "models/y_external_train.pkl"
)

joblib.dump(
    y_test,
    "models/y_external_test.pkl"
)


print(
    "\nExternal training/test datasets saved successfully!"
)
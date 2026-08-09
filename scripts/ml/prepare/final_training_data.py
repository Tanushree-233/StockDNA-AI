import os
import joblib
import pandas as pd
import numpy as np

INPUT = "data/processed/internal/final_ml_dataset.csv"
MODEL_DIR = "models"

TRAIN_END = "2024-07-26"
TEST_START = "2024-07-29"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("PREPARING FINAL TRAINING DATA")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"])
df["Ticker"] = df["Ticker"].astype(str)

print("\nOriginal shape:", df.shape)

# ---------------------------------------------------------
# 2. SORT CHRONOLOGICALLY
# ---------------------------------------------------------

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# ---------------------------------------------------------
# 3. REMOVE TARGET-RELATED / NON-FEATURE COLUMNS
# ---------------------------------------------------------

TARGET = "Target"

DROP_COLUMNS = [
    "Target",
    "Target_Return",
]

# These dates are metadata rather than numerical model features.
DATE_COLUMNS = [
    "Date",
    "FinancialDate",
    "AnnouncementDate",
    "Last_Earnings_Date",
    "EarningsAnnouncementDate",
]

# ---------------------------------------------------------
# 4. TARGET
# ---------------------------------------------------------

label_map = {
    "SELL": 0,
    "HOLD": 1,
    "BUY": 2,
}

df[TARGET] = df[TARGET].map(label_map)

if df[TARGET].isna().any():
    raise ValueError("Target contains unknown or missing labels.")

# ---------------------------------------------------------
# 5. FINANCIAL FEATURES
# ---------------------------------------------------------

financial_features = [
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
    "FCFGrowth",
]

financial_features = [
    c for c in financial_features
    if c in df.columns
]

# ---------------------------------------------------------
# 6. CREATE MISSINGNESS INDICATORS
# ---------------------------------------------------------

print("\nCreating financial missing indicators...")

for col in financial_features:
    df[f"{col}_Missing"] = df[col].isna().astype(int)

# ---------------------------------------------------------
# 7. RESPECT ANNOUNCEMENT DATE
# ---------------------------------------------------------
#
# Financial information may only be used from the
# announcement date onward.
#
# We therefore:
#   - keep financial values only when Date >= AnnouncementDate
#   - forward-fill only within ticker
#   - never back-fill
#

if "AnnouncementDate" in df.columns:

    df["AnnouncementDate"] = pd.to_datetime(
        df["AnnouncementDate"],
        errors="coerce"
    )

    for col in financial_features:

        # Values before announcement are unavailable.
        invalid = (
            df["AnnouncementDate"].notna()
            & (df["Date"] < df["AnnouncementDate"])
        )

        df.loc[invalid, col] = np.nan

        # Forward-fill only within the same company.
        df[col] = (
            df.groupby("Ticker")[col]
            .ffill()
        )

# ---------------------------------------------------------
# 8. PREPARE FEATURE COLUMNS
# ---------------------------------------------------------

DROP_COLUMNS = DROP_COLUMNS + DATE_COLUMNS

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
    and c != TARGET
    and c != "Ticker"
]

# Keep only numeric features.
feature_columns = [
    c for c in feature_columns
    if pd.api.types.is_numeric_dtype(df[c])
]

print("\nNumber of features:", len(feature_columns))

print("\nFeatures:")
for i, col in enumerate(feature_columns, 1):
    print(f"{i:02d}. {col}")

# ---------------------------------------------------------
# 9. FINAL NUMERIC CLEANUP
# ---------------------------------------------------------

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Replace infinite values.
X = X.replace([np.inf, -np.inf], np.nan)

# ---------------------------------------------------------
# 10. TIME-BASED SPLIT
# ---------------------------------------------------------

train_mask = df["Date"] <= pd.Timestamp(TRAIN_END)

test_mask = df["Date"] >= pd.Timestamp(TEST_START)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

# ---------------------------------------------------------
# 11. FILL REMAINING MISSING VALUES
# ---------------------------------------------------------
#
# At this point remaining NaNs mean that no historical
# financial observation was available for that company.
#
# We use training-set medians ONLY.
#

train_medians = X_train.median()

X_train = X_train.fillna(train_medians)
X_test = X_test.fillna(train_medians)

# Any completely empty feature gets 0 after the median step.
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# ---------------------------------------------------------
# 12. VALIDATION
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining:")
print("Shape:", X_train.shape)
print(
    "Date:",
    df.loc[train_mask, "Date"].min(),
    "→",
    df.loc[train_mask, "Date"].max()
)

print("\nTesting:")
print("Shape:", X_test.shape)
print(
    "Date:",
    df.loc[test_mask, "Date"].min(),
    "→",
    df.loc[test_mask, "Date"].max()
)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTraining target percentages:")
print(
    (y_train.value_counts(normalize=True).sort_index() * 100)
    .round(2)
)

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())

print("\nRemaining NaNs:")
print("X_train:", X_train.isna().sum().sum())
print("X_test :", X_test.isna().sum().sum())

print("\nRemaining infinite values:")
print(
    "X_train:",
    np.isinf(X_train.to_numpy()).sum()
)

print(
    "X_test :",
    np.isinf(X_test.to_numpy()).sum()
)

# ---------------------------------------------------------
# 13. SAVE
# ---------------------------------------------------------

joblib.dump(X_train, f"{MODEL_DIR}/X_train.pkl")
joblib.dump(X_test, f"{MODEL_DIR}/X_test.pkl")
joblib.dump(y_train, f"{MODEL_DIR}/y_train.pkl")
joblib.dump(y_test, f"{MODEL_DIR}/y_test.pkl")

joblib.dump(
    feature_columns,
    f"{MODEL_DIR}/feature_columns.pkl"
)

joblib.dump(
    train_medians,
    f"{MODEL_DIR}/training_medians.pkl"
)

# ---------------------------------------------------------
# 14. FINAL SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("TRAINING DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nSaved:")
print("models/X_train.pkl")
print("models/X_test.pkl")
print("models/y_train.pkl")
print("models/y_test.pkl")
print("models/feature_columns.pkl")
print("models/training_medians.pkl")

print("\nFinal feature count:", len(feature_columns))
print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))

print("\nREADY FOR MODEL TRAINING.")
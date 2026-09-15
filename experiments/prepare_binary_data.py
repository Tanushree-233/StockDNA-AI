import os
import joblib
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

INPUT = "data/processed/internal/final_ml_dataset.csv"
MODEL_DIR = "models"

TRAIN_END = "2024-07-26"
TEST_START = "2024-07-29"

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("PREPARING BINARY BUY / SELL DATA")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv(INPUT)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Ticker"] = df["Ticker"].astype(str)

print("\nOriginal shape:", df.shape)

# ============================================================
# 2. SORT
# ============================================================

df = (
    df.sort_values(["Ticker", "Date"])
      .reset_index(drop=True)
)

# ============================================================
# 3. REMOVE INVALID DATES
# ============================================================

df = df.dropna(subset=["Date"]).copy()

# ============================================================
# 4. TARGET CHECK
# ============================================================

print("\nOriginal target distribution:")
print(df["Target"].value_counts(dropna=False))

# Keep only BUY and SELL
df = df[
    df["Target"].isin(["BUY", "SELL"])
].copy()

print("\nAfter removing HOLD:")
print(df["Target"].value_counts())

# ============================================================
# 5. CONVERT TARGET
# ============================================================

label_map = {
    "SELL": 0,
    "BUY": 1
}

df["Target"] = df["Target"].map(label_map)

if df["Target"].isna().any():
    raise ValueError("Target contains invalid labels.")

# ============================================================
# 6. DATE COLUMNS
# ============================================================

DATE_COLUMNS = [
    "Date",
    "FinancialDate",
    "AnnouncementDate",
    "Last_Earnings_Date",
    "EarningsAnnouncementDate",
]

# Only use columns that actually exist
DATE_COLUMNS = [
    col for col in DATE_COLUMNS
    if col in df.columns
]

# ============================================================
# 7. FINANCIAL FEATURES
# ============================================================

FINANCIAL_FEATURES = [
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

FINANCIAL_FEATURES = [
    col
    for col in FINANCIAL_FEATURES
    if col in df.columns
]

print("\nFinancial features:")
for col in FINANCIAL_FEATURES:
    print("-", col)

# ============================================================
# 8. RESPECT ANNOUNCEMENT DATE
# ============================================================

if "AnnouncementDate" in df.columns:

    print("\nApplying announcement-date protection...")

    df["AnnouncementDate"] = pd.to_datetime(
        df["AnnouncementDate"],
        errors="coerce"
    )

    invalid_mask = (
        df["AnnouncementDate"].notna()
        & (df["Date"] < df["AnnouncementDate"])
    )

    print(
        "Rows before financial announcement:",
        invalid_mask.sum()
    )

    # IMPORTANT:
    # Do not allow any financial feature from a future
    # announcement to be visible.
    for col in FINANCIAL_FEATURES:
        df.loc[invalid_mask, col] = np.nan

    # Forward-fill only information that has already
    # become available.
    for col in FINANCIAL_FEATURES:
        df[col] = (
            df.groupby("Ticker")[col]
              .ffill()
        )

else:
    print(
        "\nWARNING: AnnouncementDate column not found."
    )

# ============================================================
# 9. CREATE FINANCIAL MISSING INDICATORS
# ============================================================

print("\nCreating financial availability indicators...")

MISSING_FEATURES = []

for col in FINANCIAL_FEATURES:

    missing_col = f"{col}_Missing"

    df[missing_col] = (
        df[col].isna().astype(int)
    )

    MISSING_FEATURES.append(missing_col)

# ============================================================
# 10. EXTERNAL FEATURES
# ============================================================

EXTERNAL_FEATURES = [
    "NIFTY_Close",
    "NIFTY_Volume",
    "NIFTY_Return",
    "VIX_Close",
    "VIX_Change",
]

EXTERNAL_FEATURES = [
    col
    for col in EXTERNAL_FEATURES
    if col in df.columns
]

print("\nExternal features:")

for col in EXTERNAL_FEATURES:
    print("-", col)

if len(EXTERNAL_FEATURES) == 0:
    raise ValueError(
        "No external features were found."
    )

# ============================================================
# 11. COLUMNS THAT MUST NEVER BE FEATURES
# ============================================================

DROP_COLUMNS = [
    "Target",
    "Target_Return",
    "Ticker",
]

DROP_COLUMNS.extend(DATE_COLUMNS)

# ============================================================
# 12. ALL MODEL FEATURES
# ============================================================

feature_columns = [
    col
    for col in df.columns
    if col not in DROP_COLUMNS
]

# Keep numeric columns only
feature_columns = [
    col
    for col in feature_columns
    if pd.api.types.is_numeric_dtype(df[col])
]

# Safety check
for forbidden in [
    "Target",
    "Target_Return",
]:
    if forbidden in feature_columns:
        raise ValueError(
            f"LEAKAGE ERROR: {forbidden} is still a feature."
        )

print("\nTotal features:", len(feature_columns))

# ============================================================
# 13. INTERNAL FEATURES
# ============================================================

internal_features = [
    col
    for col in feature_columns
    if col not in EXTERNAL_FEATURES
]

print(
    "Internal feature count:",
    len(internal_features)
)

print(
    "External feature count:",
    len(EXTERNAL_FEATURES)
)

# ============================================================
# 14. CREATE X AND y
# ============================================================

X = df[feature_columns].copy()
y = df["Target"].copy()

# Replace infinite values
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# 15. TIME-BASED SPLIT
# ============================================================

train_mask = (
    df["Date"] <= pd.Timestamp(TRAIN_END)
)

test_mask = (
    df["Date"] >= pd.Timestamp(TEST_START)
)

# Make sure the split does not overlap
if (
    train_mask & test_mask
).any():
    raise ValueError(
        "Training and testing periods overlap."
    )

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

# ============================================================
# 16. SPLIT INTERNAL / EXTERNAL
# ============================================================

internal_X_train = (
    X_train[internal_features]
    .copy()
)

internal_X_test = (
    X_test[internal_features]
    .copy()
)

external_X_train = (
    X_train[EXTERNAL_FEATURES]
    .copy()
)

external_X_test = (
    X_test[EXTERNAL_FEATURES]
    .copy()
)

# ============================================================
# 17. TRAINING MEDIANS
# ============================================================

internal_medians = (
    internal_X_train.median()
)

external_medians = (
    external_X_train.median()
)

# Fill using training medians only
internal_X_train = (
    internal_X_train
    .fillna(internal_medians)
)

internal_X_test = (
    internal_X_test
    .fillna(internal_medians)
)

external_X_train = (
    external_X_train
    .fillna(external_medians)
)

external_X_test = (
    external_X_test
    .fillna(external_medians)
)

# Any remaining NaN → 0
internal_X_train = (
    internal_X_train.fillna(0)
)

internal_X_test = (
    internal_X_test.fillna(0)
)

external_X_train = (
    external_X_train.fillna(0)
)

external_X_test = (
    external_X_test.fillna(0)
)

# ============================================================
# 18. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("BINARY TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining:")
print(
    "Internal:",
    internal_X_train.shape
)

print(
    "External:",
    external_X_train.shape
)

print(
    "Target:",
    y_train.shape
)

print("\nTesting:")

print(
    "Internal:",
    internal_X_test.shape
)

print(
    "External:",
    external_X_test.shape
)

print(
    "Target:",
    y_test.shape
)

print("\nTraining target:")
print(
    y_train.value_counts()
    .sort_index()
)

print("\nTesting target:")
print(
    y_test.value_counts()
    .sort_index()
)

print("\nTarget mapping:")
print("0 = SELL")
print("1 = BUY")

# ============================================================
# 19. CHECK FOR LEAKAGE
# ============================================================

print("\n" + "=" * 60)
print("FEATURE LEAKAGE CHECK")
print("=" * 60)

leakage_columns = [
    "Target",
    "Target_Return",
]

found_leakage = [
    col
    for col in leakage_columns
    if col in internal_features
    or col in EXTERNAL_FEATURES
]

if found_leakage:

    print(
        "WARNING: Leakage columns found:"
    )

    for col in found_leakage:
        print("-", col)

    raise ValueError(
        "Feature leakage detected."
    )

else:

    print(
        "PASS: Target and Target_Return "
        "are not model features."
    )

# ============================================================
# 20. CHECK REMAINING NaNs
# ============================================================

print("\nRemaining NaNs:")

print(
    "Internal train:",
    internal_X_train.isna().sum().sum()
)

print(
    "Internal test:",
    internal_X_test.isna().sum().sum()
)

print(
    "External train:",
    external_X_train.isna().sum().sum()
)

print(
    "External test:",
    external_X_test.isna().sum().sum()
)

# ============================================================
# 21. SAVE INTERNAL DATA
# ============================================================

joblib.dump(
    internal_X_train,
    f"{MODEL_DIR}/internal_X_train.pkl"
)

joblib.dump(
    internal_X_test,
    f"{MODEL_DIR}/internal_X_test.pkl"
)

joblib.dump(
    y_train,
    f"{MODEL_DIR}/internal_y_train.pkl"
)

joblib.dump(
    y_test,
    f"{MODEL_DIR}/internal_y_test.pkl"
)

joblib.dump(
    internal_features,
    f"{MODEL_DIR}/internal_features.pkl"
)

joblib.dump(
    internal_medians,
    f"{MODEL_DIR}/internal_training_medians.pkl"
)

# ============================================================
# 22. SAVE EXTERNAL DATA
# ============================================================

joblib.dump(
    external_X_train,
    f"{MODEL_DIR}/external_X_train.pkl"
)

joblib.dump(
    external_X_test,
    f"{MODEL_DIR}/external_X_test.pkl"
)

joblib.dump(
    y_train,
    f"{MODEL_DIR}/external_y_train.pkl"
)

joblib.dump(
    y_test,
    f"{MODEL_DIR}/external_y_test.pkl"
)

joblib.dump(
    EXTERNAL_FEATURES,
    f"{MODEL_DIR}/external_features.pkl"
)

joblib.dump(
    external_medians,
    f"{MODEL_DIR}/external_training_medians.pkl"
)

# ============================================================
# 23. FINAL REPORT
# ============================================================

print("\n" + "=" * 60)
print("BINARY DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nSaved internal files:")
print(
    "models/internal_X_train.pkl"
)
print(
    "models/internal_X_test.pkl"
)
print(
    "models/internal_y_train.pkl"
)
print(
    "models/internal_y_test.pkl"
)

print("\nSaved external files:")
print(
    "models/external_X_train.pkl"
)
print(
    "models/external_X_test.pkl"
)
print(
    "models/external_y_train.pkl"
)
print(
    "models/external_y_test.pkl"
)

print(
    "\nInternal feature count:",
    len(internal_features)
)

print(
    "External feature count:",
    len(EXTERNAL_FEATURES)
)

print("\nREADY FOR BINARY MODEL TRAINING.")
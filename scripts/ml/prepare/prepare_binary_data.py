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

df["Date"] = pd.to_datetime(df["Date"])
df["Ticker"] = df["Ticker"].astype(str)

print("\nOriginal shape:", df.shape)

# ============================================================
# 2. SORT
# ============================================================

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

# ============================================================
# 3. REMOVE HOLD
# ============================================================

print("\nOriginal target distribution:")
print(df["Target"].value_counts())

# Keep only BUY and SELL
df = df[
    df["Target"].isin(["BUY", "SELL"])
].copy()

print("\nAfter removing HOLD:")
print(df["Target"].value_counts())

# ============================================================
# 4. CONVERT TARGET
# ============================================================

label_map = {
    "SELL": 0,
    "BUY": 1
}

df["Target"] = df["Target"].map(label_map)

if df["Target"].isna().any():
    raise ValueError("Target contains invalid labels.")

# ============================================================
# 5. COLUMNS TO DROP
# ============================================================

DROP_COLUMNS = [
    "Target",
    "Target_Return",
    "Ticker",
]

DATE_COLUMNS = [
    "Date",
    "FinancialDate",
    "AnnouncementDate",
    "Last_Earnings_Date",
    "EarningsAnnouncementDate",
]

DROP_COLUMNS = DROP_COLUMNS + DATE_COLUMNS

# ============================================================
# 6. FINANCIAL FEATURES
# ============================================================

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

# ============================================================
# 7. FINANCIAL MISSING INDICATORS
# ============================================================

print("\nCreating financial missing indicators...")

for col in financial_features:
    df[f"{col}_Missing"] = df[col].isna().astype(int)

# ============================================================
# 8. RESPECT ANNOUNCEMENT DATE
# ============================================================

if "AnnouncementDate" in df.columns:

    df["AnnouncementDate"] = pd.to_datetime(
        df["AnnouncementDate"],
        errors="coerce"
    )

    for col in financial_features:

        invalid = (
            df["AnnouncementDate"].notna()
            & (df["Date"] < df["AnnouncementDate"])
        )

        df.loc[invalid, col] = np.nan

        df[col] = (
            df.groupby("Ticker")[col]
            .ffill()
        )

# ============================================================
# 9. DEFINE EXTERNAL FEATURES
# ============================================================

EXTERNAL_FEATURES = [
    "NIFTY_Close",
    "NIFTY_Volume",
    "NIFTY_Return",
    "VIX_Close",
    "VIX_Change",
]

EXTERNAL_FEATURES = [
    c for c in EXTERNAL_FEATURES
    if c in df.columns
]

print("\nExternal features:")
for col in EXTERNAL_FEATURES:
    print("-", col)

# ============================================================
# 10. ALL MODEL FEATURES
# ============================================================

feature_columns = [
    c for c in df.columns
    if c not in DROP_COLUMNS
]

# Keep numeric columns only
feature_columns = [
    c for c in feature_columns
    if pd.api.types.is_numeric_dtype(df[c])
]

print("\nTotal features:", len(feature_columns))

# ============================================================
# 11. INTERNAL FEATURES
# ============================================================

internal_features = [
    c for c in feature_columns
    if c not in EXTERNAL_FEATURES
]

# ============================================================
# 12. FINAL CLEANUP
# ============================================================

X = df[feature_columns].copy()
y = df["Target"].copy()

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# ============================================================
# 13. TIME-BASED SPLIT
# ============================================================

train_mask = (
    df["Date"] <= pd.Timestamp(TRAIN_END)
)

test_mask = (
    df["Date"] >= pd.Timestamp(TEST_START)
)

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

# ============================================================
# 14. SPLIT INTERNAL / EXTERNAL
# ============================================================

internal_X_train = X_train[internal_features].copy()
internal_X_test = X_test[internal_features].copy()

external_X_train = X_train[EXTERNAL_FEATURES].copy()
external_X_test = X_test[EXTERNAL_FEATURES].copy()

# ============================================================
# 15. TRAINING MEDIANS
# ============================================================

internal_medians = internal_X_train.median()
external_medians = external_X_train.median()

internal_X_train = internal_X_train.fillna(
    internal_medians
)

internal_X_test = internal_X_test.fillna(
    internal_medians
)

external_X_train = external_X_train.fillna(
    external_medians
)

external_X_test = external_X_test.fillna(
    external_medians
)

internal_X_train = internal_X_train.fillna(0)
internal_X_test = internal_X_test.fillna(0)

external_X_train = external_X_train.fillna(0)
external_X_test = external_X_test.fillna(0)

# ============================================================
# 16. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("BINARY TRAIN / TEST SPLIT")
print("=" * 60)

print("\nTraining:")
print("Internal:", internal_X_train.shape)
print("External:", external_X_train.shape)
print("Target:", y_train.shape)

print("\nTesting:")
print("Internal:", internal_X_test.shape)
print("External:", external_X_test.shape)
print("Target:", y_test.shape)

print("\nTraining target:")
print(
    y_train.value_counts().sort_index()
)

print("\nTesting target:")
print(
    y_test.value_counts().sort_index()
)

print("\nTarget mapping:")
print("0 = SELL")
print("1 = BUY")

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
# 17. SAVE INTERNAL DATA
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
# 18. SAVE EXTERNAL DATA
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
# 19. COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("BINARY DATA PREPARATION COMPLETE")
print("=" * 60)

print("\nSaved internal files:")
print("models/internal_X_train.pkl")
print("models/internal_X_test.pkl")
print("models/internal_y_train.pkl")
print("models/internal_y_test.pkl")

print("\nSaved external files:")
print("models/external_X_train.pkl")
print("models/external_X_test.pkl")
print("models/external_y_train.pkl")
print("models/external_y_test.pkl")

print("\nInternal feature count:", len(internal_features))
print("External feature count:", len(EXTERNAL_FEATURES))

print("\nREADY FOR BINARY MODEL TRAINING.")
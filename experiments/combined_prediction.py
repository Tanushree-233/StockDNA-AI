import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/processed/internal/final_ml_dataset.csv"

INTERNAL_MODEL_PATH = "models/binary_internal_xgboost.pkl"
EXTERNAL_MODEL_PATH = "models/binary_external_xgboost.pkl"

# Change this to test another stock
TICKER = "RELIANCE"

# Equal contribution for now
INTERNAL_WEIGHT = 0.50
EXTERNAL_WEIGHT = 0.50


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 60)
print("COMBINED BUY / SELL PREDICTION")
print("=" * 60)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading models...")

if not os.path.exists(INTERNAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Internal model not found: {INTERNAL_MODEL_PATH}"
    )

if not os.path.exists(EXTERNAL_MODEL_PATH):
    raise FileNotFoundError(
        f"External model not found: {EXTERNAL_MODEL_PATH}"
    )

internal_model = joblib.load(INTERNAL_MODEL_PATH)
external_model = joblib.load(EXTERNAL_MODEL_PATH)

print("Internal model loaded:")
print(INTERNAL_MODEL_PATH)

print("\nExternal model loaded:")
print(EXTERNAL_MODEL_PATH)


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Ticker"] = df["Ticker"].astype(str)

print("\nDataset shape:", df.shape)


# ============================================================
# SELECT STOCK
# ============================================================

stock_df = df[
    df["Ticker"].str.upper() == TICKER.upper()
].copy()

if stock_df.empty:
    raise ValueError(
        f"No data found for ticker: {TICKER}"
    )

# Sort by date
stock_df = stock_df.sort_values("Date")

# Take latest available record
latest = stock_df.iloc[-1].copy()


# ============================================================
# STOCK INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("STOCK")
print("=" * 60)

print("Ticker:", TICKER)
print("Date:", latest["Date"])


# ============================================================
# GET MODEL FEATURES
# ============================================================

internal_features = list(
    internal_model.feature_names_in_
)

external_features = list(
    external_model.feature_names_in_
)

print("\nInternal features:", len(internal_features))
print("External features:", len(external_features))


# ============================================================
# CREATE MISSINGNESS INDICATORS
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

print("\nCreating missingness indicators...")

for col in financial_features:

    if col in latest.index:

        latest[f"{col}_Missing"] = int(
            pd.isna(latest[col])
        )


# ============================================================
# CHECK FEATURES
# ============================================================

missing_internal = [
    col
    for col in internal_features
    if col not in latest.index
]

missing_external = [
    col
    for col in external_features
    if col not in latest.index
]

if missing_internal:

    print("\nMissing internal features:")
    for col in missing_internal:
        print("-", col)

    raise ValueError(
        f"Missing internal features: {missing_internal}"
    )


if missing_external:

    print("\nMissing external features:")
    for col in missing_external:
        print("-", col)

    raise ValueError(
        f"Missing external features: {missing_external}"
    )


print("\nAll model features are available.")


# ============================================================
# CREATE INTERNAL INPUT
# ============================================================

X_internal = pd.DataFrame(
    [
        [
            latest[col]
            for col in internal_features
        ]
    ],
    columns=internal_features
)


# ============================================================
# CREATE EXTERNAL INPUT
# ============================================================

X_external = pd.DataFrame(
    [
        [
            latest[col]
            for col in external_features
        ]
    ],
    columns=external_features
)


# ============================================================
# CLEAN INTERNAL DATA
# ============================================================

X_internal = X_internal.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# CLEAN EXTERNAL DATA
# ============================================================

X_external = X_external.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

# For the first working prediction pipeline,
# remaining missing values are filled with 0.

X_internal = X_internal.fillna(0)

X_external = X_external.fillna(0)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

if X_internal.isna().sum().sum() > 0:
    raise ValueError(
        "Internal input still contains NaN values."
    )

if X_external.isna().sum().sum() > 0:
    raise ValueError(
        "External input still contains NaN values."
    )


# ============================================================
# INTERNAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("INTERNAL MODEL PREDICTION")
print("=" * 60)

internal_probabilities = (
    internal_model.predict_proba(X_internal)[0]
)


# Class 0 = SELL
# Class 1 = BUY

internal_sell_probability = float(
    internal_probabilities[0]
)

internal_buy_probability = float(
    internal_probabilities[1]
)


if (
    internal_buy_probability
    >= internal_sell_probability
):

    internal_prediction = "BUY"

else:

    internal_prediction = "SELL"


print(
    f"SELL Probability : "
    f"{internal_sell_probability * 100:.2f}%"
)

print(
    f"BUY Probability  : "
    f"{internal_buy_probability * 100:.2f}%"
)

print(
    f"Internal Signal  : "
    f"{internal_prediction}"
)


# ============================================================
# EXTERNAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("EXTERNAL MODEL PREDICTION")
print("=" * 60)

external_probabilities = (
    external_model.predict_proba(X_external)[0]
)


# Class 0 = SELL
# Class 1 = BUY

external_sell_probability = float(
    external_probabilities[0]
)

external_buy_probability = float(
    external_probabilities[1]
)


if (
    external_buy_probability
    >= external_sell_probability
):

    external_prediction = "BUY"

else:

    external_prediction = "SELL"


print(
    f"SELL Probability : "
    f"{external_sell_probability * 100:.2f}%"
)

print(
    f"BUY Probability  : "
    f"{external_buy_probability * 100:.2f}%"
)

print(
    f"External Signal  : "
    f"{external_prediction}"
)


# ============================================================
# COMBINED PREDICTION
# ============================================================

print("\n" + "=" * 60)
print("COMBINED PREDICTION")
print("=" * 60)


combined_buy_probability = (
    internal_buy_probability
    * INTERNAL_WEIGHT
    +
    external_buy_probability
    * EXTERNAL_WEIGHT
)


combined_sell_probability = (
    internal_sell_probability
    * INTERNAL_WEIGHT
    +
    external_sell_probability
    * EXTERNAL_WEIGHT
)


# Final signal

if (
    combined_buy_probability
    >= combined_sell_probability
):

    final_prediction = "BUY"

else:

    final_prediction = "SELL"


# Confidence

final_confidence = (
    max(
        combined_buy_probability,
        combined_sell_probability
    )
    * 100
)


print(
    f"Internal Weight : "
    f"{INTERNAL_WEIGHT * 100:.0f}%"
)

print(
    f"External Weight : "
    f"{EXTERNAL_WEIGHT * 100:.0f}%"
)

print(
    f"\nCombined SELL Probability : "
    f"{combined_sell_probability * 100:.2f}%"
)

print(
    f"Combined BUY Probability  : "
    f"{combined_buy_probability * 100:.2f}%"
)

print(
    f"\nFINAL PREDICTION : "
    f"{final_prediction}"
)

print(
    f"CONFIDENCE       : "
    f"{final_confidence:.2f}%"
)


# ============================================================
# FACTOR BIFURCATION
# ============================================================

print("\n" + "=" * 60)
print("FACTOR BIFURCATION")
print("=" * 60)


internal_score = (
    internal_buy_probability * 100
)

external_score = (
    external_buy_probability * 100
)


print(
    f"Internal Factor Score : "
    f"{internal_score:.2f}/100"
)

print(
    f"External Factor Score : "
    f"{external_score:.2f}/100"
)


# ============================================================
# FACTOR INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("FACTOR INTERPRETATION")
print("=" * 60)


if internal_prediction == "BUY":

    print(
        "Internal factors are supporting BUY."
    )

else:

    print(
        "Internal factors are supporting SELL."
    )


if external_prediction == "BUY":

    print(
        "External factors are supporting BUY."
    )

else:

    print(
        "External factors are supporting SELL."
    )


# ============================================================
# AGREEMENT CHECK
# ============================================================

if internal_prediction == external_prediction:

    print(
        "\nInternal and External models AGREE."
    )

else:

    print(
        "\nInternal and External models DISAGREE."
    )


# ============================================================
# RESULT DICTIONARY
# ============================================================

result = {

    "Ticker": TICKER,

    "Date": latest["Date"],

    # Internal
    "Internal_Prediction":
        internal_prediction,

    "Internal_BUY_Probability":
        internal_buy_probability,

    "Internal_SELL_Probability":
        internal_sell_probability,

    "Internal_Score":
        internal_score,

    # External
    "External_Prediction":
        external_prediction,

    "External_BUY_Probability":
        external_buy_probability,

    "External_SELL_Probability":
        external_sell_probability,

    "External_Score":
        external_score,

    # Combined
    "Final_Prediction":
        final_prediction,

    "Combined_BUY_Probability":
        combined_buy_probability,

    "Combined_SELL_Probability":
        combined_sell_probability,

    "Confidence":
        final_confidence,

    # Agreement
    "Models_Agree":
        internal_prediction == external_prediction,
}


# ============================================================
# SAVE RESULT
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

result_df = pd.DataFrame(
    [result]
)

output_path = (
    "models/combined_prediction_result.csv"
)

result_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)

print(
    f"Stock       : {TICKER}"
)

print(
    f"Date        : {latest['Date']}"
)

print(
    f"Internal    : {internal_prediction}"
)

print(
    f"External    : {external_prediction}"
)

print(
    f"Final       : {final_prediction}"
)

print(
    f"Confidence  : {final_confidence:.2f}%"
)

print(
    f"\nInternal Score : "
    f"{internal_score:.2f}/100"
)

print(
    f"External Score : "
    f"{external_score:.2f}/100"
)

print(
    "\nResult saved:"
)

print(output_path)


print("\n" + "=" * 60)
print("COMBINED PREDICTION COMPLETE")
print("=" * 60)
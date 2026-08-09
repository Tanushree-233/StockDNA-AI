import os
import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "models/final_xgboost_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

LABELS = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model


def load_features():
    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature list not found: {FEATURES_PATH}"
        )

    features = joblib.load(FEATURES_PATH)

    return list(features)


# ============================================================
# PREPARE INPUT
# ============================================================

def prepare_input(df, feature_columns):

    df = df.copy()
    # --------------------------------------------------------
# Recreate financial missing indicators exactly as during
# training.
# --------------------------------------------------------

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

    for column in financial_features:

        missing_column = f"{column}_Missing"

        if column in df.columns:
            df[missing_column] = (
                df[column].isna().astype(int)
            )
        else:
            # If the financial feature itself isn't supplied,
            # mark it as missing.
            df[missing_column] = 1
    # --------------------------------------------------------
    # Remove columns that must never enter the model
    # --------------------------------------------------------

    forbidden = [
        "Target",
        "Target_Return",
        "Date",
        "Ticker",
        "FinancialDate",
        "AnnouncementDate",
        "Last_Earnings_Date",
        "EarningsAnnouncementDate",
    ]

    for column in forbidden:
        if column in df.columns:
            df = df.drop(columns=[column])

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing required model features:\n"
            + "\n".join(missing_features)
        )

    # --------------------------------------------------------
    # Keep exactly the training feature order
    # --------------------------------------------------------

    X = df[feature_columns].copy()

    # --------------------------------------------------------
    # Convert everything to numeric
    # --------------------------------------------------------

    X = X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinite values
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Fill missing values
    #
    # Use the medians saved during training.
    # --------------------------------------------------------

    medians_path = "models/training_medians.pkl"

    if os.path.exists(medians_path):

        medians = joblib.load(medians_path)

        for column in feature_columns:

            if column in medians:
                X[column] = X[column].fillna(medians[column])

            else:
                # Training median unavailable.
                # Use the current feature median as fallback.
                X[column] = X[column].fillna(X[column].median())

    else:

        X = X.fillna(X.median())

    # --------------------------------------------------------
    # Final fallback
    #
    # Some financial features may be completely NaN for the
    # latest observation. If no median exists, use 0.
    # --------------------------------------------------------

    X = X.fillna(0)

    # --------------------------------------------------------
    # Final safety check
    # --------------------------------------------------------

    if X.isna().sum().sum() > 0:

        missing = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "NaN values remain in features:\n"
            + "\n".join(missing)
        )

    if np.isinf(X.values).any():

        raise ValueError(
            "Infinite values remain in input."
        )

    return X


# ============================================================
# PREDICT
# ============================================================

def predict(df):

    model = load_model()

    feature_columns = load_features()

    X = prepare_input(
        df,
        feature_columns
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(X)

    prediction = np.asarray(
        prediction
    ).astype(int)

    labels = [
        LABELS[int(value)]
        for value in prediction
    ]

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probabilities = None

    if hasattr(
        model,
        "predict_proba"
    ):

        probabilities = model.predict_proba(
            X
        )

    # --------------------------------------------------------
    # Build result
    # --------------------------------------------------------

    result = pd.DataFrame({
        "Prediction": labels
    })

    if probabilities is not None:

        result["SELL_Probability"] = (
            probabilities[:, 0]
        )

        result["HOLD_Probability"] = (
            probabilities[:, 1]
        )

        result["BUY_Probability"] = (
            probabilities[:, 2]
        )

        result["Confidence"] = (
            probabilities.max(axis=1)
        )

    # Keep useful identifiers if supplied
    for column in [
        "Ticker",
        "Date",
    ]:

        if column in df.columns:

            result.insert(
                0,
                column,
                df[column].values
            )

    return result


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("STOCKDNA-AI PRODUCTION PREDICTION")
    print("=" * 60)

    print("\nModel:")
    print(MODEL_PATH)

    print("\nFeature list:")
    print(FEATURES_PATH)

    print("\nNOTE:")
    print(
        "Provide a dataframe containing the same "
        "87 features used during training."
    )

    print("\nPrediction module loaded successfully.")

    print("\nUsage from Python:")
    print(
        "from scripts.ml.predict.predict_stock "
        "import predict"
    )

    print("\n" + "=" * 60)
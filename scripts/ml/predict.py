import os
import joblib
import pandas as pd


MODEL_PATH = "models/final_xgboost_model.pkl"

FEATURES = [
    "High",
    "Low",
    "Open",
    "Volume",
    "Daily_Return",
    "SMA20",
    "SMA50",
    "SMA100",
    "SMA200",
    "EMA20",
    "EMA50",
    "EMA100",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_High",
    "BB_Low",
    "BB_Middle",
    "ATR",
    "Volatility",
    "Lag_Close_1",
    "Lag_Close_3",
    "Lag_Close_5",
    "Lag_Volume_1",
    "Momentum_5",
    "Momentum_10",
    "Momentum_20",
    "ROC",
    "Volume_Change",
    "Volume_MA20",
    "High_Low_Spread",
    "Open_Close_Spread",
]

CLASS_NAMES = {
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}


def predict(df):
    model = joblib.load(MODEL_PATH)

    missing = [col for col in FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[FEATURES]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    result = pd.DataFrame({
        "Date": df["Date"].values,
        "Predicted_Class": [
            CLASS_NAMES[int(pred)]
            for pred in predictions
        ],
        "Sell_Probability": probabilities[:, 0],
        "Hold_Probability": probabilities[:, 1],
        "Buy_Probability": probabilities[:, 2],
    })

    return result


if __name__ == "__main__":

    input_path = "data/final/master_dataset.csv"

    df = pd.read_csv(input_path)

    result = predict(df)

    print("\n" + "=" * 60)
    print("STOCKDNA PREDICTIONS")
    print("=" * 60)

    print(result.tail(10).to_string(index=False))

    output_path = "data/processed/predictions.csv"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result.to_csv(output_path, index=False)

    print("\nPredictions saved to:")
    print(output_path)
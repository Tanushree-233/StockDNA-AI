import pandas as pd

from predict_stock import predict

DATA_PATH = "data/processed/internal/final_ml_dataset.csv"


def main():

    print("\n" + "=" * 60)
    print("STOCKDNA-AI PREDICTION TEST")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    df["Date"] = pd.to_datetime(df["Date"])

    # Get latest available row for each company
    latest = (
        df.sort_values(["Ticker", "Date"])
        .groupby("Ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    print("\nLatest rows:")
    print(
        latest[
            ["Ticker", "Date", "Close"]
        ].to_string(index=False)
    )

    # Run prediction
    result = predict(latest)

    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)

    display_columns = [
        "Ticker",
        "Date",
        "Prediction",
        "SELL_Probability",
        "HOLD_Probability",
        "BUY_Probability",
        "Confidence",
    ]

    print(
        result[display_columns].to_string(index=False)
    )


if __name__ == "__main__":
    main()
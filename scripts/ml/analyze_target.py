import pandas as pd


DATASET = "data/final/master_dataset.csv"

df = pd.read_csv(DATASET)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


print("=" * 60)
print("TARGET HORIZON ANALYSIS")
print("=" * 60)


HORIZONS = [1, 5, 10, 20]


for horizon in HORIZONS:

    print("\n")
    print("=" * 60)
    print(f"HORIZON: {horizon} TRADING DAY(S)")
    print("=" * 60)

    # Future return
    future_return = (
        df.groupby("Ticker")["Close"]
        .shift(-horizon) / df["Close"]
        - 1
    )

    print("\nReturn statistics:")
    print(future_return.describe())

    sell = (future_return < -0.01).mean() * 100
    hold = (
        (future_return >= -0.01)
        & (future_return <= 0.01)
    ).mean() * 100
    buy = (future_return > 0.01).mean() * 100

    print("\nUsing ±1% threshold:")

    print(f"SELL: {sell:.2f}%")
    print(f"HOLD: {hold:.2f}%")
    print(f"BUY : {buy:.2f}%")


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
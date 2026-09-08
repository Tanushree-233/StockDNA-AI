import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from data_tool.pipeline import run_data_tool


def main():
    data_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "prices",
        "TCS.csv"
    )

    print("\n========================================")
    print("STOCKDNA DATA TOOL TEST")
    print("========================================")

    print(f"\nLoading data from:")
    print(data_path)

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"\nTCS.csv was not found at:\n{data_path}"
        )

    df = pd.read_csv(
        data_path,
        parse_dates=["Date"]
    )

    print(f"\nLoaded {len(df)} rows.")

    print("\nInput columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))

    result = run_data_tool(
        df,
        model_name="xgboost"
    )


    print("\n========================================")
    print("CLASSIFICATION RESULTS")
    print("========================================")

    print(
        result.tail(10).to_string(index=False)
    )

    output_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "data_tool_predictions.csv"
    )

    result.to_csv(
        output_path,
        index=False
    )

    print("\n========================================")
    print("COMPLETE")
    print("========================================")

    print(
        f"\nPredictions saved to:\n{output_path}"
    )


if __name__ == "__main__":
    main()
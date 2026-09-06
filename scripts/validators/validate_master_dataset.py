import pandas as pd
import numpy as np

MASTER_DATASET = "data/final/master_dataset.csv"

validation_results = []


def load_dataset():
    return pd.read_csv(MASTER_DATASET)


def record_result(name, passed):
    validation_results.append((name, passed))


def dataset_info(df):

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumn Names:\n")
    print(df.columns.tolist())


def check_required_columns(df):

    required_columns = [
        "Date",
        "Ticker",
        "Close",
        "High",
        "Low",
        "Open",
        "Volume",
        "Target_Return",
        "Target"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    print("\n" + "=" * 50)
    print("REQUIRED COLUMN CHECK")
    print("=" * 50)

    passed = len(missing_columns) == 0

    if passed:
        print("PASS: All required columns are present.")
    else:
        print("FAIL: Missing required columns:")
        print(missing_columns)

    record_result("Required columns", passed)


def check_missing(df):

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    passed = len(missing) == 0

    if passed:
        print("PASS: No missing values.")
    else:
        print("FAIL: Missing values found:")
        print(missing)

    record_result("Missing values", passed)


def check_duplicates(df):

    print("\n" + "=" * 50)
    print("DUPLICATES")
    print("=" * 50)

    duplicate_rows = df.duplicated().sum()

    print("Duplicate rows:", duplicate_rows)

    passed = duplicate_rows == 0

    if passed:
        print("PASS: No duplicate rows.")
    else:
        print("FAIL: Duplicate rows found.")

    record_result("Duplicate rows", passed)


def check_ticker_date_duplicates(df):

    print("\n" + "=" * 50)
    print("TICKER + DATE DUPLICATES")
    print("=" * 50)

    if "Ticker" not in df.columns or "Date" not in df.columns:
        print("FAIL: Ticker or Date column is missing.")
        record_result("Ticker + Date duplicates", False)
        return

    duplicate_count = df.duplicated(
        subset=["Ticker", "Date"]
    ).sum()

    print(
        "Duplicate Ticker + Date rows:",
        duplicate_count
    )

    passed = duplicate_count == 0

    if passed:
        print(
            "PASS: No duplicate Ticker + Date combinations."
        )
    else:
        print(
            "FAIL: Duplicate Ticker + Date combinations found."
        )

    record_result("Ticker + Date duplicates", passed)


def check_dates(df):

    print("\n" + "=" * 50)
    print("DATE VALIDATION")
    print("=" * 50)

    if "Date" not in df.columns:
        print("FAIL: Date column is missing.")
        record_result("Date validation", False)
        return

    converted_dates = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    invalid_dates = converted_dates.isna().sum()

    print("Invalid dates:", invalid_dates)

    passed = invalid_dates == 0

    if passed:
        print("PASS: All dates are valid.")
        print("Minimum date:", converted_dates.min())
        print("Maximum date:", converted_dates.max())
    else:
        print("FAIL: Invalid dates found.")

    record_result("Date validation", passed)


def check_tickers(df):

    print("\n" + "=" * 50)
    print("TICKER VALIDATION")
    print("=" * 50)

    if "Ticker" not in df.columns:
        print("FAIL: Ticker column is missing.")
        record_result("Ticker validation", False)
        return

    ticker_series = (
        df["Ticker"]
        .astype("string")
        .str.strip()
    )

    blank_tickers = (
        ticker_series.isna() |
        (ticker_series == "")
    ).sum()

    unique_tickers = ticker_series.nunique(
        dropna=True
    )

    print("Blank Tickers:", blank_tickers)
    print("Unique Tickers:", unique_tickers)

    passed = blank_tickers == 0

    if passed:
        print("PASS: No blank Ticker values.")
    else:
        print("FAIL: Blank Ticker values found.")

    print("\nTickers found:")
    print(
        sorted(
            ticker_series
            .dropna()
            .unique()
        )
    )

    record_result("Ticker validation", passed)


def check_date_order(df):

    print("\n" + "=" * 50)
    print("DATE ORDER VALIDATION")
    print("=" * 50)

    if "Ticker" not in df.columns or "Date" not in df.columns:
        print("FAIL: Ticker or Date column is missing.")
        record_result("Date ordering", False)
        return

    temp = df.copy()

    temp["Date"] = pd.to_datetime(
        temp["Date"],
        errors="coerce"
    )

    temp = temp.sort_values(
        ["Ticker", "Date"]
    )

    order_check = (
        temp.groupby("Ticker")["Date"]
        .apply(
            lambda x: x.is_monotonic_increasing
        )
    )

    invalid_tickers = order_check[
        ~order_check
    ]

    passed = len(invalid_tickers) == 0

    if passed:
        print(
            "PASS: Dates are in ascending order for all tickers."
        )
    else:
        print(
            "FAIL: Date ordering problems found for:"
        )
        print(invalid_tickers.index.tolist())

    record_result("Date ordering", passed)


def check_numeric_ranges(df):

    print("\n" + "=" * 50)
    print("NUMERIC / RANGE VALIDATION")
    print("=" * 50)

    passed = True

    price_columns = [
        "Close",
        "High",
        "Low",
        "Open"
    ]

    for column in price_columns:

        if column not in df.columns:
            print(
                f"FAIL: {column} column is missing."
            )
            passed = False
            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid_count = (
            values <= 0
        ).sum()

        print(
            f"{column} <= 0:",
            invalid_count
        )

        if invalid_count > 0:
            passed = False

    if "Volume" in df.columns:

        volume = pd.to_numeric(
            df["Volume"],
            errors="coerce"
        )

        invalid_volume = (
            volume < 0
        ).sum()

        print(
            "Negative Volume:",
            invalid_volume
        )

        if invalid_volume > 0:
            passed = False

    if (
        "High" in df.columns
        and "Low" in df.columns
    ):

        invalid_high_low = (
            df["High"] < df["Low"]
        ).sum()

        print(
            "High < Low:",
            invalid_high_low
        )

        if invalid_high_low > 0:
            passed = False

    if "RSI" in df.columns:

        invalid_rsi = (
            (df["RSI"] < 0) |
            (df["RSI"] > 100)
        ).sum()

        print(
            "RSI outside 0-100:",
            invalid_rsi
        )

        if invalid_rsi > 0:
            passed = False

    if "Target_Return" in df.columns:

        target_return = pd.to_numeric(
            df["Target_Return"],
            errors="coerce"
        )

        invalid_target_return = (
            target_return.isna()
        ).sum()

        print(
            "Invalid Target_Return:",
            invalid_target_return
        )

        if invalid_target_return > 0:
            passed = False

    if "Target" in df.columns:

        valid_targets = {
            "BUY",
            "HOLD",
            "SELL"
        }

        target_values = (
            df["Target"]
            .astype("string")
            .str.strip()
        )

        invalid_targets = (
            ~target_values.isin(valid_targets)
        ).sum()

        print(
            "Invalid Target values:",
            invalid_targets
        )

        if invalid_targets > 0:
            passed = False

    if passed:
        print(
            "\nPASS: Numeric and range validation passed."
        )
    else:
        print(
            "\nFAIL: Numeric or range validation failed."
        )

    record_result(
        "Numeric / range validation",
        passed
    )


def check_infinite(df):

    print("\n" + "=" * 50)
    print("INFINITE VALUES")
    print("=" * 50)

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite = np.isinf(
        numeric_df
    ).sum().sum()

    print(
        "Infinite values:",
        infinite
    )

    passed = infinite == 0

    if passed:
        print(
            "PASS: No infinite values."
        )
    else:
        print(
            "FAIL: Infinite values found."
        )

    record_result("Infinite values", passed)


def check_dtypes(df):

    print("\n" + "=" * 50)
    print("DATA TYPES")
    print("=" * 50)

    print(df.dtypes)


def summary_statistics(df):

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(df.describe())


def final_validation_summary():

    print("\n" + "=" * 50)
    print("FINAL VALIDATION RESULT")
    print("=" * 50)

    failed_checks = [
        name
        for name, passed in validation_results
        if not passed
    ]

    total_checks = len(validation_results)
    passed_checks = total_checks - len(failed_checks)

    print(f"Checks passed : {passed_checks}")
    print(f"Checks failed : {len(failed_checks)}")

    if failed_checks:

        print("\nFAILED CHECKS:")

        for check in failed_checks:
            print(f"- {check}")

        print("\nOVERALL RESULT: FAIL")

    else:

        print("\nOVERALL RESULT: PASS")


def main():

    df = load_dataset()

    dataset_info(df)

    check_required_columns(df)

    check_missing(df)

    check_duplicates(df)

    check_ticker_date_duplicates(df)

    check_dates(df)

    check_tickers(df)

    check_date_order(df)

    check_numeric_ranges(df)

    check_infinite(df)

    check_dtypes(df)

    summary_statistics(df)

    final_validation_summary()


if __name__ == "__main__":
    main()
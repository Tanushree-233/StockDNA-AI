import os
import pandas as pd
import numpy as np


PRICE_FOLDER = "data/processed/prices"

REQUIRED_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume"
]


def validate_price_file(filepath):
    """
    Validate one processed stock price CSV.
    """

    filename = os.path.basename(filepath)

    print("\n" + "=" * 60)
    print(f"VALIDATING: {filename}")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"FAIL: Could not read file: {e}")
        return False

    if df.empty:
        print("FAIL: Dataset is empty.")
        return False

    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    valid = True

    # ---------------------------------------------------------
    # 2. REQUIRED COLUMNS
    # ---------------------------------------------------------

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        print("FAIL: Missing required columns:")
        print(missing_columns)
        valid = False
    else:
        print("PASS: Required columns present.")

    # ---------------------------------------------------------
    # 3. DATE VALIDATION
    # ---------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        invalid_dates = df["Date"].isna().sum()

        if invalid_dates > 0:
            print(
                f"FAIL: Invalid/missing dates: {invalid_dates}"
            )
            valid = False
        else:
            print("PASS: All dates are valid.")

        if df["Date"].duplicated().any():

            duplicates = df["Date"].duplicated().sum()

            print(
                f"FAIL: Duplicate dates: {duplicates}"
            )
            valid = False

        else:
            print("PASS: No duplicate dates.")

        if not df["Date"].is_monotonic_increasing:
            print("FAIL: Dates are not sorted.")
            valid = False
        else:
            print("PASS: Dates are sorted.")

    # ---------------------------------------------------------
    # 4. MISSING VALUES
    # ---------------------------------------------------------

    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:

        print("\nFAIL: Missing values found:")
        print(missing)

        valid = False

    else:
        print("PASS: No missing values.")

    # ---------------------------------------------------------
    # 5. DUPLICATE ROWS
    # ---------------------------------------------------------

    duplicate_rows = df.duplicated().sum()

    if duplicate_rows > 0:

        print(
            f"FAIL: Duplicate rows: {duplicate_rows}"
        )

        valid = False

    else:
        print("PASS: No duplicate rows.")

    # ---------------------------------------------------------
    # 6. NUMERIC COLUMNS
    # ---------------------------------------------------------

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume"
    ]

    for column in numeric_columns:

        if column not in df.columns:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        invalid = converted.isna().sum()

        if invalid > 0:

            print(
                f"FAIL: {column} contains "
                f"{invalid} non-numeric values."
            )

            valid = False

        else:
            print(
                f"PASS: {column} is numeric."
            )

    # ---------------------------------------------------------
    # 7. INFINITE VALUES
    # ---------------------------------------------------------

    numeric_df = df.select_dtypes(
        include=np.number
    )

    infinite_count = np.isinf(
        numeric_df
    ).sum().sum()

    if infinite_count > 0:

        print(
            f"FAIL: Infinite values: "
            f"{infinite_count}"
        )

        valid = False

    else:
        print("PASS: No infinite values.")

    # ---------------------------------------------------------
    # 8. PRICE RANGE CHECKS
    # ---------------------------------------------------------

    price_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close"
    ]

    for column in price_columns:

        if column not in df.columns:
            continue

        if (df[column] <= 0).any():

            count = (
                df[column] <= 0
            ).sum()

            print(
                f"FAIL: {column} has "
                f"{count} non-positive values."
            )

            valid = False

    # ---------------------------------------------------------
    # 9. HIGH / LOW CONSISTENCY
    # ---------------------------------------------------------

    if all(
        col in df.columns
        for col in ["High", "Low"]
    ):

        invalid_high_low = (
            df["High"] < df["Low"]
        ).sum()

        if invalid_high_low > 0:

            print(
                "FAIL: High < Low in "
                f"{invalid_high_low} rows."
            )

            valid = False

        else:
            print(
                "PASS: High >= Low for all rows."
            )

    # ---------------------------------------------------------
    # 10. FINAL RESULT
    # ---------------------------------------------------------

    print("\n" + "-" * 60)

    if valid:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL")

    print("-" * 60)

    return valid


def main():

    print("=" * 60)
    print("STOCK PRICE DATA VALIDATION")
    print("=" * 60)

    if not os.path.exists(PRICE_FOLDER):

        print(
            f"FAIL: Folder does not exist: "
            f"{PRICE_FOLDER}"
        )

        return

    files = sorted(
        file
        for file in os.listdir(PRICE_FOLDER)
        if file.endswith(".csv")
    )

    if not files:

        print(
            "FAIL: No CSV price files found."
        )

        return

    results = []

    for file in files:

        filepath = os.path.join(
            PRICE_FOLDER,
            file
        )

        result = validate_price_file(
            filepath
        )

        results.append(
            (file, result)
        )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0

    for filename, result in results:

        status = "PASS" if result else "FAIL"

        print(
            f"{filename:<30} {status}"
        )

        if result:
            passed += 1
        else:
            failed += 1

    print("\nFiles passed :", passed)
    print("Files failed :", failed)

    if failed == 0:
        print("\nOVERALL RESULT: PASS")
    else:
        print("\nOVERALL RESULT: FAIL")


if __name__ == "__main__":
    main()
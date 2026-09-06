import os
import sys
import pandas as pd
from sqlalchemy import text

from backend.database.database import engine
from backend.database.models import StockData


MASTER_DATASET = "data/final/master_dataset.csv"


def verify_database():
    print("=" * 60)
    print("STOCKDNA AI - DATABASE VERIFICATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Check database connection
    # ---------------------------------------------------------
    print("\n[1] Checking database connection...")

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        print("PASS - Database connection successful")

    except Exception as e:
        print("FAIL - Database connection failed")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 2. Check stock_data table
    # ---------------------------------------------------------
    print("\n[2] Checking stock_data table...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='stock_data'"
                )
            )

            table = result.fetchone()

        if table:
            print("PASS - stock_data table exists")
        else:
            print("FAIL - stock_data table does not exist")
            return False

    except Exception as e:
        print("FAIL - Could not check stock_data table")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 3. Load master dataset for comparison
    # ---------------------------------------------------------
    print("\n[3] Loading master dataset...")

    if not os.path.exists(MASTER_DATASET):
        print(f"FAIL - Master dataset not found: {MASTER_DATASET}")
        return False

    try:
        master_df = pd.read_csv(MASTER_DATASET)

        print(f"Master dataset rows: {len(master_df)}")

    except Exception as e:
        print("FAIL - Could not read master dataset")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 4. Check database row count
    # ---------------------------------------------------------
    print("\n[4] Checking database row count...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT COUNT(*) FROM stock_data")
            )

            db_count = result.scalar()

        expected_count = len(master_df)

        print(f"Expected rows : {expected_count}")
        print(f"Database rows : {db_count}")

        if db_count == expected_count:
            print("PASS - Database row count matches master dataset")
        else:
            print("FAIL - Database row count does not match")
            return False

    except Exception as e:
        print("FAIL - Could not count database records")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 5. Check ticker count and names
    # ---------------------------------------------------------
    print("\n[5] Checking tickers...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT DISTINCT Ticker "
                    "FROM stock_data "
                    "ORDER BY Ticker"
                )
            )

            db_tickers = [row[0] for row in result.fetchall()]

        expected_tickers = sorted(
            master_df["Ticker"]
            .astype(str)
            .str.strip()
            .str.upper()
            .unique()
            .tolist()
        )

        print(f"Expected tickers : {expected_tickers}")
        print(f"Database tickers : {db_tickers}")

        if db_tickers == expected_tickers:
            print("PASS - Tickers match")
        else:
            print("FAIL - Tickers do not match")
            return False

    except Exception as e:
        print("FAIL - Could not verify tickers")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 6. Check date range
    # ---------------------------------------------------------
    print("\n[6] Checking date range...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    "SELECT MIN(Date), MAX(Date) "
                    "FROM stock_data"
                )
            )

            db_min_date, db_max_date = result.fetchone()

        master_df["Date"] = pd.to_datetime(
            master_df["Date"],
            errors="coerce"
        )

        expected_min_date = master_df["Date"].min().date()
        expected_max_date = master_df["Date"].max().date()

        print(f"Expected range : {expected_min_date} -> {expected_max_date}")
        print(f"Database range : {db_min_date} -> {db_max_date}")

        if (
            str(db_min_date) == str(expected_min_date)
            and str(db_max_date) == str(expected_max_date)
        ):
            print("PASS - Date range matches")
        else:
            print("FAIL - Date range does not match")
            return False

    except Exception as e:
        print("FAIL - Could not verify date range")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 7. Check duplicate Ticker + Date records
    # ---------------------------------------------------------
    print("\n[7] Checking duplicate Ticker + Date records...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT Ticker, Date, COUNT(*) AS record_count
                    FROM stock_data
                    GROUP BY Ticker, Date
                    HAVING COUNT(*) > 1
                    """
                )
            )

            duplicates = result.fetchall()

        if len(duplicates) == 0:
            print("PASS - No duplicate Ticker + Date records")
        else:
            print(
                f"FAIL - Found {len(duplicates)} duplicate "
                "Ticker + Date groups"
            )
            return False

    except Exception as e:
        print("FAIL - Could not check duplicates")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 8. Check NULL values
    # ---------------------------------------------------------
    print("\n[8] Checking NULL values...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT
                        SUM(CASE WHEN Date IS NULL THEN 1 ELSE 0 END),
                        SUM(CASE WHEN Ticker IS NULL THEN 1 ELSE 0 END),
                        SUM(CASE WHEN Close IS NULL THEN 1 ELSE 0 END),
                        SUM(CASE WHEN Target IS NULL THEN 1 ELSE 0 END)
                    FROM stock_data
                    """
                )
            )

            null_counts = result.fetchone()

        date_nulls = null_counts[0]
        ticker_nulls = null_counts[1]
        close_nulls = null_counts[2]
        target_nulls = null_counts[3]

        print(f"Date NULLs   : {date_nulls}")
        print(f"Ticker NULLs : {ticker_nulls}")
        print(f"Close NULLs  : {close_nulls}")
        print(f"Target NULLs : {target_nulls}")

        if (
            date_nulls == 0
            and ticker_nulls == 0
            and close_nulls == 0
            and target_nulls == 0
        ):
            print("PASS - Required fields contain no NULL values")
        else:
            print("FAIL - NULL values detected")
            return False

    except Exception as e:
        print("FAIL - Could not check NULL values")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # 9. Check Target distribution
    # ---------------------------------------------------------
    print("\n[9] Checking Target distribution...")

    try:
        with engine.connect() as connection:
            result = connection.execute(
                text(
                    """
                    SELECT Target, COUNT(*) AS count
                    FROM stock_data
                    GROUP BY Target
                    ORDER BY Target
                    """
                )
            )

            target_rows = result.fetchall()

        if len(target_rows) > 0:
            for target, count in target_rows:
                print(f"{target}: {count}")

            print("PASS - Target values are present")
        else:
            print("FAIL - No Target values found")
            return False

    except Exception as e:
        print("FAIL - Could not check Target distribution")
        print(f"Error: {e}")
        return False

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("DATABASE VERIFICATION COMPLETED")
    print("=" * 60)
    print("OVERALL RESULT: PASS")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = verify_database()

    if not success:
        sys.exit(1)
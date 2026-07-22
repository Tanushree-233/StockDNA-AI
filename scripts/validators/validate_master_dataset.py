import pandas as pd
import numpy as np

MASTER_DATASET = "data/final/master_dataset.csv"

def load_dataset():
    return pd.read_csv(MASTER_DATASET)

def dataset_info(df):

    print("=" * 50)
    print("DATASET INFORMATION")
    print("=" * 50)

    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    print("\nColumn Names:\n")
    print(df.columns.tolist())

def check_missing(df):

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)

def check_missing(df):

    print("\n" + "=" * 50)
    print("MISSING VALUES")
    print("=" * 50)

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values.")
    else:
        print(missing)

def check_duplicates(df):

    print("\n" + "=" * 50)
    print("DUPLICATES")
    print("=" * 50)

    print("Duplicate rows:", df.duplicated().sum())

def check_infinite(df):

    print("\n" + "=" * 50)
    print("INFINITE VALUES")
    print("=" * 50)

    numeric_df = df.select_dtypes(include=np.number)

    infinite = np.isinf(numeric_df).sum().sum()

    print("Infinite values:", infinite)

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

def main():

    df = load_dataset()

    dataset_info(df)

    check_missing(df)

    check_duplicates(df)

    check_infinite(df)

    check_dtypes(df)

    summary_statistics(df)


if __name__ == "__main__":
    main()
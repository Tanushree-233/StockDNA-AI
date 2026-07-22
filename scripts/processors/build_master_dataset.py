import os
import pandas as pd

FEATURES_FOLDER = "data/processed/features"
FINAL_FOLDER = "data/final"

os.makedirs(FINAL_FOLDER, exist_ok=True)

def load_feature_files():

    dataframes = []

    for file in os.listdir(FEATURES_FOLDER):

        if file.endswith(".csv"):

            df = pd.read_csv(
                os.path.join(FEATURES_FOLDER, file)
            )

            ticker = file.replace("_features.csv", "")

            df["Ticker"] = ticker

            dataframes.append(df)

    return dataframes

def merge_all():

    dfs = load_feature_files()

    master = pd.concat(
        dfs,
        ignore_index=True
    )

    master.to_csv(
        "data/final/master_dataset.csv",
        index=False
    )

    print(master.head())

    print("\nRows:", len(master))

if __name__ == "__main__":
    merge_all()
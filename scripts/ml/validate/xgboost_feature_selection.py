import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

X_PATH = "models/X_train.pkl"
Y_PATH = "models/y_train.pkl"

IMPORTANCE_OUTPUT = "models/xgboost_feature_importance.csv"
CV_OUTPUT = "models/xgboost_feature_selection_cv.csv"

N_SPLITS = 5

CLASS_WEIGHTS = {
    0: 3.0,   # SELL
    1: 1.0,   # HOLD
    2: 2.5,   # BUY
}

# Test different numbers of top features
FEATURE_COUNTS = [20, 30, 40, 50, 60, 70, 87]


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("XGBOOST FEATURE SELECTION")
print("=" * 60)

X = joblib.load(X_PATH)
y = joblib.load(Y_PATH)

print("\nTraining data:")
print("X:", X.shape)
print("y:", y.shape)

# Make sure data is in time order
if "Date" in X.columns:
    print("\nWARNING: Date is present in X.")
    print("Date should NOT normally be used as a model feature.")

    X = X.drop(columns=["Date"])


# ============================================================
# TRAIN BASELINE XGBOOST
# ============================================================

print("\n" + "=" * 60)
print("TRAINING BASELINE XGBOOST")
print("=" * 60)

baseline_model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

sample_weights = y.map(CLASS_WEIGHTS).values

baseline_model.fit(
    X,
    y,
    sample_weight=sample_weights,
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": baseline_model.feature_importances_,
})

importance = importance.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)

importance["Rank"] = np.arange(
    1,
    len(importance) + 1
)

importance = importance[
    ["Rank", "Feature", "Importance"]
]

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.to_string(index=False)
)

importance.to_csv(
    IMPORTANCE_OUTPUT,
    index=False
)

print("\nSaved:")
print(IMPORTANCE_OUTPUT)


# ============================================================
# TIME-SERIES CV FOR FEATURE COUNTS
# ============================================================

print("\n" + "=" * 60)
print("FEATURE COUNT ABLATION")
print("=" * 60)

ranked_features = importance["Feature"].tolist()

tscv = TimeSeriesSplit(
    n_splits=N_SPLITS
)

results = []


for feature_count in FEATURE_COUNTS:

    selected_features = ranked_features[
        :feature_count
    ]

    print(
        f"\nTesting top {feature_count} features..."
    )

    X_selected = X[
        selected_features
    ]

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(
        tscv.split(X_selected),
        start=1
    ):

        X_train = X_selected.iloc[
            train_idx
        ]

        X_val = X_selected.iloc[
            val_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_val = y.iloc[
            val_idx
        ]

        model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softmax",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )

        weights = y_train.map(
            CLASS_WEIGHTS
        ).values

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

        predictions = model.predict(
            X_val
        )

        accuracy = accuracy_score(
            y_val,
            predictions
        )

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_val,
            predictions
        )

        fold_scores.append({
            "Fold": fold,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced_accuracy,
        })

    fold_df = pd.DataFrame(
        fold_scores
    )

    results.append({
        "Feature_Count": feature_count,
        "Accuracy_Mean": fold_df["Accuracy"].mean(),
        "Accuracy_STD": fold_df["Accuracy"].std(),
        "Macro_F1_Mean": fold_df["Macro_F1"].mean(),
        "Macro_F1_STD": fold_df["Macro_F1"].std(),
        "Balanced_Accuracy_Mean": (
            fold_df["Balanced_Accuracy"].mean()
        ),
        "Balanced_Accuracy_STD": (
            fold_df["Balanced_Accuracy"].std()
        ),
    })

    print(
        f"Accuracy: "
        f"{fold_df['Accuracy'].mean():.4f}"
    )

    print(
        f"Macro F1: "
        f"{fold_df['Macro_F1'].mean():.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{fold_df['Balanced_Accuracy'].mean():.4f}"
    )


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "Macro_F1_Mean",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("FEATURE SELECTION RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ============================================================
# BEST FEATURE COUNT
# ============================================================

best_feature_count = int(
    results_df.iloc[0]["Feature_Count"]
)

best_macro_f1 = (
    results_df.iloc[0]["Macro_F1_Mean"]
)

best_balanced_accuracy = (
    results_df.iloc[0]["Balanced_Accuracy_Mean"]
)

best_features = ranked_features[
    :best_feature_count
]

print("\n" + "=" * 60)
print("BEST FEATURE SET")
print("=" * 60)

print(
    "Feature count:",
    best_feature_count
)

print(
    "Mean Macro F1:",
    round(best_macro_f1, 4)
)

print(
    "Mean Balanced Accuracy:",
    round(
        best_balanced_accuracy,
        4
    )
)

print("\nSelected features:")

for i, feature in enumerate(
    best_features,
    start=1
):
    print(
        f"{i:02d}. {feature}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    CV_OUTPUT,
    index=False
)

joblib.dump(
    best_features,
    "models/selected_features.pkl"
)

print("\nSaved:")
print(CV_OUTPUT)
print("models/selected_features.pkl")

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE")
print("=" * 60)
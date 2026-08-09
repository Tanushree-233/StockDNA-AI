import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)
from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

X_TRAIN_PATH = "models/X_train.pkl"
Y_TRAIN_PATH = "models/y_train.pkl"

OUTPUT_PATH = "models/time_series_cv_results.csv"

N_SPLITS = 5

# Use the best class-weight configuration found previously.
CLASS_WEIGHTS = {
    0: 3.0,   # SELL
    1: 1.0,   # HOLD
    2: 2.5,   # BUY
}


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("TIME-SERIES CROSS VALIDATION")
print("=" * 60)

X = joblib.load(X_TRAIN_PATH)
y = joblib.load(Y_TRAIN_PATH)

print("\nTraining data:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ============================================================
# TIME-SERIES SPLIT
# ============================================================

tscv = TimeSeriesSplit(n_splits=N_SPLITS)

results = []


# ============================================================
# MODELS
# ============================================================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        class_weight=CLASS_WEIGHTS,
        solver="lbfgs",
        random_state=42,
    ),

    "XGBoost": XGBClassifier(
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
    ),
}


# ============================================================
# CROSS VALIDATION
# ============================================================

for fold, (train_idx, val_idx) in enumerate(
    tscv.split(X),
    start=1
):

    print("\n" + "-" * 60)
    print(f"FOLD {fold}")
    print("-" * 60)

    X_fold_train = X.iloc[train_idx]
    X_fold_val = X.iloc[val_idx]

    y_fold_train = y.iloc[train_idx]
    y_fold_val = y.iloc[val_idx]

    print(
        f"Train: {X_fold_train.shape} | "
        f"Validation: {X_fold_val.shape}"
    )

    print(
        f"Train indices: {train_idx[0]} → {train_idx[-1]}"
    )

    print(
        f"Validation indices: {val_idx[0]} → {val_idx[-1]}"
    )

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # XGBoost does not use sklearn's class_weight parameter.
        # Give it equivalent sample weights.
        if model_name == "XGBoost":

            sample_weights = y_fold_train.map(
                CLASS_WEIGHTS
            ).values

            model.fit(
                X_fold_train,
                y_fold_train,
                sample_weight=sample_weights,
            )

        else:

            model.fit(
                X_fold_train,
                y_fold_train,
            )

        predictions = model.predict(X_fold_val)

        accuracy = accuracy_score(
            y_fold_val,
            predictions
        )

        macro_f1 = f1_score(
            y_fold_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced_accuracy = balanced_accuracy_score(
            y_fold_val,
            predictions
        )

        print(
            f"{model_name}: "
            f"Accuracy={accuracy:.4f}, "
            f"Macro F1={macro_f1:.4f}, "
            f"Balanced Accuracy={balanced_accuracy:.4f}"
        )

        results.append({
            "Fold": fold,
            "Model": model_name,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced_accuracy,
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)


# ============================================================
# AVERAGE RESULTS
# ============================================================

summary = (
    results_df
    .groupby("Model")[
        [
            "Accuracy",
            "Macro_F1",
            "Balanced_Accuracy",
        ]
    ]
    .agg(["mean", "std"])
)

print("\n" + "=" * 60)
print("AVERAGE CV PERFORMANCE")
print("=" * 60)

print(summary)


# ============================================================
# BEST MODEL
# ============================================================

mean_results = (
    results_df
    .groupby("Model")[
        [
            "Accuracy",
            "Macro_F1",
            "Balanced_Accuracy",
        ]
    ]
    .mean()
    .sort_values(
        "Macro_F1",
        ascending=False
    )
)

print("\n" + "=" * 60)
print("MODEL RANKING")
print("=" * 60)

print(mean_results)


best_model = mean_results.index[0]

print("\nSelected by average Macro F1:")
print(best_model)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("TIME-SERIES CV COMPLETE")
print("=" * 60)
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)

from xgboost import XGBClassifier


MODEL_DIR = "models"

X_train = joblib.load(f"{MODEL_DIR}/X_train.pkl")
X_test = joblib.load(f"{MODEL_DIR}/X_test.pkl")
y_train = joblib.load(f"{MODEL_DIR}/y_train.pkl")
y_test = joblib.load(f"{MODEL_DIR}/y_test.pkl")

print("=" * 60)
print("MODEL OPTIMIZATION")
print("=" * 60)

print("Train:", X_train.shape)
print("Test :", X_test.shape)


# =========================================================
# CLASS-WEIGHT EXPERIMENTS
# =========================================================

weight_sets = [
    {0: 1.5, 1: 1.0, 2: 1.5},
    {0: 2.0, 1: 1.0, 2: 2.0},
    {0: 2.5, 1: 1.0, 2: 2.0},
    {0: 3.0, 1: 1.0, 2: 2.5},
]


results = []


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

for weights in weight_sets:

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION")
    print("Weights:", weights)
    print("=" * 60)

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight=weights,
                C=0.5,
                solver="lbfgs",
                random_state=42,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )
    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced:.4f}")

    print(
        classification_report(
            y_test,
            pred,
            target_names=["SELL", "HOLD", "BUY"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    results.append({
        "Model": "LogisticRegression",
        "Weights": str(weights),
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Balanced_Accuracy": balanced,
    })


# =========================================================
# XGBOOST
# =========================================================

print("\n" + "=" * 60)
print("XGBOOST OPTIMIZATION")
print("=" * 60)

for weights in weight_sets:

    print("\nWeights:", weights)

    sample_weights = y_train.map(weights).values

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        gamma=0.1,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
    )

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)
    macro_f1 = f1_score(
        y_test,
        pred,
        average="macro"
    )
    balanced = balanced_accuracy_score(
        y_test,
        pred
    )

    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Balanced Accuracy: {balanced:.4f}")

    print(
        classification_report(
            y_test,
            pred,
            target_names=["SELL", "HOLD", "BUY"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    results.append({
        "Model": "XGBoost",
        "Weights": str(weights),
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Balanced_Accuracy": balanced,
    })


# =========================================================
# COMPARISON
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["Macro_F1", "Balanced_Accuracy"],
    ascending=False
)

print("\n" + "=" * 60)
print("OPTIMIZATION RESULTS")
print("=" * 60)

print(
    results_df.to_string(index=False)
)

results_df.to_csv(
    f"{MODEL_DIR}/optimization_results.csv",
    index=False
)

print("\nSaved:")
print("models/optimization_results.csv")

print("\nOPTIMIZATION COMPLETE.")
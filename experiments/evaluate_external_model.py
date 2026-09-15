import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    balanced_accuracy_score
)


# ==================================================
# LOAD
# ==================================================

model = joblib.load(
    "models/external_xgboost.pkl"
)

X_test = joblib.load(
    "models/X_external_test.pkl"
)

y_test = joblib.load(
    "models/y_external_test.pkl"
)


# ==================================================
# PREDICT
# ==================================================

predictions = model.predict(X_test)


# ==================================================
# METRICS
# ==================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

balanced_acc = balanced_accuracy_score(
    y_test,
    predictions
)


# ==================================================
# RESULTS
# ==================================================

print("=" * 60)
print("EXTERNAL MODEL EVALUATION")
print("=" * 60)

print(f"\nAccuracy:          {accuracy * 100:.2f}%")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")


print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "SELL",
            "HOLD",
            "BUY"
        ],
        digits=4
    )
)


print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


# ==================================================
# PREDICTION DISTRIBUTION
# ==================================================

print("\n" + "=" * 60)
print("PREDICTION DISTRIBUTION")
print("=" * 60)

unique, counts = np.unique(
    predictions,
    return_counts=True
)

for label, count in zip(unique, counts):

    name = {
        0: "SELL",
        1: "HOLD",
        2: "BUY"
    }[label]

    percentage = (
        count / len(predictions)
    ) * 100

    print(
        f"{name}: {count} ({percentage:.2f}%)"
    )
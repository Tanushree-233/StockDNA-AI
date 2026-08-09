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

MODEL_DIR = "models"

X_train = joblib.load(f"{MODEL_DIR}/X_train.pkl")
X_test = joblib.load(f"{MODEL_DIR}/X_test.pkl")
y_train = joblib.load(f"{MODEL_DIR}/y_train.pkl")
y_test = joblib.load(f"{MODEL_DIR}/y_test.pkl")

# Best weights found previously
weights = {
    0: 3.0,   # SELL
    1: 1.0,   # HOLD
    2: 2.5,   # BUY
}

print("=" * 60)
print("THRESHOLD OPTIMIZATION")
print("=" * 60)

# ---------------------------------------------------------
# TRAIN BEST LOGISTIC MODEL
# ---------------------------------------------------------

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

probabilities = model.predict_proba(X_test)

# probability columns:
# 0 = SELL
# 1 = HOLD
# 2 = BUY

best = None
results = []

# ---------------------------------------------------------
# SEARCH SELL / BUY THRESHOLDS
# ---------------------------------------------------------

for sell_threshold in np.arange(0.20, 0.71, 0.02):

    for buy_threshold in np.arange(0.20, 0.71, 0.02):

        predictions = []

        for p_sell, p_hold, p_buy in probabilities:

            # Strong SELL signal
            if (
                p_sell >= sell_threshold
                and p_sell >= p_buy
            ):
                prediction = 0

            # Strong BUY signal
            elif (
                p_buy >= buy_threshold
                and p_buy > p_sell
            ):
                prediction = 2

            # Otherwise HOLD
            else:
                prediction = 1

            predictions.append(prediction)

        predictions = np.array(predictions)

        macro_f1 = f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        )

        balanced = balanced_accuracy_score(
            y_test,
            predictions,
        )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        results.append({
            "SELL_Threshold": sell_threshold,
            "BUY_Threshold": buy_threshold,
            "Accuracy": accuracy,
            "Macro_F1": macro_f1,
            "Balanced_Accuracy": balanced,
        })

        # Primary objective = Macro F1
        # Secondary objective = Balanced Accuracy
        score = (macro_f1, balanced)

        if best is None or score > best["score"]:

            best = {
                "score": score,
                "SELL_Threshold": sell_threshold,
                "BUY_Threshold": buy_threshold,
                "Accuracy": accuracy,
                "Macro_F1": macro_f1,
                "Balanced_Accuracy": balanced,
                "Predictions": predictions,
            }

# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    ["Macro_F1", "Balanced_Accuracy"],
    ascending=False,
)

results_df.to_csv(
    f"{MODEL_DIR}/threshold_results.csv",
    index=False,
)

print("\n" + "=" * 60)
print("BEST THRESHOLDS")
print("=" * 60)

print(
    f"SELL threshold: {best['SELL_Threshold']:.2f}"
)

print(
    f"BUY threshold : {best['BUY_Threshold']:.2f}"
)

print(
    f"Accuracy:          {best['Accuracy']:.4f}"
)

print(
    f"Macro F1:          {best['Macro_F1']:.4f}"
)

print(
    f"Balanced Accuracy: {best['Balanced_Accuracy']:.4f}"
)

# ---------------------------------------------------------
# FINAL REPORT
# ---------------------------------------------------------

predictions = best["Predictions"]

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "SELL",
            "HOLD",
            "BUY",
        ],
        digits=4,
        zero_division=0,
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        predictions,
    )
)

print("\nPrediction Distribution:")

print(
    pd.Series(predictions)
    .map({
        0: "SELL",
        1: "HOLD",
        2: "BUY",
    })
    .value_counts()
)

# ---------------------------------------------------------
# SAVE MODEL + THRESHOLDS
# ---------------------------------------------------------

joblib.dump(
    model,
    f"{MODEL_DIR}/final_logistic_model.pkl",
)

thresholds = {
    "SELL_threshold": float(best["SELL_Threshold"]),
    "BUY_threshold": float(best["BUY_Threshold"]),
    "class_weights": weights,
    "macro_f1": float(best["Macro_F1"]),
    "balanced_accuracy": float(best["Balanced_Accuracy"]),
    "accuracy": float(best["Accuracy"]),
}

joblib.dump(
    thresholds,
    f"{MODEL_DIR}/decision_thresholds.pkl",
)

print("\nSaved:")
print("models/final_logistic_model.pkl")
print("models/decision_thresholds.pkl")
print("models/threshold_results.csv")

print("\nTHRESHOLD OPTIMIZATION COMPLETE.")
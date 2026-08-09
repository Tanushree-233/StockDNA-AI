import os
import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

X_TRAIN_PATH = "models/X_train.pkl"
X_TEST_PATH = "models/X_test.pkl"

Y_TRAIN_PATH = "models/y_train.pkl"
Y_TEST_PATH = "models/y_test.pkl"

MODEL_OUTPUT = "models/final_xgboost_model.pkl"
RESULTS_OUTPUT = "models/final_xgboost_results.csv"


CLASS_WEIGHTS = {
    0: 3.0,   # SELL
    1: 1.0,   # HOLD
    2: 2.5,   # BUY
}


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 60)
print("FINAL XGBOOST MODEL EVALUATION")
print("=" * 60)

X_train = joblib.load(X_TRAIN_PATH)
X_test = joblib.load(X_TEST_PATH)

y_train = joblib.load(Y_TRAIN_PATH)
y_test = joblib.load(Y_TEST_PATH)

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting:")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)


# ============================================================
# SAFETY CHECKS
# ============================================================

assert list(X_train.columns) == list(X_test.columns), (
    "Train/test feature columns do not match."
)

assert X_train.isna().sum().sum() == 0, (
    "NaN values found in X_train."
)

assert X_test.isna().sum().sum() == 0, (
    "NaN values found in X_test."
)

assert X_train.isin([float("inf"), float("-inf")]).sum().sum() == 0
assert X_test.isin([float("inf"), float("-inf")]).sum().sum() == 0


# ============================================================
# MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FINAL XGBOOST")
print("=" * 60)

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

sample_weights = y_train.map(
    CLASS_WEIGHTS
).values

model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights,
)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_OUTPUT),
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_OUTPUT
)

print("\nModel saved:")
print(MODEL_OUTPUT)


# ============================================================
# FINAL TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

predictions = model.predict(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0,
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    predictions
)


print(f"\nAccuracy:          {accuracy:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2],
        target_names=[
            "SELL",
            "HOLD",
            "BUY",
        ],
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2],
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_distribution = pd.Series(
    predictions
).map({
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}).value_counts()

print("\nPrediction Distribution:")
print(prediction_distribution)


# ============================================================
# ACTUAL DISTRIBUTION
# ============================================================

actual_distribution = pd.Series(
    y_test
).map({
    0: "SELL",
    1: "HOLD",
    2: "BUY",
}).value_counts()

print("\nActual Test Distribution:")
print(actual_distribution)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame([
    {
        "Model": "Final XGBoost",
        "Accuracy": accuracy,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Balanced_Accuracy": balanced_accuracy,
    }
])

results.to_csv(
    RESULTS_OUTPUT,
    index=False
)

print("\nResults saved:")
print(RESULTS_OUTPUT)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("FINAL XGBOOST EVALUATION COMPLETE")
print("=" * 60)
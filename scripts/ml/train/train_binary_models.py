import os
import joblib

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = "models"

# Internal data
INTERNAL_X_TRAIN = f"{MODEL_DIR}/internal_X_train.pkl"
INTERNAL_X_TEST = f"{MODEL_DIR}/internal_X_test.pkl"

# External data
EXTERNAL_X_TRAIN = f"{MODEL_DIR}/external_X_train.pkl"
EXTERNAL_X_TEST = f"{MODEL_DIR}/external_X_test.pkl"

# Targets
Y_TRAIN = f"{MODEL_DIR}/internal_y_train.pkl"
Y_TEST = f"{MODEL_DIR}/internal_y_test.pkl"

# Output models
INTERNAL_MODEL_OUTPUT = f"{MODEL_DIR}/binary_internal_xgboost.pkl"
EXTERNAL_MODEL_OUTPUT = f"{MODEL_DIR}/binary_external_xgboost.pkl"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("TRAINING BINARY BUY / SELL MODELS")
print("=" * 60)

X_internal_train = joblib.load(INTERNAL_X_TRAIN)
X_internal_test = joblib.load(INTERNAL_X_TEST)

X_external_train = joblib.load(EXTERNAL_X_TRAIN)
X_external_test = joblib.load(EXTERNAL_X_TEST)

y_train = joblib.load(Y_TRAIN)
y_test = joblib.load(Y_TEST)

print("\nInternal:")
print("Train:", X_internal_train.shape)
print("Test :", X_internal_test.shape)

print("\nExternal:")
print("Train:", X_external_train.shape)
print("Test :", X_external_test.shape)

print("\nTargets:")
print("Train:", y_train.shape)
print("Test :", y_test.shape)


# ============================================================
# INTERNAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING INTERNAL MODEL")
print("=" * 60)

internal_model = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

internal_model.fit(
    X_internal_train,
    y_train
)

print("Internal model training complete.")


# ============================================================
# INTERNAL EVALUATION
# ============================================================

internal_predictions = internal_model.predict(
    X_internal_test
)

internal_accuracy = accuracy_score(
    y_test,
    internal_predictions
)

internal_f1 = f1_score(
    y_test,
    internal_predictions,
    average="macro",
    zero_division=0
)

internal_balanced_accuracy = balanced_accuracy_score(
    y_test,
    internal_predictions
)

print("\nInternal Model Results:")
print(
    f"Accuracy:           {internal_accuracy:.4f}"
)

print(
    f"Macro F1:           {internal_f1:.4f}"
)

print(
    f"Balanced Accuracy:  {internal_balanced_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        internal_predictions,
        target_names=["SELL", "BUY"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        internal_predictions
    )
)


# ============================================================
# SAVE INTERNAL MODEL
# ============================================================

joblib.dump(
    internal_model,
    INTERNAL_MODEL_OUTPUT
)

print("\nInternal model saved:")
print(INTERNAL_MODEL_OUTPUT)


# ============================================================
# EXTERNAL MODEL
# ============================================================

print("\n" + "=" * 60)
print("TRAINING EXTERNAL MODEL")
print("=" * 60)

external_model = XGBClassifier(
    n_estimators=300,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

external_model.fit(
    X_external_train,
    y_train
)

print("External model training complete.")


# ============================================================
# EXTERNAL EVALUATION
# ============================================================

external_predictions = external_model.predict(
    X_external_test
)

external_accuracy = accuracy_score(
    y_test,
    external_predictions
)

external_f1 = f1_score(
    y_test,
    external_predictions,
    average="macro",
    zero_division=0
)

external_balanced_accuracy = balanced_accuracy_score(
    y_test,
    external_predictions
)

print("\nExternal Model Results:")
print(
    f"Accuracy:           {external_accuracy:.4f}"
)

print(
    f"Macro F1:           {external_f1:.4f}"
)

print(
    f"Balanced Accuracy:  {external_balanced_accuracy:.4f}"
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        external_predictions,
        target_names=["SELL", "BUY"],
        zero_division=0
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        external_predictions
    )
)


# ============================================================
# SAVE EXTERNAL MODEL
# ============================================================

joblib.dump(
    external_model,
    EXTERNAL_MODEL_OUTPUT
)

print("\nExternal model saved:")
print(EXTERNAL_MODEL_OUTPUT)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("BINARY MODEL TRAINING COMPLETE")
print("=" * 60)

print("\nModels created:")
print("1.", INTERNAL_MODEL_OUTPUT)
print("2.", EXTERNAL_MODEL_OUTPUT)

print("\nTarget:")
print("0 = SELL")
print("1 = BUY")

print("\nREADY FOR COMBINED PREDICTION.")
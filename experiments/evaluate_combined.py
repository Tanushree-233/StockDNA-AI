import os
import joblib
import numpy as np

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

INTERNAL_MODEL = f"{MODEL_DIR}/binary_internal_xgboost.pkl"
EXTERNAL_MODEL = f"{MODEL_DIR}/binary_external_xgboost.pkl"

INTERNAL_X_TEST = f"{MODEL_DIR}/internal_X_test.pkl"
EXTERNAL_X_TEST = f"{MODEL_DIR}/external_X_test.pkl"

Y_TEST = f"{MODEL_DIR}/internal_y_test.pkl"


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("COMBINED INTERNAL / EXTERNAL MODEL EVALUATION")
print("=" * 70)

print("\nLoading models...")

internal_model = joblib.load(INTERNAL_MODEL)
external_model = joblib.load(EXTERNAL_MODEL)

print("Internal model loaded.")
print("External model loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test data...")

internal_X_test = joblib.load(INTERNAL_X_TEST)
external_X_test = joblib.load(EXTERNAL_X_TEST)

y_test = joblib.load(Y_TEST)

print("Internal test shape:", internal_X_test.shape)
print("External test shape:", external_X_test.shape)
print("Target shape:", y_test.shape)


# ============================================================
# INDIVIDUAL MODEL PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("INDIVIDUAL MODEL PREDICTIONS")
print("=" * 70)


# Internal probabilities
internal_proba = internal_model.predict_proba(
    internal_X_test
)[:, 1]

internal_pred = (
    internal_proba >= 0.50
).astype(int)


# External probabilities
external_proba = external_model.predict_proba(
    external_X_test
)[:, 1]

external_pred = (
    external_proba >= 0.50
).astype(int)


# ============================================================
# INTERNAL MODEL EVALUATION
# ============================================================

internal_accuracy = accuracy_score(
    y_test,
    internal_pred
)

internal_f1 = f1_score(
    y_test,
    internal_pred,
    average="macro"
)

internal_balanced = balanced_accuracy_score(
    y_test,
    internal_pred
)


print("\nINTERNAL MODEL")
print("-" * 40)

print(
    f"Accuracy            : {internal_accuracy:.4f}"
)

print(
    f"Macro F1            : {internal_f1:.4f}"
)

print(
    f"Balanced Accuracy   : {internal_balanced:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        internal_pred,
        target_names=["SELL", "BUY"]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        internal_pred
    )
)


# ============================================================
# EXTERNAL MODEL EVALUATION
# ============================================================

external_accuracy = accuracy_score(
    y_test,
    external_pred
)

external_f1 = f1_score(
    y_test,
    external_pred,
    average="macro"
)

external_balanced = balanced_accuracy_score(
    y_test,
    external_pred
)


print("\nEXTERNAL MODEL")
print("-" * 40)

print(
    f"Accuracy            : {external_accuracy:.4f}"
)

print(
    f"Macro F1            : {external_f1:.4f}"
)

print(
    f"Balanced Accuracy   : {external_balanced:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        external_pred,
        target_names=["SELL", "BUY"]
    )
)

print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        external_pred
    )
)


# ============================================================
# COMBINED MODEL
# ============================================================

print("\n" + "=" * 70)
print("COMBINED MODEL")
print("=" * 70)


# Equal contribution
INTERNAL_WEIGHT = 0.50
EXTERNAL_WEIGHT = 0.50


combined_proba = (
    INTERNAL_WEIGHT * internal_proba
    +
    EXTERNAL_WEIGHT * external_proba
)


combined_pred = (
    combined_proba >= 0.50
).astype(int)


# ============================================================
# COMBINED EVALUATION
# ============================================================

combined_accuracy = accuracy_score(
    y_test,
    combined_pred
)

combined_f1 = f1_score(
    y_test,
    combined_pred,
    average="macro"
)

combined_balanced = balanced_accuracy_score(
    y_test,
    combined_pred
)


print("\nWeights:")
print(
    f"Internal Weight : {INTERNAL_WEIGHT:.0%}"
)

print(
    f"External Weight : {EXTERNAL_WEIGHT:.0%}"
)


print("\nCOMBINED RESULTS")
print("-" * 40)

print(
    f"Accuracy            : {combined_accuracy:.4f}"
)

print(
    f"Macro F1            : {combined_f1:.4f}"
)

print(
    f"Balanced Accuracy   : {combined_balanced:.4f}"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        combined_pred,
        target_names=["SELL", "BUY"]
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        combined_pred
    )
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


print(
    f"{'Model':<15}"
    f"{'Accuracy':>12}"
    f"{'Macro F1':>12}"
    f"{'Balanced Acc':>15}"
)

print("-" * 55)

print(
    f"{'Internal':<15}"
    f"{internal_accuracy:>12.4f}"
    f"{internal_f1:>12.4f}"
    f"{internal_balanced:>15.4f}"
)

print(
    f"{'External':<15}"
    f"{external_accuracy:>12.4f}"
    f"{external_f1:>12.4f}"
    f"{external_balanced:>15.4f}"
)

print(
    f"{'Combined':<15}"
    f"{combined_accuracy:>12.4f}"
    f"{combined_f1:>12.4f}"
    f"{combined_balanced:>15.4f}"
)


# ============================================================
# IMPROVEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("COMBINATION IMPACT")
print("=" * 70)


accuracy_change = (
    combined_accuracy
    -
    max(
        internal_accuracy,
        external_accuracy
    )
)

f1_change = (
    combined_f1
    -
    max(
        internal_f1,
        external_f1
    )
)

balanced_change = (
    combined_balanced
    -
    max(
        internal_balanced,
        external_balanced
    )
)


print(
    f"Accuracy change vs best individual model : "
    f"{accuracy_change:+.4f}"
)

print(
    f"Macro F1 change vs best individual model : "
    f"{f1_change:+.4f}"
)

print(
    f"Balanced accuracy change                : "
    f"{balanced_change:+.4f}"
)


if combined_accuracy > max(
    internal_accuracy,
    external_accuracy
):

    print(
        "\nRESULT: Combined model improves accuracy."
    )

elif combined_accuracy == max(
    internal_accuracy,
    external_accuracy
):

    print(
        "\nRESULT: Combined model matches the best "
        "individual model."
    )

else:

    print(
        "\nRESULT: Combined model does NOT improve "
        "accuracy yet."
    )


# ============================================================
# FACTOR AGREEMENT ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("INTERNAL vs EXTERNAL FACTOR AGREEMENT")
print("=" * 70)


internal_buy = internal_pred == 1
external_buy = external_pred == 1

both_buy = np.sum(
    internal_buy & external_buy
)

both_sell = np.sum(
    (~internal_buy) & (~external_buy)
)

disagree = np.sum(
    internal_pred != external_pred
)

total = len(y_test)


print(
    f"Both BUY       : {both_buy}"
)

print(
    f"Both SELL      : {both_sell}"
)

print(
    f"Disagree       : {disagree}"
)

print(
    f"Total samples  : {total}"
)

print(
    f"Agreement rate : "
    f"{(both_buy + both_sell) / total:.2%}"
)

print(
    f"Disagreement rate : "
    f"{disagree / total:.2%}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "internal_accuracy": internal_accuracy,
    "internal_macro_f1": internal_f1,
    "internal_balanced_accuracy": internal_balanced,

    "external_accuracy": external_accuracy,
    "external_macro_f1": external_f1,
    "external_balanced_accuracy": external_balanced,

    "combined_accuracy": combined_accuracy,
    "combined_macro_f1": combined_f1,
    "combined_balanced_accuracy": combined_balanced,

    "internal_weight": INTERNAL_WEIGHT,
    "external_weight": EXTERNAL_WEIGHT,

    "both_buy": int(both_buy),
    "both_sell": int(both_sell),
    "disagreement": int(disagree),

    "agreement_rate": (
        (both_buy + both_sell) / total
    ),

    "disagreement_rate": (
        disagree / total
    )
}


output_file = (
    f"{MODEL_DIR}/combined_evaluation_results.pkl"
)

joblib.dump(
    results,
    output_file
)


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

print("\nResults saved:")
print(output_file)

print("\nREADY FOR MODEL IMPROVEMENT.")
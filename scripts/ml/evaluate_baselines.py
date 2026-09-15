"""
Scientific Baseline Evaluator for StockDNA-AI.

Evaluates 6 baseline models across Validation and Test sets:
1. Always HOLD (Zero-rule)
2. Majority Class Predictor
3. Stratified Random Classifier
4. Balanced Logistic Regression
5. Balanced Random Forest Classifier
6. Production XGBoost Classifier (multi:softprob)

Metrics computed:
- Accuracy, Balanced Accuracy
- Macro Precision, Macro Recall, Macro F1
- Weighted F1
- Per-Class Precision, Recall, F1 for SELL (0), HOLD (1), BUY (2)
- Confusion Matrix

Outputs saved to `results/model_comparison.csv` and `results/confusion_matrix.csv`.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.feature_registry import get_feature_names
from scripts.ml.target_design import LABEL_MAP
from scripts.ml.train_production_model import (
    load_and_split_data,
    compute_training_imputation,
    apply_imputation,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str, split_name: str) -> dict:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    per_class = {}
    for c_idx, c_name in LABEL_MAP.items():
        yt = (y_true == c_idx).astype(int)
        yp = (y_pred == c_idx).astype(int)
        per_class[f"{c_name}_Precision"] = precision_score(yt, yp, zero_division=0)
        per_class[f"{c_name}_Recall"] = recall_score(yt, yp, zero_division=0)
        per_class[f"{c_name}_F1"] = f1_score(yt, yp, zero_division=0)
        per_class[f"{c_name}_Support"] = int(yt.sum())

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    row = {
        "Model": model_name,
        "Split": split_name,
        "Accuracy": round(acc, 4),
        "Balanced_Accuracy": round(bal_acc, 4),
        "Macro_Precision": round(macro_prec, 4),
        "Macro_Recall": round(macro_rec, 4),
        "Macro_F1": round(macro_f1, 4),
        "Weighted_F1": round(weighted_f1, 4),
        "SELL_Precision": round(per_class["SELL_Precision"], 4),
        "SELL_Recall": round(per_class["SELL_Recall"], 4),
        "SELL_F1": round(per_class["SELL_F1"], 4),
        "HOLD_Precision": round(per_class["HOLD_Precision"], 4),
        "HOLD_Recall": round(per_class["HOLD_Recall"], 4),
        "HOLD_F1": round(per_class["HOLD_F1"], 4),
        "BUY_Precision": round(per_class["BUY_Precision"], 4),
        "BUY_Recall": round(per_class["BUY_Recall"], 4),
        "BUY_F1": round(per_class["BUY_F1"], 4),
        "Confusion_Matrix": json.dumps(cm.tolist()),
    }
    return row


def evaluate_all_baselines():
    print("=" * 70)
    print("EVALUATING SCIENTIFIC BASELINES (VALIDATION & TEST SETS)")
    print("=" * 70)

    df = pd.read_csv("data/processed/production_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    # Strict zero-overlap split dates
    train_df = df[(df["Date"] >= "2018-10-23") & (df["Date"] <= "2023-12-21")].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    val_df = df[(df["Date"] >= "2024-01-02") & (df["Date"] <= "2024-12-23")].sort_values(["Date", "Ticker"]).reset_index(drop=True)
    test_df = df[(df["Date"] >= "2025-01-02") & (df["Date"] <= "2025-12-19")].sort_values(["Date", "Ticker"]).reset_index(drop=True)

    feature_names = get_feature_names()
    imputation_dict = compute_training_imputation(train_df, feature_names)

    X_train = apply_imputation(train_df, feature_names, imputation_dict)
    y_train = train_df["Target"].values.astype(int)

    X_val = apply_imputation(val_df, feature_names, imputation_dict)
    y_val = val_df["Target"].values.astype(int)

    X_test = apply_imputation(test_df, feature_names, imputation_dict)
    y_test = test_df["Target"].values.astype(int)

    sample_weights_train = compute_sample_weight("balanced", y_train)

    models = {}

    # 1. Always HOLD
    dummy_hold = DummyClassifier(strategy="constant", constant=1)
    dummy_hold.fit(X_train, y_train)
    models["Always_HOLD"] = dummy_hold

    # 2. Majority Class
    dummy_majority = DummyClassifier(strategy="most_frequent")
    dummy_majority.fit(X_train, y_train)
    models["Majority_Class"] = dummy_majority

    # 3. Stratified Random
    dummy_stratified = DummyClassifier(strategy="stratified", random_state=42)
    dummy_stratified.fit(X_train, y_train)
    models["Stratified_Random"] = dummy_stratified

    # 4. Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # 5. Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    # 6. XGBoost Classifier
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric="mlogloss",
    )
    xgb.fit(X_train, y_train, sample_weight=sample_weights_train, eval_set=[(X_val, y_val)], verbose=False)
    models["XGBoost"] = xgb

    results = []
    for name, model in models.items():
        # Validation
        val_pred = model.predict(X_val)
        val_row = compute_metrics(y_val, val_pred, name, "Validation")
        results.append(val_row)

        # Test
        test_pred = model.predict(X_test)
        test_row = compute_metrics(y_test, test_pred, name, "Test")
        results.append(test_row)

    res_df = pd.DataFrame(results)

    # Save to results/
    os.makedirs("results", exist_ok=True)
    res_df.to_csv("results/model_comparison.csv", index=False)
    print(f"\nSaved baseline comparison to results/model_comparison.csv")

    # Format and save confusion matrices
    cm_records = []
    for r in results:
        cm_matrix = json.loads(r["Confusion_Matrix"])
        cm_records.append({
            "Model": r["Model"],
            "Split": r["Split"],
            "SELL_True_SELL_Pred": cm_matrix[0][0],
            "SELL_True_HOLD_Pred": cm_matrix[0][1],
            "SELL_True_BUY_Pred": cm_matrix[0][2],
            "HOLD_True_SELL_Pred": cm_matrix[1][0],
            "HOLD_True_HOLD_Pred": cm_matrix[1][1],
            "HOLD_True_BUY_Pred": cm_matrix[1][2],
            "BUY_True_SELL_Pred": cm_matrix[2][0],
            "BUY_True_HOLD_Pred": cm_matrix[2][1],
            "BUY_True_BUY_Pred": cm_matrix[2][2],
        })
    cm_df = pd.DataFrame(cm_records)
    cm_df.to_csv("results/confusion_matrix.csv", index=False)
    print(f"Saved confusion matrices to results/confusion_matrix.csv")

    # Print summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<20} | {'Split':<10} | {'Accuracy':<8} | {'Bal Acc':<8} | {'Macro F1':<8} | {'Weighted F1':<10}")
    print("-" * 90)
    for _, row in res_df.iterrows():
        print(f"{row['Model']:<20} | {row['Split']:<10} | {row['Accuracy']:<8.4f} | {row['Balanced_Accuracy']:<8.4f} | {row['Macro_F1']:<8.4f} | {row['Weighted_F1']:<10.4f}")

    return res_df


if __name__ == "__main__":
    evaluate_all_baselines()

import os
import joblib
from xgboost import XGBClassifier
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ============================================================
# CONFIG
# ============================================================

X_TRAIN_PATH = "models/X_train.pkl"
X_TEST_PATH = "models/X_test.pkl"
Y_TRAIN_PATH = "models/y_train.pkl"
Y_TEST_PATH = "models/y_test.pkl"

MODEL_PATH = "models/final_xgboost_model.pkl"

# ============================================================
# LOAD PREPARED DATA
# ============================================================

X_train = joblib.load(X_TRAIN_PATH)
X_test = joblib.load(X_TEST_PATH)

y_train = joblib.load(Y_TRAIN_PATH)
y_test = joblib.load(Y_TEST_PATH)

print("=" * 60)
print("FINAL XGBOOST TRAINING")
print("=" * 60)

print("\nTraining:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting:")
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# ============================================================
# CREATE MODEL
# ============================================================

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,

    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,

    subsample=0.8,
    colsample_bytree=0.8,

    random_state=42,
    eval_metric="mlogloss"
)

# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 60)
print("TRAINING MODEL")
print("=" * 60)

model.fit(
    X_train,
    y_train
)

print("\nTraining Complete!")

# ============================================================
# QUICK SANITY CHECK
# ============================================================

train_accuracy = model.score(
    X_train,
    y_train
)

test_accuracy = model.score(
    X_test,
    y_test
)

print("\nTraining Accuracy:", round(train_accuracy, 4))
print("Test Accuracy    :", round(test_accuracy, 4))

# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)

print("\nModel Saved:")
print(MODEL_PATH)

print("\n" + "=" * 60)
print("FINAL XGBOOST TRAINING COMPLETE")
print("=" * 60)
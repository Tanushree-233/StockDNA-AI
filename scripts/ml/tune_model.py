import joblib
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
import numpy as np

# -----------------------
# Load Data
# -----------------------

X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")
# -----------------------
# Class Weights
# -----------------------

class_counts = y_train.value_counts()

total = len(y_train)
num_classes = len(class_counts)

class_weights = {
    cls: total / (num_classes * count)
    for cls, count in class_counts.items()
}

print("\nClass Weights:")
print(class_weights)

sample_weights = np.array([
    class_weights[label]
    for label in y_train
])
# -----------------------
# Model
# -----------------------

model = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)

# -----------------------
# Hyperparameters
# -----------------------

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0]
}

# -----------------------
# Time Series Cross Validation
# -----------------------

tscv = TimeSeriesSplit(n_splits=5)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=params,
    n_iter=10,
    cv=tscv,
    scoring="f1_macro",
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("=" * 50)
print("Searching Best Parameters...")
print("=" * 50)

search.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
)

print("\nBest Parameters:\n")
print(search.best_params_)

print("\nBest Macro F1 Score:")
print(search.best_score_)

joblib.dump(search.best_estimator_, "models/best_xgboost.pkl")

print("\nBest Model Saved!")
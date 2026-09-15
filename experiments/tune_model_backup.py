import joblib
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

# -----------------------
# Load Data
# -----------------------

X_train = joblib.load("models/X_train.pkl")
y_train = joblib.load("models/y_train.pkl")

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
    n_iter=20,
    cv=tscv,
    scoring="accuracy",
    verbose=2,
    n_jobs=-1,
    random_state=42
)

print("=" * 50)
print("Searching Best Parameters...")
print("=" * 50)

search.fit(X_train, y_train)

print("\nBest Parameters:\n")
print(search.best_params_)

print("\nBest Score:")
print(search.best_score_)

joblib.dump(search.best_estimator_, "models/best_xgboost.pkl")

print("\nBest Model Saved!")
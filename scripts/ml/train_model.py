import os
import joblib
from xgboost import XGBClassifier

# -----------------------
# Load Prepared Data
# -----------------------

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

# -----------------------
# Create Model
# -----------------------

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

# -----------------------
# Train
# -----------------------

print("=" * 50)
print("Training Model...")
print("=" * 50)

model.fit(X_train, y_train)

print("\nTraining Complete!")

# -----------------------
# Save Model
# -----------------------

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/xgboost_model.pkl")

print("\nModel Saved!")
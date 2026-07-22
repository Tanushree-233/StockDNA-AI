import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------
# Load
# -----------------------

model = joblib.load("models/best_xgboost.pkl")

X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# -----------------------
# Predict
# -----------------------

predictions = model.predict(X_test)

# -----------------------
# Accuracy
# -----------------------

accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("Accuracy")
print("=" * 50)

print(f"{accuracy*100:.2f}%")

print("\n")

print("=" * 50)
print("Classification Report")
print("=" * 50)

print(classification_report(y_test, predictions))

print("\n")

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(confusion_matrix(y_test, predictions))
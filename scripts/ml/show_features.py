import joblib

X_train = joblib.load("models/X_train.pkl")

print("=" * 50)
print("Number of features:", len(X_train.columns))
print("=" * 50)

for col in X_train.columns:
    print(col)
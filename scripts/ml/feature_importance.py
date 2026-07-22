import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/best_xgboost.pkl")

X_train = joblib.load("models/X_train.pkl")

importance = pd.Series(
    model.feature_importances_,
    index=X_train.columns
)

importance = importance.sort_values(ascending=False)

print(importance.head(20))

plt.figure(figsize=(10,8))
importance.head(20).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 20 Important Features")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()
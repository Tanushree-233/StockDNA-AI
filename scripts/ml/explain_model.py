import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("models/xgboost_model.pkl")

# -----------------------------
# Load Training Data
# -----------------------------

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")

print("=" * 50)
print("Generating SHAP Values...")
print("=" * 50)

# -----------------------------
# SHAP Explainer
# -----------------------------

explainer = shap.TreeExplainer(model)

# Calculate SHAP values for first 100 samples
shap_values = explainer.shap_values(X_test.iloc[:100])

print(type(shap_values))
print(shap_values.shape)
print(type(explainer.expected_value))
print(explainer.expected_value)

print("SHAP Values Generated!")

# -----------------------------
# Summary Plot
# -----------------------------

shap.summary_plot(
    shap_values,
    X_test.iloc[:100],
    show=False
)

plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=300)
plt.close()

print("Summary Plot Saved!")

# -----------------------------
# Waterfall Plot
# -----------------------------
sample = 0
prediction = int(model.predict(X_test.iloc[[sample]])[0])

print("Predicted Class:", prediction)

explanation = shap.Explanation(
    values=shap_values[sample, :, prediction],
    base_values=explainer.expected_value[prediction],
    data=X_test.iloc[sample].values,
    feature_names=X_test.columns.tolist()
)

shap.plots.waterfall(
    explanation,
    show=False
)

plt.tight_layout()
plt.savefig("models/shap_waterfall.png", dpi=300)
plt.close()

print("Waterfall Plot Saved!")
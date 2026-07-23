import shap
import numpy as np


def get_shap_explanation(model, X):
    """
    Returns the top 5 features influencing the prediction.
    """

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    prediction = int(model.predict(X)[0])

    # Multiclass SHAP
    if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        values = shap_values[0, :, prediction]
    else:
        values = shap_values[prediction][0]

    feature_names = X.columns.tolist()

    feature_impacts = []

    for feature, value in zip(feature_names, values):
        feature_impacts.append({
            "feature": feature,
            "impact": round(float(value), 4)
        })

    feature_impacts = sorted(
        feature_impacts,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return feature_impacts[:5]
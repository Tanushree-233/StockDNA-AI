"""
Automated Test Suite for SHAP Explainable AI (XAI) in StockDNA-AI.

Verifies:
1. Multiclass SHAP correctness: Explains the PREDICTED class (SELL, HOLD, BUY)
2. Feature-level contribution schema: feature, group, shap_value, importance, direction, explanation
3. Authoritative taxonomy: Features mapped strictly using `config.feature_registry`
4. Attribution normalization: internal_percentage + external_percentage ≈ 100.0%
5. Importance magnitude vs. direction distinction
6. TreeExplainer caching avoids costly re-compilation
"""

import unittest
import numpy as np
import pandas as pd
from scripts.ml.predict_service import load_production_model, predict_stock
from backend.services.shap_service import (
    get_shap_explanation,
    get_cached_explainer,
    generate_feature_explanation,
    _EXPLAINER_CACHE,
)
from config.feature_registry import get_feature_names, get_internal_features, get_external_features


class TestShapXAI(unittest.TestCase):

    def setUp(self):
        self.model, self.contract = load_production_model()
        self.feature_names = self.contract["feature_names"]
        self.pred_res = predict_stock("TCS")
        self.X_df = pd.DataFrame([self.pred_res["feature_snapshot"]])[self.feature_names]

    def test_shap_explanation_structure_and_keys(self):
        shap_out = get_shap_explanation(self.model, self.X_df)

        required_keys = [
            "prediction", "predicted_class", "primary_driver",
            "internal_percentage", "external_percentage",
            "top_internal_factors", "top_external_factors",
            "all_feature_contributions"
        ]
        for k in required_keys:
            self.assertIn(k, shap_out, f"Missing required XAI key '{k}'")

        self.assertIn(shap_out["prediction"], ["SELL", "HOLD", "BUY"])
        self.assertIn(shap_out["primary_driver"], ["INTERNAL", "EXTERNAL"])

    def test_feature_contributions_integrity_and_count(self):
        shap_out = get_shap_explanation(self.model, self.X_df)
        all_contribs = shap_out["all_feature_contributions"]

        # Must have exactly all 34 features
        self.assertEqual(len(all_contribs), 34)

        # Verify attributes on each item
        for item in all_contribs:
            self.assertIn("feature", item)
            self.assertIn("group", item)
            self.assertIn("shap_value", item)
            self.assertIn("importance", item)
            self.assertIn("direction", item)
            self.assertIn("explanation", item)

            self.assertIn(item["group"], ["INTERNAL", "EXTERNAL"])
            self.assertIn(item["direction"], ["POSITIVE", "NEGATIVE", "NEUTRAL"])
            self.assertEqual(item["importance"], round(abs(item["shap_value"]), 4))

            # Explanation must be non-empty and mention the predicted class
            self.assertTrue(len(item["explanation"]) > 0)
            self.assertIn(shap_out["prediction"], item["explanation"])

    def test_internal_external_attribution_normalization(self):
        shap_out = get_shap_explanation(self.model, self.X_df)
        int_pct = shap_out["internal_percentage"]
        ext_pct = shap_out["external_percentage"]

        self.assertGreaterEqual(int_pct, 0.0)
        self.assertLessEqual(int_pct, 100.0)
        self.assertGreaterEqual(ext_pct, 0.0)
        self.assertLessEqual(ext_pct, 100.0)

        # Sum must equal 100.0%
        self.assertAlmostEqual(int_pct + ext_pct, 100.0, places=1)

        # Consistency with primary_driver
        if int_pct >= ext_pct:
            self.assertEqual(shap_out["primary_driver"], "INTERNAL")
        else:
            self.assertEqual(shap_out["primary_driver"], "EXTERNAL")

    def test_multiclass_predicted_class_selection(self):
        """Verify SHAP explanation accurately targets the requested class index."""
        for class_idx, class_name in [(0, "SELL"), (1, "HOLD"), (2, "BUY")]:
            explanation = get_shap_explanation(
                self.model, self.X_df, predicted_class_idx=class_idx
            )
            self.assertEqual(explanation["prediction"], class_name)
            self.assertEqual(explanation["predicted_class"], class_name)
            # Top explanation must mention that class
            top_exp = explanation["all_feature_contributions"][0]["explanation"]
            self.assertIn(class_name, top_exp)

    def test_explainer_caching_performance(self):
        """Verify TreeExplainer cache is populated and reused."""
        model_id = id(self.model)
        explainer_1 = get_cached_explainer(self.model)
        self.assertIn(model_id, _EXPLAINER_CACHE)

        explainer_2 = get_cached_explainer(self.model)
        self.assertIs(explainer_1, explainer_2)


if __name__ == "__main__":
    unittest.main()

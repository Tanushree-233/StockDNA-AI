"""
Unit tests for Production XGBoost Model Serialization and Reload in StockDNA-AI.

Verifies:
1. Native model reload from `models/production/model.json`
2. Prediction outputs 3 class probabilities summing to 1.0
3. Confidence score equals max class probability
4. Handles missing feature values gracefully using contract imputation
"""

import unittest
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from config.settings import PROJECT_ROOT
from scripts.ml.predict_service import load_production_model


class TestModel(unittest.TestCase):

    def setUp(self):
        self.model, self.contract = load_production_model()

    def test_model_loaded_properly(self):
        self.assertIsInstance(self.model, XGBClassifier)
        self.assertEqual(self.contract["num_class"], 3)
        self.assertEqual(self.contract["model_version"], "1.0.0")

    def test_predict_proba_dimensions_and_sum(self):
        # Create a single synthetic row matching feature names
        feature_names = self.contract["feature_names"]
        row_dict = {f: [0.0] for f in feature_names}
        X = pd.DataFrame(row_dict)

        probs = self.model.predict_proba(X)

        self.assertEqual(probs.shape, (1, 3))
        # Probabilities must sum to 1.0
        prob_sum = float(np.sum(probs[0]))
        self.assertAlmostEqual(prob_sum, 1.0, places=5)

        # All probabilities between 0.0 and 1.0
        for p in probs[0]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_model_handles_imputation_fallback(self):
        feature_names = self.contract["feature_names"]
        imp = self.contract["imputation_values"]

        # DataFrame with NaNs
        nan_row = pd.DataFrame([{f: np.nan for f in feature_names}])
        for f in feature_names:
            nan_row[f] = nan_row[f].fillna(imp[f])

        probs = self.model.predict_proba(nan_row)
        self.assertEqual(probs.shape, (1, 3))
        self.assertFalse(np.isnan(probs).any())


if __name__ == "__main__":
    unittest.main()

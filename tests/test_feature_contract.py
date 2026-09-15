"""
Unit tests for Production Model Contract in StockDNA-AI.

Verifies:
1. `models/production/model_contract.json` exists and validates against specifications
2. Exact feature ordering and names match `config.feature_registry`
3. Imputation values are non-null and defined for all 34 features
4. Class mapping has 3 classes: SELL (0), HOLD (1), BUY (2)
"""

import os
import json
import unittest
from pathlib import Path

from config.settings import PROJECT_ROOT
from config.feature_registry import get_feature_names, get_internal_features, get_external_features


class TestFeatureContract(unittest.TestCase):

    def setUp(self):
        self.contract_path = PROJECT_ROOT / "models" / "production" / "model_contract.json"
        self.assertTrue(self.contract_path.exists(), f"Contract file not found at {self.contract_path}")

        with open(self.contract_path, "r", encoding="utf-8") as f:
            self.contract = json.load(f)

    def test_contract_metadata_fields(self):
        required_keys = [
            "model_version", "model_type", "objective", "num_class",
            "class_mapping", "target_definition", "horizon_trading_days",
            "thresholds", "feature_names", "feature_count", "feature_groups",
            "internal_feature_count", "external_feature_count", "imputation_values",
            "date_ranges", "hyperparameters"
        ]
        for key in required_keys:
            self.assertIn(key, self.contract, f"Missing required key '{key}' in contract")

    def test_feature_names_and_ordering(self):
        expected_names = get_feature_names()
        contract_names = self.contract["feature_names"]

        self.assertEqual(len(contract_names), 34)
        self.assertEqual(contract_names, expected_names)

        # Internal and external counts
        self.assertEqual(self.contract["internal_feature_count"], 10)
        self.assertEqual(self.contract["external_feature_count"], 24)
        self.assertEqual(self.contract["feature_groups"]["INTERNAL"], get_internal_features())
        self.assertEqual(self.contract["feature_groups"]["EXTERNAL"], get_external_features())

    def test_imputation_values_defined_for_all_features(self):
        imp = self.contract["imputation_values"]
        for feat in get_feature_names():
            self.assertIn(feat, imp, f"Feature {feat} has no imputation rule in contract")
            val = imp[feat]
            self.assertIsInstance(val, (int, float))
            self.assertFalse(val is None)

    def test_class_mapping(self):
        mapping = self.contract["class_mapping"]
        # Keys in JSON are strings: "0", "1", "2"
        self.assertEqual(mapping.get("0") or mapping.get(0), "SELL")
        self.assertEqual(mapping.get("1") or mapping.get(1), "HOLD")
        self.assertEqual(mapping.get("2") or mapping.get(2), "BUY")


if __name__ == "__main__":
    unittest.main()

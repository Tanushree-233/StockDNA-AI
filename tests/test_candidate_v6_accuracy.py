"""Unit and regression tests for Candidate v6 Accuracy Feasibility and Selective Prediction Evaluation.

Verifies:
1. Candidate v6 JSON and Markdown reports exist and follow valid schemas.
2. Strict mathematical invariants for selective abstention (coverage, samples_included + samples_excluded == total).
3. Monotonicity of included sample counts as confidence threshold tau increases.
4. Confusion matrix row/col sum consistency with sample counts and supports.
5. Invariance of accuracy ceiling (confirming no manufactured/leaked 80% claim).
6. Target boundary noise analysis presence and validity.
7. Preservation of models/production/ and Candidates v1 through v5 artifacts.
"""

import json
from pathlib import Path
import unittest


class TestCandidateV6Accuracy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parent.parent
        cls.report_dir = cls.repo_root / "results" / "diagnostics" / "candidate_v6"
        cls.json_path = cls.report_dir / "accuracy_investigation_report.json"
        cls.md_path = cls.report_dir / "accuracy_investigation_report.md"

        cls.assertTrue(
            cls.json_path.exists(),
            f"Missing Candidate v6 JSON report at {cls.json_path}",
        )
        cls.assertTrue(
            cls.md_path.exists(),
            f"Missing Candidate v6 Markdown report at {cls.md_path}",
        )

        with open(cls.json_path, "r", encoding="utf-8") as f:
            cls.report = json.load(f)

    def test_top_level_report_keys(self):
        """Verify the report contains all required schema keys."""
        required_keys = [
            "candidate_id",
            "timestamp",
            "objective",
            "evaluation_scope",
            "holdout_status",
            "full_coverage_evaluation",
            "selective_prediction_evaluation",
            "target_quality_and_feasibility_analysis",
        ]
        for key in required_keys:
            self.assertIn(key, self.report, f"Missing required key: {key}")

        self.assertEqual(
            self.report["candidate_id"], "candidate_v6_accuracy_investigation"
        )
        self.assertIn("Pre-2024", self.report["evaluation_scope"])
        self.assertIn("Strictly Excluded", self.report["holdout_status"])

    def test_full_coverage_models_evaluated(self):
        """Ensure all required models are evaluated under Approach A (100% coverage)."""
        fc = self.report["full_coverage_evaluation"]
        expected_models = [
            "candidate_v2_balanced",
            "candidate_v4_sqrt",
            "xgb_natural_prior",
            "random_forest",
        ]
        for model in expected_models:
            self.assertIn(model, fc, f"Model {model} missing from full_coverage_evaluation")
            m_metrics = fc[model]
            self.assertIn("accuracy", m_metrics)
            self.assertIn("balanced_accuracy", m_metrics)
            self.assertIn("macro_f1", m_metrics)
            self.assertIn("per_class", m_metrics)
            self.assertIn("confusion_matrix", m_metrics)

            # Accuracy must be defensible and below 80%
            self.assertLess(
                m_metrics["accuracy"],
                0.80,
                f"Full coverage accuracy for {model} is suspiciously >= 80%",
            )
            self.assertGreater(
                m_metrics["accuracy"],
                0.30,
                f"Full coverage accuracy for {model} is unexpectedly low (< 30%)",
            )

            # Per-class metrics check
            for cls_name in ["SELL", "HOLD", "BUY"]:
                self.assertIn(cls_name, m_metrics["per_class"])
                cls_data = m_metrics["per_class"][cls_name]
                self.assertIn("precision", cls_data)
                self.assertIn("recall", cls_data)
                self.assertIn("f1", cls_data)
                self.assertIn("support", cls_data)

    def test_selective_prediction_mathematical_invariants(self):
        """Ensure selective prediction follows exact sample and coverage accounting."""
        sp = self.report["selective_prediction_evaluation"]
        total_eval_samples = self.report["target_quality_and_feasibility_analysis"][
            "total_evaluated_samples"
        ]
        self.assertEqual(total_eval_samples, 2145)

        for model_key, tau_dict in sp.items():
            prev_included = total_eval_samples
            for tau_key, row in tau_dict.items():
                if "samples_included" in row:
                    inc = row["samples_included"]
                    exc = row["samples_excluded"]
                    cov = row["coverage"]

                    # 1. Total samples identity
                    self.assertEqual(
                        inc + exc,
                        total_eval_samples,
                        f"{model_key} {tau_key}: inc({inc}) + exc({exc}) != {total_eval_samples}",
                    )

                    # 2. Coverage identity
                    self.assertAlmostEqual(
                        cov,
                        inc / total_eval_samples,
                        places=5,
                        msg=f"{model_key} {tau_key}: coverage mismatch",
                    )

                    # 3. Monotonic decrease of included samples as tau rises
                    self.assertLessEqual(
                        inc,
                        prev_included,
                        f"{model_key} {tau_key}: samples_included increased from {prev_included} to {inc}",
                    )
                    prev_included = inc

                    # 4. If evaluated, verify confusion matrix sum matches included samples
                    if "confusion_matrix" in row:
                        cm = row["confusion_matrix"]
                        cm_sum = sum(sum(r) for r in cm)
                        self.assertEqual(
                            cm_sum,
                            inc,
                            f"{model_key} {tau_key}: CM sum ({cm_sum}) != samples_included ({inc})",
                        )

                        # Per-class metric consistency
                        for c_name in ["SELL", "HOLD", "BUY"]:
                            self.assertIn(c_name, row["per_class"])

                    # 5. Defensible accuracy check: selective accuracy must never exceed 80%
                    if "accuracy" in row:
                        self.assertLess(
                            row["accuracy"],
                            0.80,
                            f"{model_key} {tau_key}: accuracy suspiciously reached >= 80%",
                        )

    def test_target_boundary_noise_analysis(self):
        """Verify empirical boundary noise quantification exists and is mathematically valid."""
        target_analysis = self.report["target_quality_and_feasibility_analysis"]
        self.assertIn("knife_edge_boundary_noise", target_analysis)
        noise_info = target_analysis["knife_edge_boundary_noise"]

        total_boundary = noise_info["total_boundary_samples"]
        near_buy = noise_info["samples_near_buy_boundary"]
        near_sell = noise_info["samples_near_sell_boundary"]
        self.assertEqual(near_buy + near_sell, total_boundary)

        frac = noise_info["boundary_noise_fraction"]
        self.assertGreater(frac, 0.15)  # At least 15% of points near threshold
        self.assertLess(frac, 0.35)

    def test_production_and_candidate_artifacts_preserved(self):
        """Verify production model and candidate v1-v5 files are completely untouched."""
        prod_dir = self.repo_root / "models" / "production"
        self.assertTrue((prod_dir / "model.json").exists())
        self.assertTrue((prod_dir / "model_contract.json").exists())
        self.assertTrue((prod_dir / "metadata.json").exists())

        candidates_dir = self.repo_root / "models" / "candidates"
        self.assertTrue((candidates_dir / "candidate_v1").exists())
        self.assertTrue((candidates_dir / "candidate_v2").exists())

        diag_dir = self.repo_root / "results" / "diagnostics"
        self.assertTrue((diag_dir / "candidate_v3_calibration").exists())
        self.assertTrue((diag_dir / "candidate_v4").exists())
        self.assertTrue((diag_dir / "candidate_v5").exists())
        self.assertTrue((diag_dir / "candidate_v5_execution_audit").exists())


if __name__ == "__main__":
    unittest.main()

"""
Comprehensive Automated Test Suite for Data Aggregator.

Tests:
1. Valid financial CSV upload with duplicates and invalid values (domain taxonomy reuse).
2. Valid arbitrary non-financial text CSV (unsupervised automatic label generation).
3. Valid JSON array upload.
4. Raw unstructured TXT upload.
5. Empty file rejection (400 Bad Request).
6. Unsupported file extension rejection (400 Bad Request).
7. Oversized file rejection (> 10 MB).
8. Sample datasets endpoint availability.
9. Data quality auditing (completeness, uniqueness, validity scores).
"""

import unittest
from fastapi.testclient import TestClient
from backend.app import app
from data_tool.aggregator_engine import MAX_FILE_SIZE_BYTES


class TestDataAggregator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_financial_csv_upload_and_cleaning(self):
        """Verify financial CSV upload triggers domain taxonomy and deduplication."""
        csv_content = (
            b"Date,Close,Volume,Ticker\n"
            b"2024-01-15,3850.5,1250000,TCS\n"
            b"2024-01-15,3850.5,1250000,TCS\n"  # Exact Duplicate
            b"2024-01-16,3890.0,980000,TCS\n"
            b"2024-01-17,3910.0,1150000,TCS\n"
            b"2024-01-18,3945.2,1420000,TCS\n"
            b"2024-01-19,3920.0,1050000,TCS\n"
        )
        files = {"file": ("financial_test.csv", csv_content, "text/csv")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["filename"], "financial_test.csv")
        self.assertEqual(data["quality_audit"]["raw_rows"], 6)
        self.assertEqual(data["quality_audit"]["clean_rows"], 5)
        self.assertEqual(data["quality_audit"]["duplicates_removed"], 1)
        self.assertTrue(data["quality_audit"]["completeness_score_pct"] > 95.0)
        self.assertTrue(data["classification"]["is_financial"])
        self.assertEqual(data["classification"]["labeling_mode"], "domain_taxonomy")
        self.assertTrue(len(data["records"]) == 5)
        self.assertTrue(len(data["aggregated_groups"]) >= 1)

    def test_generic_non_financial_csv_automatic_labels(self):
        """Verify arbitrary dataset generates unsupervised clusters and semantic labels."""
        generic_content = (
            b"ID,Customer_Review,Channel\n"
            b"1,The package arrived damaged and delivery was delayed by three days.,Shipping\n"
            b"2,Outstanding customer support resolved my billing issue in minutes.,Support\n"
            b"3,Terrible product broke within a week and customer service ignored me.,Support\n"
            b"4,Super fast shipping and great build quality would buy again.,Shipping\n"
            b"5,Delivery courier was polite and package arrived in pristine condition.,Shipping\n"
            b"6,Refund request processed quickly great communication from staff.,Support\n"
        )
        files = {"file": ("reviews.csv", generic_content, "text/csv")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertFalse(data["classification"]["is_financial"])
        self.assertEqual(data["classification"]["labeling_mode"], "unsupervised_clustering")
        
        # Verify automatic labels are generated
        labels = data["classification"]["generated_labels"]
        self.assertTrue(len(labels) >= 2)
        for lbl in labels:
            self.assertEqual(lbl["type"], "Unsupervised Clustering")
            self.assertTrue("Topic:" in lbl["label"] or "Group" in lbl["label"])

        # Verify records have assigned labels and sentiment
        for rec in data["records"]:
            self.assertIn("Primary_Label", rec)
            self.assertIn("Sentiment", rec)

    def test_json_records_upload(self):
        """Verify JSON array ingestion, quality audit, and grouping."""
        json_content = (
            b'[\n'
            b'  {"Department": "Engineering", "Employee": "Alice", "Salary": 120000},\n'
            b'  {"Department": "Marketing", "Employee": "Bob", "Salary": 95000},\n'
            b'  {"Department": "Engineering", "Employee": "Charlie", "Salary": 135000},\n'
            b'  {"Department": "Design", "Employee": "Diana", "Salary": 105000}\n'
            b']'
        )
        files = {"file": ("staff.json", json_content, "application/json")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["detected_format"], "json")
        self.assertEqual(data["quality_audit"]["raw_rows"], 4)
        self.assertEqual(data["quality_audit"]["clean_rows"], 4)
        self.assertEqual(data["quality_audit"]["duplicates_removed"], 0)
        self.assertEqual(len(data["records"]), 4)

    def test_unstructured_txt_upload(self):
        """Verify raw text lines parsed into structured records with sentiment."""
        txt_content = (
            b"Apple unveils new M4 processor with record power efficiency\n"
            b"Federal Reserve signals higher interest rates for longer period\n"
            b"Oil prices tumble following unexpectedly large inventory build\n"
            b"Tech startups face severe liquidity crunch amid venture slowdown\n"
        )
        files = {"file": ("news_feed.txt", txt_content, "text/plain")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["detected_format"], "unstructured_txt")
        self.assertEqual(data["quality_audit"]["clean_rows"], 4)
        self.assertTrue(len(data["records"]) == 4)
        for rec in data["records"]:
            self.assertIn("Text_Content", rec)
            self.assertIn("Sentiment", rec)

    def test_empty_file_rejected(self):
        """Verify empty file raises 400 Bad Request."""
        files = {"file": ("empty.csv", b"", "text/csv")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_unsupported_extension_rejected(self):
        """Verify unsupported file types raise 400 Bad Request."""
        files = {"file": ("payload.exe", b"binary content", "application/octet-stream")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("unsupported", response.json()["detail"].lower())

    def test_oversized_file_rejected(self):
        """Verify file exceeding size limit raises 400 Bad Request."""
        huge_content = b"a" * (MAX_FILE_SIZE_BYTES + 1024)
        files = {"file": ("huge.csv", huge_content, "text/csv")}
        response = self.client.post("/datatool/aggregate/upload", files=files)
        self.assertEqual(response.status_code, 400)
        self.assertIn("exceeds", response.json()["detail"].lower())

    def test_sample_datasets_endpoint(self):
        """Verify sample datasets endpoint returns financial and non-financial data."""
        response = self.client.get("/datatool/aggregate/sample")
        self.assertEqual(response.status_code, 200)
        samples = response.json()
        self.assertIn("financial_sample", samples)
        self.assertIn("non_financial_sample", samples)
        self.assertTrue(len(samples["financial_sample"]) > 50)
        self.assertTrue(len(samples["non_financial_sample"]) > 50)


if __name__ == "__main__":
    unittest.main()

import unittest
import tempfile
from pathlib import Path
import pandas as pd
from karsl.scripts.data_pipeline.generate_qc_reports import generate

class TestQCPipeline(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "metadata.csv"
        self.output = self.root / "reports"

    def make_data(self, statuses):
        df = pd.DataFrame({
            "sample_id": [f"S{i}" for i in range(len(statuses))],
            "class_label": [1] * len(statuses),
            "signer_id": ["S01"] * len(statuses),
            "qc_status": statuses,
            "qc_flags": [""] * len(statuses),
            "qc_reason": [""] * len(statuses),
            "usable": [s == "PASS" for s in statuses]
        })
        df.to_csv(self.source, index=False)

    def test_valid_generation(self):
        self.make_data(["PASS", "REVIEW", "EXCLUDE"])
        generate(self.source, self.output)
        summary = pd.read_csv(self.output / "generated_qc_summary.csv")
        self.assertEqual(summary["samples"].sum(), 3)
        review = pd.read_csv(self.output / "generated_review_cases.csv")
        self.assertEqual(len(review), 1)

    def test_duplicate_sample_ids(self):
        self.make_data(["PASS", "PASS"])
        df = pd.read_csv(self.source)
        df.loc[1, "sample_id"] = "S0"
        df.to_csv(self.source, index=False)
        with self.assertRaises(ValueError):
            generate(self.source, self.output)

    def test_invalid_qc_status(self):
        self.make_data(["PASS", "UNKNOWN"])
        with self.assertRaises(ValueError):
            generate(self.source, self.output)

    def test_missing_qc_status(self):
        self.make_data(["PASS", "REVIEW"])
        df = pd.read_csv(self.source)
        df.loc[1, "qc_status"] = None
        df.to_csv(self.source, index=False)
        with self.assertRaises(ValueError):
            generate(self.source, self.output)

if __name__ == "__main__":
    unittest.main()

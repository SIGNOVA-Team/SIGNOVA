import tempfile
import unittest
from pathlib import Path

import pandas as pd

from karsl.scripts.data_pipeline.generate_karsl_splits import (
    COLUMNS,
    generate,
    validate,
)


class TestSplitPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "metadata.csv"

    def make_metadata(self):
        rows = []
        for sign_id in [1, 2]:
            for signer in ["S01", "S02", "S03"]:
                rows.append(
                    {
                        "sample_id": f"{sign_id}_{signer}",
                        "signer_id": signer,
                        "session_id": f"{signer}_SESSION",
                        "class_label": sign_id,
                        "author_split": "train",
                        "qc_status": "PASS",
                        "usable": True,
                    }
                )
        pd.DataFrame(rows).to_csv(self.source, index=False)

    def test_generation_shape(self):
        self.make_metadata()
        result = generate(self.source)
        self.assertEqual(len(result), 18)
        self.assertEqual(result["sample_id"].nunique(), 6)
        self.assertEqual(result["fold"].nunique(), 3)
        self.assertEqual(list(result.columns), COLUMNS)

    def test_signer_independence(self):
        self.make_metadata()
        result = generate(self.source)

        for _, fold in result.groupby("fold"):
            signer_sets = [
                set(fold.loc[fold["split"] == split, "signer_id"])
                for split in ["train", "validation", "test"]
            ]
            for i in range(3):
                for j in range(i + 1, 3):
                    self.assertTrue(signer_sets[i].isdisjoint(signer_sets[j]))

    def test_excluded_sample(self):
        self.make_metadata()
        df = pd.read_csv(self.source)
        df.loc[0, "qc_status"] = "EXCLUDE"
        df.loc[0, "usable"] = False
        df.to_csv(self.source, index=False)
        result = generate(self.source)
        self.assertNotIn("1_S01", set(result["sample_id"]))

    def test_duplicate_ids_rejected(self):
        self.make_metadata()
        df = pd.read_csv(self.source)
        df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
        df.to_csv(self.source, index=False)
        with self.assertRaises(ValueError):
            generate(self.source)

    def test_exact_reference_match(self):
        self.make_metadata()
        result = generate(self.source)
        passed, missing, differences = validate(result, result.copy())
        self.assertTrue(passed)
        self.assertEqual(missing, 0)
        self.assertTrue(all(v == 0 for v in differences.values()))

    def test_detect_changed_split(self):
        self.make_metadata()
        result = generate(self.source)
        reference = result.copy()
        reference.loc[0, "split"] = "validation"
        passed, missing, differences = validate(result, reference)
        self.assertFalse(passed)
        self.assertEqual(missing, 0)
        self.assertGreater(differences["split"], 0)


if __name__ == "__main__":
    unittest.main()

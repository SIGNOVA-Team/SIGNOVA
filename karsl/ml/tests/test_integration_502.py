import unittest

import numpy as np

from karsl.ml.integration.karsl_cv_adapter import extract_karsl_landmarks
from karsl.ml.integration.research_preview import as_research_candidate


class IntegrationTests(unittest.TestCase):
    def test_candidate_does_not_claim_acceptance(self):
        raw = {
            "status": "uncertain",
            "confidence": 0.27,
            "top_k": [{"class_id": 2, "label": "1", "label_arabic": "1", "score": 0.27}],
            "model_version": "cnn502",
            "calibration_version": "validation-v1",
        }
        output = as_research_candidate(raw)
        self.assertFalse(output["confirmed"])
        self.assertEqual(output["candidate"]["class_id"], 2)
        self.assertEqual(output["model_status"], "uncertain")
        self.assertNotIn("class_id", output)
        with self.assertRaises(ValueError):
            as_research_candidate({**raw, "status": "accepted"})

    def test_cv_frame_validation_before_detector(self):
        with self.assertRaises(ValueError):
            extract_karsl_landmarks([])
        with self.assertRaisesRegex(ValueError, "RGB uint8"):
            extract_karsl_landmarks([np.zeros((8, 8, 3), dtype=np.float32)])


if __name__ == "__main__":
    unittest.main()

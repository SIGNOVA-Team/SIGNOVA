import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
SCRIPTS = ROOT / "ml_expansion/scripts"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predict = load_module(SCRIPTS / "signova_predict_502.py", "karsl_predict")
        cls.trainer = load_module(ROOT / "training/train_gru_502.py", "karsl_gru_train")
        cls.frozen = cls.trainer.load_frozen_model_source(SCRIPTS / "signova_compare_cnn_502.py")

    def test_model_and_preprocessor_fingerprints(self):
        self.assertEqual(
            self.trainer.sha256(ROOT / "src/preprocessing.py"), self.predict.PREPROCESSING_SHA256
        )
        self.assertEqual(
            self.trainer.sha256(SCRIPTS / "signova_compare_cnn_502.py"),
            self.predict.MODEL_MODULE_SHA256,
        )
        self.assertEqual(self.frozen.CLASSES, list(range(1, 503)))

    def test_metric_class_order_and_values(self):
        class Model(torch.nn.Module):
            def forward(self, x):
                result = torch.full((len(x), 502), -10.0)
                result[0, 0] = 10.0
                result[1, 1] = 10.0
                return result

        x = np.zeros((2, 32, 150), dtype=np.float32)
        y = np.array([0, 1], dtype=np.int64)
        loader = self.frozen.make_loader(x, y, 2, False, torch.device("cpu"))
        result = self.frozen.evaluate(Model(), loader, torch.device("cpu"))
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["top5_accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 2 / 502)

    def test_reject_bad_split_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            x = np.zeros((2, 32, 150), dtype=np.float32)
            labels = np.array([0, 1], dtype=np.int64)
            with self.assertRaisesRegex(ValueError, "Invalid encoded labels"):
                self.trainer.train_gru(
                    self.frozen,
                    x,
                    np.array([0, 502]),
                    x,
                    labels,
                    Path(directory) / "bad",
                    torch.device("cpu"),
                    epochs=1,
                )
            self.assertFalse((Path(directory) / "bad").exists())

    def test_gru_training_and_reload_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            self.frozen.seed_all(42)
            x = np.zeros((4, 32, 150), dtype=np.float32)
            x[:, :, 2::3] = 1.0
            x[1, :, 0] = 0.25
            x[2, :, 0] = -0.25
            y = np.array([0, 1, 0, 1], dtype=np.int64)
            output = Path(directory) / "gru_smoke"
            result = self.trainer.train_gru(
                self.frozen, x, y, x, y, output, torch.device("cpu"), epochs=1
            )
            checkpoint = torch.load(
                output / "best_gru_502_pass_v1.pt", map_location="cpu", weights_only=True
            )
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(checkpoint["model_name"], "SignGRU502")
            self.assertEqual(checkpoint["input_shape"], [32, 150])
            self.assertFalse(checkpoint["test_evaluated"])
            self.assertTrue((output / "history.csv").is_file())
            self.assertFalse((output / "test_report.json").exists())
            self.assertEqual(json.loads((output / "result.json").read_text())["status"], "COMPLETE")

    @unittest.skipUnless(
        (ROOT / "ml_expansion/experiments/cnn_502_pass_v1/best_cnn_502_pass_v1.pt").is_file(),
        "Private checkpoint absent",
    )
    def test_prediction_validation_and_label_order(self):
        predictor = self.predict.KArSL502Predictor(ROOT / "ml_expansion", PROJECT_ROOT)
        features = predictor.preprocessing.preprocess_pose(self.predict.synthetic_pose())
        result = predictor.predict_features(features, self.predict.PREPROCESSING_VERSION, 3)
        self.assertEqual(result["status"], "uncertain")
        self.assertIsNone(result["class_id"])
        self.assertEqual(len(result["top_k"]), 3)
        self.assertTrue(all(1 <= c["class_id"] <= 502 for c in result["top_k"]))
        with self.assertRaisesRegex(ValueError, "Expected"):
            predictor.predict_features(np.zeros((31, 150)), self.predict.PREPROCESSING_VERSION)
        with self.assertRaises(ValueError):
            predictor.predict_features(
                np.full((32, 150), np.nan), self.predict.PREPROCESSING_VERSION
            )
        with self.assertRaisesRegex(ValueError, "Preprocessing version mismatch"):
            predictor.predict_features(features, "wrong")


if __name__ == "__main__":
    unittest.main(verbosity=2)

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from karsl.ml.training.train_baselines_502 import load_frozen_source, run_baselines


class BaselineTests(unittest.TestCase):
    def test_full_502_class_smoke_and_checkpoint_reload(self):
        root = Path(__file__).resolve().parents[1]
        frozen = load_frozen_source(root / "scripts/signova_compare_cnn_502.py")
        frozen.seed_all(42)
        x = np.zeros((502, 32, 150), dtype=np.float32)
        x[:, :, 2::3] = 1
        x[:, 0, 0:27:3] = ((np.arange(502)[:, None] >> np.arange(9)) & 1) * 2
        y = np.arange(502, dtype=np.int64)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline"
            result = run_baselines(
                frozen, x, y, x[:3], y[:3], output, torch.device("cpu"), epochs=1
            )
            self.assertEqual(result["centroid_validation"]["accuracy"], 1.0)
            self.assertEqual(result["linear_best_epoch"], 1)
            self.assertFalse(result["test_evaluated"])
            self.assertEqual(
                json.loads((output / "experiment.json").read_text())["class_ids"],
                list(range(1, 503)),
            )
            ckpt = torch.load(output / "linear_best.pt", map_location="cpu", weights_only=True)
            self.assertFalse(ckpt["test_evaluated"])
            with self.assertRaises(FileExistsError):
                run_baselines(frozen, x, y, x[:3], y[:3], output, torch.device("cpu"), epochs=1)

    def test_invalid_label_rejected_before_output(self):
        root = Path(__file__).resolve().parents[1]
        frozen = load_frozen_source(root / "scripts/signova_compare_cnn_502.py")
        x = np.zeros((1, 32, 150), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad"
            with self.assertRaisesRegex(ValueError, "Invalid encoded class IDs"):
                run_baselines(
                    frozen,
                    x,
                    np.array([502]),
                    x,
                    np.array([0]),
                    output,
                    torch.device("cpu"),
                    epochs=1,
                )
            self.assertFalse(output.exists())

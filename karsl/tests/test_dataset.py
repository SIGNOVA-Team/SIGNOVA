import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "validate_dataset.py"


def _get_validator():
    spec = importlib.util.spec_from_file_location("karsl_validate_dataset", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate


validate = _get_validator()


class TestDataset(unittest.TestCase):
    def test_dataset_integrity(self):
        for name, passed in validate().items():
            with self.subTest(check=name):
                self.assertTrue(passed)


if __name__ == "__main__":
    unittest.main()

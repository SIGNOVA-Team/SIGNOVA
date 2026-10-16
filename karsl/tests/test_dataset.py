
import unittest
from scripts.validate_dataset import validate

class TestDataset(unittest.TestCase):
    def test_dataset_integrity(self):
        for name, passed in validate().items():
            with self.subTest(check=name):
                self.assertTrue(passed)

if __name__ == "__main__":
    unittest.main()

"""Tests for the isolated SI held-out test evaluation workflow."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

import evaluate_si_test  # noqa: E402
from isharah_dataset import PreprocessingConfig  # noqa: E402


def test_report_schema_is_test_only_and_canonical():
    metrics = {
        "sample_count": 2000,
        "reference_tokens": 10,
        "substitutions": 1,
        "deletions": 2,
        "insertions": 3,
        "edit_errors": 6,
        "gloss_error_rate": 0.6,
        "exact_matches": 100,
        "exact_match": 0.05,
    }
    report = evaluate_si_test._build_report(
        "frozen.pt",
        torch.device("cpu"),
        PreprocessingConfig(missing_file_policy="error"),
        metrics,
        {"metadata_samples": 2000, "keypoint_files_present": 2000},
        {"valid_samples": 2000, "rejected_samples": []},
        2000,
    )

    assert report["split"] == "test"
    assert set(report["metrics"]) == {
        "sample_count",
        "reference_tokens",
        "substitutions",
        "deletions",
        "insertions",
        "edit_errors",
        "gloss_error_rate",
        "exact_matches",
        "exact_match",
    }
    assert report["metrics"]["edit_errors"] == sum(
        report["metrics"][key] for key in ("substitutions", "deletions", "insertions")
    )
    json.dumps(report, ensure_ascii=False)


def test_availability_check_does_not_load_keypoint_arrays(monkeypatch):
    class SampleSet:
        samples = [
            {"keypoints_path": "data/processed/a.npy"},
            {"keypoints_path": "data/processed/b.npy"},
        ]

        def __len__(self):
            return len(self.samples)

    monkeypatch.setattr(
        evaluate_si_test,
        "build_si_datasets",
        lambda project_root, data_root: (None, None, SampleSet(), None),
    )
    monkeypatch.setattr(
        evaluate_si_test,
        "validate_ctc_lengths",
        lambda *_args, **_kwargs: pytest.fail("CTC evaluation must not run"),
    )

    result = evaluate_si_test.inspect_si_test_availability(
        ".",
        "/definitely/missing",
        expected_samples=2,
    )

    assert result["split"] == "test"
    assert result["metadata_samples"] == 2
    assert result["keypoint_files_present"] == 0
    assert len(result["missing_keypoint_files"]) == 2
    assert not result["available"]


def test_evaluation_entry_point_is_read_only():
    source = Path(evaluate_si_test.__file__).read_text(encoding="utf-8")
    assert "torch.optim" not in source
    assert "save_checkpoint" not in source
    assert "train_si(" not in source

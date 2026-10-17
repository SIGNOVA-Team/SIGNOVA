"""Tests for resumable hand-keypoint processing."""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from hand_pipeline import process_sample  # noqa: E402
from keypoint_validation import is_valid_keypoint_file  # noqa: E402


def make_keypoints(frame_count=3):
    """Create a valid synthetic keypoint sequence."""
    return np.zeros((frame_count, 2, 63), dtype=np.float32)


def prepare_directories(tmp_path):
    """Create isolated directories and three placeholder frames."""
    sample_directory = tmp_path / "02_0001"
    raw_directory = tmp_path / "raw"
    tracked_directory = tmp_path / "tracked"

    sample_directory.mkdir()
    raw_directory.mkdir()
    tracked_directory.mkdir()

    for frame_index in range(3):
        (sample_directory / f"frame{frame_index:04d}.jpg").touch()

    return sample_directory, raw_directory, tracked_directory


def test_recovers_corrupted_tracked_file(tmp_path):
    """Rebuild tracked output without rerunning extraction."""
    sample_directory, raw_directory, tracked_directory = prepare_directories(tmp_path)

    keypoints = make_keypoints()
    np.save(raw_directory / "02_0001.npy", keypoints)

    tracked_path = tracked_directory / "02_0001.npy"
    tracked_path.write_bytes(b"corrupted")

    with patch("hand_pipeline.extract_sample_hand_keypoints") as extraction_mock:
        sample_id, status = process_sample(
            sample_directory,
            tmp_path / "model.task",
            raw_directory,
            tracked_directory,
        )

    extraction_mock.assert_not_called()
    assert sample_id == "02_0001"
    assert status == "processed"
    assert is_valid_keypoint_file(tracked_path, expected_frames=3)


def test_recovers_corrupted_raw_file(tmp_path):
    """Rerun extraction when raw keypoints are corrupted."""
    sample_directory, raw_directory, tracked_directory = prepare_directories(tmp_path)

    raw_path = raw_directory / "02_0001.npy"
    raw_path.write_bytes(b"corrupted")

    np.save(
        tracked_directory / "02_0001.npy",
        make_keypoints(),
    )

    expected_keypoints = make_keypoints()
    detection_counts = np.full(3, 2, dtype=np.int32)

    with patch(
        "hand_pipeline.extract_sample_hand_keypoints",
        return_value=(expected_keypoints, detection_counts),
    ) as extraction_mock:
        sample_id, status = process_sample(
            sample_directory,
            tmp_path / "model.task",
            raw_directory,
            tracked_directory,
        )

    extraction_mock.assert_called_once()
    assert sample_id == "02_0001"
    assert status == "processed"
    assert is_valid_keypoint_file(raw_path, expected_frames=3)
    assert is_valid_keypoint_file(
        tracked_directory / "02_0001.npy",
        expected_frames=3,
    )

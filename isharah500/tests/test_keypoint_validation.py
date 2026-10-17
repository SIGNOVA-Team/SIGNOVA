"""Unit tests for saved keypoint validation."""

import sys
from pathlib import Path

import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from keypoint_validation import is_valid_keypoint_file  # noqa: E402


def test_valid_keypoint_file(tmp_path):
    """Accept a correctly formatted keypoint array."""
    file_path = tmp_path / "valid.npy"
    keypoints = np.full((10, 2, 63), np.nan, dtype=np.float32)
    np.save(file_path, keypoints)

    assert is_valid_keypoint_file(file_path, expected_frames=10)


def test_missing_keypoint_file(tmp_path):
    """Reject files that do not exist."""
    assert not is_valid_keypoint_file(tmp_path / "missing.npy")


def test_corrupted_keypoint_file(tmp_path):
    """Reject unreadable NumPy files."""
    file_path = tmp_path / "corrupted.npy"
    file_path.write_bytes(b"invalid numpy data")

    assert not is_valid_keypoint_file(file_path)


def test_invalid_shape(tmp_path):
    """Reject arrays with unexpected dimensions."""
    file_path = tmp_path / "invalid_shape.npy"
    np.save(file_path, np.zeros((10, 63), dtype=np.float32))

    assert not is_valid_keypoint_file(file_path)


def test_invalid_dtype(tmp_path):
    """Reject arrays with unexpected data types."""
    file_path = tmp_path / "invalid_dtype.npy"
    np.save(file_path, np.zeros((10, 2, 63), dtype=np.float64))

    assert not is_valid_keypoint_file(file_path)


def test_invalid_frame_count(tmp_path):
    """Reject arrays with an unexpected frame count."""
    file_path = tmp_path / "invalid_frames.npy"
    np.save(file_path, np.zeros((10, 2, 63), dtype=np.float32))

    assert not is_valid_keypoint_file(file_path, expected_frames=20)


def test_infinite_values(tmp_path):
    """Reject arrays containing infinite coordinates."""
    file_path = tmp_path / "infinite.npy"
    keypoints = np.zeros((10, 2, 63), dtype=np.float32)
    keypoints[0, 0, 0] = np.inf
    np.save(file_path, keypoints)

    assert not is_valid_keypoint_file(file_path)

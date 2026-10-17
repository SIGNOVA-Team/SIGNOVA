"""Unit tests for experimental hand-slot tracking."""

import sys
from pathlib import Path

import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from hand_tracking import track_hand_slots  # noqa: E402


def make_keypoints(frame_count=3):
    """Create a small, valid hand-keypoint sequence."""
    keypoints = np.full(
        (frame_count, 2, 63),
        np.nan,
        dtype=np.float32,
    )

    keypoints[:, 0, :3] = [0.2, 0.3, 0.0]
    keypoints[:, 1, :3] = [0.7, 0.3, 0.0]

    return keypoints


def test_tracking_preserves_shape_and_dtype():
    """Tracking should preserve array structure."""
    original = make_keypoints()

    tracked = track_hand_slots(original)

    assert tracked.shape == original.shape
    assert tracked.dtype == original.dtype


def test_tracking_preserves_detection_counts():
    """Tracking should retain the number of detected hands."""
    original = make_keypoints()
    original[1, 1] = np.nan

    tracked = track_hand_slots(original)

    original_counts = np.isfinite(original[:, :, :2]).all(axis=2).sum(axis=1)

    tracked_counts = np.isfinite(tracked[:, :, :2]).all(axis=2).sum(axis=1)

    np.testing.assert_array_equal(
        original_counts,
        tracked_counts,
    )


def test_tracking_preserves_valid_coordinates():
    """Tracking should not modify landmark values."""
    original = make_keypoints()
    original[1, 1] = np.nan

    tracked = track_hand_slots(original)

    original_values = np.sort(original[np.isfinite(original)])
    tracked_values = np.sort(tracked[np.isfinite(tracked)])

    np.testing.assert_array_equal(
        original_values,
        tracked_values,
    )

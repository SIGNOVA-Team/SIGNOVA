"""Validate saved hand-keypoint arrays."""

from pathlib import Path

import numpy as np


HAND_COUNT = 2
HAND_FEATURE_COUNT = 63


def is_valid_keypoint_file(file_path, expected_frames=None):
    """Check whether a saved keypoint file is structurally valid."""
    file_path = Path(file_path)

    if not file_path.is_file():
        return False

    try:
        keypoints = np.load(file_path, allow_pickle=False)
    except (OSError, ValueError, EOFError):
        return False

    if keypoints.ndim != 3:
        return False

    if keypoints.shape[0] == 0:
        return False

    if keypoints.shape[1:] != (HAND_COUNT, HAND_FEATURE_COUNT):
        return False

    if keypoints.dtype != np.float32:
        return False

    if expected_frames is not None:
        if keypoints.shape[0] != expected_frames:
            return False

    return bool(
        np.all(np.isfinite(keypoints) | np.isnan(keypoints))
    )

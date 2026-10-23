"""Versioned feature representations for Isharah-500 keypoint sequences."""

from __future__ import annotations

import numpy as np

FEATURE_REPRESENTATION_RAW_63 = "raw_63"
FEATURE_REPRESENTATION_GLOBAL_POSITION_V1 = "global_position_v1"
GLOBAL_POSITION_V1_EXTRA_FEATURES = 9


def output_features_per_hand(feature_representation: str) -> int:
    """Return the model feature width for a versioned representation."""
    if feature_representation == FEATURE_REPRESENTATION_RAW_63:
        return 63
    if feature_representation == FEATURE_REPRESENTATION_GLOBAL_POSITION_V1:
        return 63 + GLOBAL_POSITION_V1_EXTRA_FEATURES
    raise ValueError(f"Unsupported feature representation: {feature_representation}")


def construct_feature_representation(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    *,
    feature_representation: str,
    global_coordinate_scale: float = 1.0,
    global_keypoints: np.ndarray | None = None,
) -> np.ndarray:
    """Construct deterministic model features from preprocessed keypoints.

    ``global_position_v1`` appends nine channels to each hand: own wrist
    position, other-wrist minus own-wrist displacement, and frame-to-frame
    own-wrist velocity. Positions use the native MediaPipe coordinate
    convention. When local hand-centric features are supplied in ``keypoints``,
    ``global_keypoints`` must contain the finite interpolated coordinates before
    local normalization. Required global channels are zeroed when observations
    are unavailable, while the original detection mask is preserved separately.
    """
    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3:
        raise ValueError("keypoints must have shape (T, 2, 63)")
    if keypoints.shape[1:] != (2, 63):
        raise ValueError(f"Expected keypoints shape (T, 2, 63), got {keypoints.shape}")
    if valid_mask.shape != keypoints.shape[:2]:
        raise ValueError("valid_mask must have shape (T, 2)")
    if global_coordinate_scale <= 0 or not np.isfinite(global_coordinate_scale):
        raise ValueError("global_coordinate_scale must be finite and positive")
    if not np.isfinite(keypoints).all():
        raise ValueError("feature construction requires finite keypoints")

    if feature_representation == FEATURE_REPRESENTATION_RAW_63:
        return keypoints.astype(np.float32, copy=True)
    if feature_representation != FEATURE_REPRESENTATION_GLOBAL_POSITION_V1:
        raise ValueError(f"Unsupported feature representation: {feature_representation}")

    if global_keypoints is None:
        global_keypoints = keypoints
    if not isinstance(global_keypoints, np.ndarray) or global_keypoints.shape != keypoints.shape:
        raise ValueError("global_keypoints must have the same shape as keypoints")
    if not np.isfinite(global_keypoints).all():
        raise ValueError("global feature source contains nonfinite values")

    frames = keypoints.shape[0]
    landmarks = global_keypoints.reshape(frames, 2, 21, 3)
    wrists = landmarks[:, :, 0, :] / np.float32(global_coordinate_scale)
    valid = valid_mask.astype(bool, copy=False)
    own_wrist = np.where(valid[:, :, None], wrists, 0.0)

    inter_hand = np.zeros_like(wrists, dtype=np.float32)
    both_valid = valid[:, 0] & valid[:, 1]
    inter_hand[both_valid, 0] = wrists[both_valid, 1] - wrists[both_valid, 0]
    inter_hand[both_valid, 1] = wrists[both_valid, 0] - wrists[both_valid, 1]

    velocity = np.zeros_like(wrists, dtype=np.float32)
    if frames > 1:
        consecutive_valid = valid[1:] & valid[:-1]
        velocity[1:] = np.where(
            consecutive_valid[:, :, None],
            wrists[1:] - wrists[:-1],
            0.0,
        )

    extra = np.concatenate((own_wrist, inter_hand, velocity), axis=-1)
    output = np.concatenate((keypoints, extra), axis=-1).astype(np.float32)
    if not np.isfinite(output).all():
        raise ValueError("feature construction produced nonfinite values")
    return output

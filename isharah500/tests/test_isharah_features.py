"""Tests for the optional global-position-v1 feature representation."""

import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_dataset import (  # noqa: E402
    IsharahDataset,
    PreprocessingConfig,
)
from isharah_features import (  # noqa: E402
    FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    FEATURE_REPRESENTATION_RAW_63,
    construct_feature_representation,
)


def make_keypoints(frames: int = 3) -> np.ndarray:
    values = np.zeros((frames, 2, 63), dtype=np.float32)
    positions = np.array([0, 1, 3], dtype=np.float32)[:frames]
    values[:, 0, 0] = positions
    values[:, 1, 0:3] = values[:, 0, 0:3] + np.array([2, 0, 0], dtype=np.float32)
    return values


def test_raw_representation_preserves_default_shape_and_values():
    keypoints = make_keypoints()
    mask = np.ones((3, 2), dtype=bool)

    output = construct_feature_representation(
        keypoints,
        mask,
        feature_representation=FEATURE_REPRESENTATION_RAW_63,
    )

    assert output.shape == (3, 2, 63)
    np.testing.assert_array_equal(output, keypoints)


def test_global_position_v1_has_expected_shape_and_motion_channels():
    keypoints = make_keypoints()
    mask = np.ones((3, 2), dtype=bool)

    output = construct_feature_representation(
        keypoints,
        mask,
        feature_representation=FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    )

    assert output.shape == (3, 2, 72)
    # Appended channels are own wrist, inter-hand displacement, velocity.
    np.testing.assert_array_equal(output[:, 0, 63:66], keypoints[:, 0, 0:3])
    np.testing.assert_array_equal(output[:, 0, 66:69], np.tile([2, 0, 0], (3, 1)))
    np.testing.assert_array_equal(output[:, 0, 69:72], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    assert np.isfinite(output).all()


def test_global_channels_use_pre_normalization_wrist_source():
    local = np.zeros((2, 2, 63), dtype=np.float32)
    global_source = make_keypoints(2)
    mask = np.ones((2, 2), dtype=bool)

    output = construct_feature_representation(
        local,
        mask,
        feature_representation=FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
        global_keypoints=global_source,
    )

    np.testing.assert_array_equal(output[:, 0, :63], local[:, 0, :])
    np.testing.assert_array_equal(output[:, 0, 63:66], global_source[:, 0, 0:3])


def test_missing_hand_zeroes_global_channels_but_keeps_finite_output():
    keypoints = make_keypoints()
    mask = np.ones((3, 2), dtype=bool)
    mask[1:, 1] = False

    output = construct_feature_representation(
        keypoints,
        mask,
        feature_representation=FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    )

    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[1:, 1, 63:], 0.0)
    np.testing.assert_array_equal(output[1:, 0, 66:69], 0.0)
    np.testing.assert_array_equal(output[1:, 0, 69:72], [[1, 0, 0], [2, 0, 0]])


def test_nan_input_is_rejected_before_feature_construction():
    keypoints = make_keypoints()
    keypoints[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite keypoints"):
        construct_feature_representation(
            keypoints,
            np.ones((3, 2), dtype=bool),
            feature_representation=FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
        )


def test_dataset_and_inference_preprocessing_produce_identical_features(tmp_path):
    torch = pytest.importorskip("torch")
    from isharah_inference import preprocess_keypoints

    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,alpha,text,s1.npy\n",
        encoding="utf-8",
    )
    raw = make_keypoints()
    np.save(tmp_path / "s1.npy", raw)
    config = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization="hand_centric",
        feature_representation=FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    )
    dataset = IsharahDataset(
        metadata,
        root_dir=tmp_path,
        preprocessing_config=config,
        strict=True,
    )
    dataset_features = dataset[0]["keypoints"].detach().cpu().numpy()
    inference_features, inference_mask = preprocess_keypoints(raw, config)

    assert isinstance(dataset[0]["keypoints"], torch.Tensor)
    np.testing.assert_allclose(dataset_features, inference_features)
    np.testing.assert_array_equal(dataset[0]["mask"].numpy(), inference_mask)

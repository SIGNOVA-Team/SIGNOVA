"""Tests for the checkpoint-backed Isharah prediction interface."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

import isharah_inference  # noqa: E402
from isharah_ctc import BiGRUCTC  # noqa: E402
from isharah_dataset import GlossVocabulary, PreprocessingConfig  # noqa: E402
from isharah_inference import (  # noqa: E402
    IsharahPredictor,
    load_predictor,
    preprocess_keypoints,
)


def make_predictor(monkeypatch, *, normalization="hand_centric"):
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha", "beta"])
    model = BiGRUCTC(len(vocabulary)).eval()
    preprocessing = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization=normalization,
    )
    monkeypatch.setattr(
        isharah_inference,
        "load_checkpoint",
        lambda checkpoint_path, device: (model, vocabulary, preprocessing, {}),
    )
    return load_predictor("trusted.pt", device_name="cpu")


def test_input_validation_rejects_invalid_sequences(monkeypatch):
    predictor = make_predictor(monkeypatch)
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((4, 63), dtype=np.float32))
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((4, 2, 63), dtype=np.float64))
    with pytest.raises(ValueError):
        predictor.predict(np.full((4, 2, 63), np.inf, dtype=np.float32))
    with pytest.raises(ValueError):
        predictor.predict(np.zeros((0, 2, 63), dtype=np.float32))


def test_preprocessing_matches_existing_hand_centric_contract():
    config = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization="hand_centric",
    )
    raw = np.zeros((3, 2, 63), dtype=np.float32)
    raw[:, 0, 0:3] = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])
    raw[:, 0, 27:30] = raw[:, 0, 0:3] + np.array([2, 0, 0])
    raw[1, 0, 3] = np.nan

    processed, mask = preprocess_keypoints(raw, config)

    assert mask.shape == (3, 2)
    assert mask[1, 0]
    assert np.isfinite(processed).all()
    np.testing.assert_allclose(processed[:, 0, 0:3], 0.0)


def test_nan_handling_follows_saved_preprocessing_configuration():
    raw = np.zeros((4, 2, 63), dtype=np.float32)
    raw[1, 0, 4] = np.nan

    interpolating = PreprocessingConfig(
        missing_file_policy="error",
        nan_strategy="interpolate",
    )
    processed, mask = preprocess_keypoints(raw, interpolating)
    assert np.isfinite(processed).all()
    assert mask.shape == (4, 2)

    rejecting = PreprocessingConfig(
        missing_file_policy="error",
        nan_strategy="none",
    )
    with pytest.raises(ValueError, match="nan_strategy='none'"):
        preprocess_keypoints(raw, rejecting)


def test_predictions_are_deterministic(monkeypatch):
    predictor = make_predictor(monkeypatch, normalization="none")
    raw = np.zeros((5, 2, 63), dtype=np.float32)

    first = predictor.predict(raw)
    second = predictor.predict(raw)

    assert first == second
    assert first.input_length == 5
    assert first.gloss_text == " ".join(first.gloss_tokens)


def test_prediction_output_schema_is_stable(monkeypatch):
    predictor = make_predictor(monkeypatch, normalization="none")
    output = predictor.predict(np.zeros((4, 2, 63), dtype=np.float32)).to_dict()

    assert set(output) == {
        "gloss_tokens",
        "gloss_text",
        "input_length",
        "valid_hand_coverage",
        "device",
        "preprocessing_config",
    }
    assert isinstance(output["gloss_tokens"], list)
    assert isinstance(output["gloss_text"], str)
    assert output["input_length"] == 4
    assert 0.0 <= output["valid_hand_coverage"] <= 1.0
    assert output["device"] == "cpu"
    assert isinstance(output["preprocessing_config"], dict)
    json.dumps(output, ensure_ascii=False)


def test_prediction_restores_saved_preprocessing_contract(monkeypatch):
    predictor = make_predictor(monkeypatch, normalization="hand_centric")
    output = predictor.predict(np.zeros((4, 2, 63), dtype=np.float32)).to_dict()

    assert output["preprocessing_config"] == predictor.preprocessing_config.to_dict()
    assert predictor.model.config.num_hands == 2
    assert predictor.model.config.features_per_hand == 63
    json.dumps(output, ensure_ascii=False)


def test_load_predictor_loads_checkpoint_once(monkeypatch):
    calls = []
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha", "beta"])
    model = BiGRUCTC(len(vocabulary)).eval()
    preprocessing = PreprocessingConfig(missing_file_policy="error")

    def counted(checkpoint_path, device):
        calls.append(checkpoint_path)
        return model, vocabulary, preprocessing, {}

    monkeypatch.setattr(isharah_inference, "load_checkpoint", counted)
    predictor = load_predictor("trusted.pt", device_name="cpu")
    raw = np.zeros((3, 2, 63), dtype=np.float32)
    predictor.predict(raw)
    predictor.predict(raw)
    assert calls == ["trusted.pt"]


def test_real_si_dev_prediction_smoke():
    """Run one prediction when explicit trusted checkpoint and data roots are supplied."""
    project_root_value = os.environ.get("ISHARAH_PROJECT_ROOT")
    data_root_value = os.environ.get("ISHARAH_DATA_ROOT")
    if not project_root_value or not data_root_value:
        pytest.skip("set ISHARAH_PROJECT_ROOT and ISHARAH_DATA_ROOT for the real-data smoke test")
    project_root = Path(project_root_value)
    data_root = Path(data_root_value)
    checkpoint = project_root / "isharah500" / "checkpoints" / "si_hand_centric_resume_15ep" / "isharah500_si_best.pt"
    if not checkpoint.is_file():
        pytest.skip("trusted SI checkpoint is not available")
    keypoints = data_root / "processed" / "isharah500" / "hand_keypoints_tracked" / "dev" / "02_0001.npy"
    if not keypoints.is_file():
        pytest.skip("external SI data root is not available")
    prediction = load_predictor(checkpoint, device_name="cpu").predict(
        np.load(keypoints, allow_pickle=False)
    )
    assert prediction.input_length > 0
    assert isinstance(prediction.gloss_text, str)

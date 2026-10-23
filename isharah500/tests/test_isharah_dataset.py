"""Unit tests for Isharah-500 ML Dataset and Collation Module."""

import sys
from pathlib import Path

import numpy as np
import pytest

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_dataset import (  # noqa: E402
    GlossVocabulary,
    IsharahDataset,
    PreprocessingConfig,
    build_training_vocabulary,
    collate_isharah_batch,
    collate_isharah_ctc_batch,
    interpolate_keypoints,
    normalize_hand_centric,
    validate_keypoints,
)


def test_interpolate_keypoints_linear():
    """Verify linear interpolation fills NaNs correctly."""
    # 5 frames, 1 hand, 1 coordinate
    data = np.array([[[1.0]], [[np.nan]], [[3.0]], [[np.nan]], [[5.0]]], dtype=np.float32)
    interpolated = interpolate_keypoints(data)

    expected = np.array([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]]], dtype=np.float32)
    np.testing.assert_allclose(interpolated, expected, rtol=1e-5)


def test_interpolate_keypoints_edge_nans():
    """Verify boundary NaNs are extrapolated from nearest valid values."""
    data = np.array([[[np.nan]], [[2.0]], [[np.nan]], [[4.0]], [[np.nan]]], dtype=np.float32)
    interpolated = interpolate_keypoints(data)

    assert not np.isnan(interpolated).any()
    assert interpolated[0, 0, 0] == 2.0
    assert interpolated[2, 0, 0] == 3.0
    assert interpolated[4, 0, 0] == 4.0


def test_interpolate_all_nans():
    """Verify an array of complete NaNs is safely converted to zeros."""
    data = np.full((10, 2, 63), np.nan, dtype=np.float32)
    interpolated = interpolate_keypoints(data)

    assert not np.isnan(interpolated).any()
    np.testing.assert_allclose(interpolated, np.zeros((10, 2, 63)))


def test_hand_centric_normalization_is_finite_for_missing_hands():
    """Missing hands and zero-sized references must not create NaNs."""
    data = np.full((4, 2, 63), np.nan, dtype=np.float32)
    data[:, 0, 0:3] = 0.0
    data[:, 0, 27:30] = 0.0
    mask = np.zeros((4, 2), dtype=bool)
    mask[:, 0] = True

    normalized = normalize_hand_centric(data, mask)

    assert normalized.shape == data.shape
    assert np.isfinite(normalized).all()
    np.testing.assert_allclose(normalized, 0.0)


def test_hand_centric_normalization_preserves_scaled_temporal_movement():
    """Wrist centering preserves relative landmark movement up to scale."""
    data = np.zeros((3, 2, 63), dtype=np.float32)
    data[:, 0, 0:3] = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    data[:, 0, 3:6] = data[:, 0, 0:3] + np.array([0, 1, 0])
    data[:, 0, 27:30] = data[:, 0, 0:3] + np.array([3, 0, 0])
    data[:, 1, 27:30] = np.array([2, 0, 0])
    mask = np.ones((3, 2), dtype=bool)

    normalized = normalize_hand_centric(data, mask)

    np.testing.assert_allclose(normalized[:, 0, 0:3], 0.0)
    np.testing.assert_allclose(
        normalized[:, 0, 3:6],
        np.tile([0, 1 / 3, 0], (3, 1)),
    )
    assert np.isfinite(normalized).all()


def test_default_preprocessing_keeps_coordinates_unchanged(tmp_path):
    """The baseline path remains unchanged when normalization is disabled."""
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,معلم,نص,s1.npy\n",
        encoding="utf-8",
    )
    original = np.zeros((2, 2, 63), dtype=np.float32)
    original[:, 0, 0:3] = 5.0
    original[:, 0, 27:30] = 7.0
    np.save(tmp_path / "s1.npy", original)

    dataset = IsharahDataset(metadata, root_dir=tmp_path, missing_file_policy="error")
    np.testing.assert_array_equal(np.asarray(dataset[0]["keypoints"]), original)


def test_dataset_applies_configured_hand_centric_normalization(tmp_path):
    """The dataset applies the serialized normalization contract after interpolation."""
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,معلم,نص,s1.npy\n",
        encoding="utf-8",
    )
    original = np.zeros((2, 2, 63), dtype=np.float32)
    original[:, 0, 0:3] = np.array([[2, 0, 0], [3, 0, 0]])
    original[:, 0, 3:6] = original[:, 0, 0:3] + np.array([0, 2, 0])
    original[:, 0, 27:30] = original[:, 0, 0:3] + np.array([4, 0, 0])
    np.save(tmp_path / "s1.npy", original)

    config = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization="hand_centric",
    )
    dataset = IsharahDataset(metadata, root_dir=tmp_path, preprocessing_config=config)
    item = dataset[0]

    assert item["mask"].tolist() == [[True, True], [True, True]]
    np.testing.assert_allclose(np.asarray(item["keypoints"])[:, 0, 0:3], 0.0)
    np.testing.assert_allclose(
        np.asarray(item["keypoints"])[:, 0, 3:6],
        np.tile([0, 0.5, 0], (2, 1)),
        rtol=1e-6,
    )
    assert np.isfinite(np.asarray(item["keypoints"])).all()


def test_preprocessing_config_serializes_normalization():
    """Normalization settings are part of the inference contract."""
    config = PreprocessingConfig(
        missing_file_policy="error",
        coordinate_normalization="hand_centric",
        normalization_epsilon=1e-5,
    )
    assert PreprocessingConfig.from_dict(config.to_dict()) == config


def test_gloss_vocabulary():
    """Verify GlossVocabulary tokenization and reconstruction."""
    vocab = GlossVocabulary(add_ctc_blank=True)
    sequences = ["معلم لغه اشاره", "انا طالب مدرسه"]
    vocab.build_vocab(sequences)

    # Encode with and without special tokens
    encoded = vocab.encode("معلم اشاره", add_sos_eos=False)
    assert len(encoded) == 2
    decoded = vocab.decode(encoded)
    assert decoded == "معلم اشاره"

    # Unknown token
    encoded_unk = vocab.encode("كلمه_غريبه")
    assert encoded_unk[0] == vocab.token2id[GlossVocabulary.UNK_TOKEN]


def test_isharah_dataset_loading(tmp_path):
    """Verify dataset loading and exclude_flagged filtering."""
    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(
        "sample_id,gloss,text,keypoints_path\n"
        "01_0001,سلام,مرحبا,data/processed/01_0001.npy\n"
        "01_0002,شكرا,شكرا لك,data/processed/01_0002.npy\n",
        encoding="utf-8",
    )

    review_file = tmp_path / "review.csv"
    review_file.write_text(
        "sample_id,review_required\n01_0001,False\n01_0002,True\n",
        encoding="utf-8",
    )

    # Dataset without filtering
    ds_all = IsharahDataset(
        metadata_csv=meta_file,
        root_dir=tmp_path,
        exclude_flagged=False,
    )
    assert len(ds_all) == 2

    # Dataset with exclude_flagged=True
    ds_filtered = IsharahDataset(
        metadata_csv=meta_file,
        root_dir=tmp_path,
        exclude_flagged=True,
        review_csv=review_file,
    )
    assert len(ds_filtered) == 1
    assert ds_filtered[0]["sample_id"] == "01_0001"


def test_real_si_metadata_bom_header_is_normalized():
    """Load the repository SI schema whose header contains a UTF-8 BOM."""
    metadata_file = (
        Path(__file__).resolve().parents[1]
        / "metadata"
        / "isharah500_si_train.csv"
    )
    dataset = IsharahDataset(
        metadata_file,
        root_dir=metadata_file.parents[1],
        missing_file_policy="error",
    )

    assert dataset.samples[0]["sample_id"] == "00_0001"
    assert dataset.samples[0]["keypoints_path"].endswith("00_0001.npy")


def test_collate_isharah_batch(tmp_path):
    """Verify batch collation and dynamic padding."""
    pytest.importorskip("torch")
    import torch

    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,معلم,مدرس,s1.npy\ns2,انا طالب,انا طالب,s2.npy\n",
        encoding="utf-8",
    )

    vocab = GlossVocabulary()
    vocab.build_vocab(["معلم", "انا", "طالب"])

    ds = IsharahDataset(
        metadata_csv=meta_file,
        root_dir=tmp_path,
        vocabulary=vocab,
    )

    batch = [ds[0], ds[1]]
    collated = collate_isharah_batch(batch)

    assert isinstance(collated["keypoints"], torch.Tensor)
    assert collated["keypoints"].shape[0] == 2
    assert collated["lengths"].tolist() == [30, 30]
    assert "token_ids" in collated
    assert collated["token_ids"].shape[0] == 2


def test_strict_dataset_rejects_missing_keypoints(tmp_path):
    """ML mode must reject missing files instead of creating placeholders."""
    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,معلم,مدرس,s1.npy\n",
        encoding="utf-8",
    )
    dataset = IsharahDataset(meta_file, root_dir=tmp_path, strict=True)

    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_validate_keypoints_contract():
    """Validate shape, dtype, length, and invalid values."""
    valid = np.zeros((4, 2, 63), dtype=np.float32)
    validate_keypoints(valid, expected_frames=4)

    with pytest.raises(ValueError):
        validate_keypoints(np.zeros((4, 63), dtype=np.float32))
    with pytest.raises(ValueError):
        validate_keypoints(np.zeros((4, 2, 63), dtype=np.float64))
    with pytest.raises(ValueError):
        validate_keypoints(valid, expected_frames=5)
    with pytest.raises(ValueError):
        validate_keypoints(np.full((4, 2, 63), np.inf, dtype=np.float32))


def test_preprocessing_config_round_trip():
    """Preprocessing settings can be serialized and reconstructed."""
    config = PreprocessingConfig(
        nan_strategy="zeros",
        flatten_hands=True,
        missing_file_policy="error",
    )
    assert PreprocessingConfig.from_dict(config.to_dict()) == config


def test_training_vocabulary_uses_only_selected_csv(tmp_path):
    """Vocabulary construction must not inspect dev or test annotations."""
    train_file = tmp_path / "train.csv"
    train_file.write_text(
        "sample_id,gloss,text,keypoints_path\ns1,الف,نص,a.npy\n",
        encoding="utf-8",
    )
    vocabulary = build_training_vocabulary(train_file)

    assert "الف" in vocabulary.token2id
    assert "باء" not in vocabulary.token2id
    assert vocabulary.oov_tokens("الف باء") == ["باء"]
    with pytest.raises(ValueError):
        vocabulary.encode("الف باء", oov_policy="error")


def test_ctc_collator_returns_concatenated_targets(tmp_path):
    """CTC batches contain flat targets and explicit input/target lengths."""
    pytest.importorskip("torch")
    import torch

    meta_file = tmp_path / "metadata.csv"
    meta_file.write_text(
        "sample_id,gloss,text,keypoints_path\n"
        "s1,معلم,مدرس,s1.npy\n"
        "s2,انا طالب,انا طالب,s2.npy\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "s1.npy", np.zeros((3, 2, 63), dtype=np.float32))
    np.save(tmp_path / "s2.npy", np.zeros((5, 2, 63), dtype=np.float32))

    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["معلم", "انا", "طالب"])
    dataset = IsharahDataset(
        meta_file,
        root_dir=tmp_path,
        vocabulary=vocabulary,
        missing_file_policy="error",
    )
    collated = collate_isharah_ctc_batch([dataset[0], dataset[1]])

    assert collated["keypoints"].shape == (2, 5, 2, 63)
    assert collated["masks"].shape == (2, 5, 2)
    assert collated["input_lengths"].tolist() == [3, 5]
    assert collated["target_lengths"].tolist() == [1, 2]
    assert collated["target_ids"].tolist() == (
        vocabulary.encode("معلم") + vocabulary.encode("انا طالب")
    )
    assert isinstance(collated["target_ids"], torch.Tensor)

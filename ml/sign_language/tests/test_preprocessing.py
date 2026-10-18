"""
Unit tests for data preprocessing and vocabulary modules in SIGNOVA.
"""

import numpy as np

from ml.sign_language.src.preprocessing import (
    SignVocabulary,
    interpolate_temporal_keypoints,
    normalize_keypoint_coordinates,
    resample_sequence,
)


def test_interpolate_temporal_keypoints():
    # Sequence with NaN values: (T, 2, 63)
    arr = np.ones((5, 2, 63), dtype=np.float32)
    arr[1, :, :] = np.nan
    result = interpolate_temporal_keypoints(arr)
    assert not np.isnan(result).any(), "NaN values must be completely interpolated"
    assert result.shape == arr.shape


def test_normalize_keypoint_coordinates():
    # (T, 2, 63) representing 2 hands, 21 joints, 3 coords
    coords = np.ones((10, 2, 63), dtype=np.float32) * 5.0
    normed = normalize_keypoint_coordinates(coords, reference_point_idx=0)
    assert normed.shape == (10, 2, 63)
    # Joint 0 should have coordinates subtracted (5 - 5 = 0)
    hand0 = normed[:, 0, :].reshape(-1, 21, 3)
    np.testing.assert_allclose(hand0[:, 0, :], 0.0, atol=1e-5)


def test_resample_sequence():
    coords = np.ones((50, 126), dtype=np.float32)
    # Resample to 30 frames
    resampled = resample_sequence(coords, target_frames=30)
    assert resampled.shape == (30, 126)

    # Resample to 70 frames
    resampled_up = resample_sequence(coords, target_frames=70)
    assert resampled_up.shape == (70, 126)


def test_sign_vocabulary():
    vocab = SignVocabulary(language="ArSL", add_ctc_blank=True)
    vocab.build_from_texts(["مرحبا بكم", "شكرا جزيلا"])

    # 5 special tokens: <blank>, <pad>, <unk>, <sos>, <eos> + 4 distinct words
    assert len(vocab) == 5 + 4
    assert vocab.token2id[SignVocabulary.BLANK_TOKEN] == 0
    assert vocab.token2id[SignVocabulary.PAD_TOKEN] == 1
    assert vocab.token2id[SignVocabulary.UNK_TOKEN] == 2

    # Test encoding
    tokens_str = "مرحبا غير_موجود"
    indices = vocab.encode(tokens_str)
    assert indices[0] == vocab.token2id["مرحبا"]
    assert indices[1] == vocab.token2id[SignVocabulary.UNK_TOKEN]

    # Test decoding
    decoded = vocab.decode(indices, remove_specials=True)
    assert decoded == "مرحبا"

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
    collate_isharah_batch,
    interpolate_keypoints,
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

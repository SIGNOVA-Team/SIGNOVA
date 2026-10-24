"""
Unit tests for DataLoader, dynamic sequence padding, and batch collation.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.sign_language.src.data_loader import (  # noqa: E402
    collate_sign_batch,
    create_dataloader,
    pad_keypoint_sequences,
    pad_token_sequences,
)


def test_pad_keypoint_sequences():
    seq1 = np.ones((10, 126), dtype=np.float32)
    seq2 = np.ones((25, 126), dtype=np.float32)
    seq3 = np.ones((15, 126), dtype=np.float32)

    padded, lengths, mask = pad_keypoint_sequences([seq1, seq2, seq3], pad_value=0.0)

    assert padded.shape == (3, 25, 126)
    assert (lengths == np.array([10, 25, 15])).all()
    assert mask.shape == (3, 25)
    assert mask[0, :10].all() and not mask[0, 10:].any()
    assert mask[1, :25].all()


def test_pad_token_sequences():
    tokens = [[1, 2, 3], [4, 5], [6, 7, 8, 9, 10]]
    padded, lengths = pad_token_sequences(tokens, pad_id=0)

    assert padded.shape == (3, 5)
    assert (lengths == np.array([3, 2, 5])).all()
    assert (padded[1, 2:] == 0).all()


def test_collate_sign_batch():
    mock_batch = [
        {
            "sample_id": "SAMPLE_001",
            "language": "ArSL",
            "keypoints": np.ones((12, 126), dtype=np.float32),
            "text": "مرحبا",
            "signer_id": "01",
            "token_ids": [5, 6],
        },
        {
            "sample_id": "SAMPLE_002",
            "language": "ArSL",
            "keypoints": np.ones((20, 126), dtype=np.float32),
            "text": "شكرا",
            "signer_id": "02",
            "token_ids": [7],
        },
    ]

    collated = collate_sign_batch(mock_batch)
    assert "keypoints" in collated
    assert "lengths" in collated
    assert "token_ids" in collated
    assert "token_lengths" in collated
    assert collated["keypoints"].shape == (2, 20, 126)
    assert isinstance(collated["keypoints"], torch.Tensor)
    assert (collated["lengths"] == torch.tensor([12, 20])).all()
    assert (collated["token_lengths"] == torch.tensor([2, 1])).all()


def test_create_dataloader_execution():
    class MockDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "sample_id": f"S_{idx}",
                "language": "ArSL",
                "keypoints": np.ones((10 + idx, 126), dtype=np.float32),
                "text": "test",
                "signer_id": "01",
                "token_ids": [1, 2],
            }

    ds = MockDataset()
    loader = create_dataloader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["keypoints"].shape[0] == 2

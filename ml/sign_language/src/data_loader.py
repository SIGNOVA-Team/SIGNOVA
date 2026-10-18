"""
SIGNOVA — DataLoader Module with Dynamic Sequence Padding and Collation.

Provides batch collation for variable-length temporal sequences and token sequences.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DataLoader = Any  # type: ignore


def pad_keypoint_sequences(
    sequences: list[np.ndarray],
    pad_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pad a list of 2D arrays (T_i, D) to (B, T_max, D) and construct a boolean mask.

    Returns:
        padded_array: (B, T_max, D)
        lengths: (B,)
        mask: (B, T_max) True where data is valid, False where padded.
    """
    batch_size = len(sequences)
    lengths = np.array([len(s) for s in sequences], dtype=np.int64)
    max_len = int(np.max(lengths)) if batch_size > 0 else 0
    dim = sequences[0].shape[-1] if batch_size > 0 else 0

    padded = np.full((batch_size, max_len, dim), pad_value, dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=bool)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        padded[i, :seq_len] = seq
        mask[i, :seq_len] = True

    return padded, lengths, mask


def pad_token_sequences(
    token_lists: list[list[int]],
    pad_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad a list of token ID lists to (B, L_max).

    Returns:
        padded_tokens: (B, L_max)
        token_lengths: (B,)
    """
    batch_size = len(token_lists)
    lengths = np.array([len(t) for t in token_lists], dtype=np.int64)
    max_len = int(np.max(lengths)) if batch_size > 0 else 0

    padded = np.full((batch_size, max_len), pad_id, dtype=np.int64)
    for i, tokens in enumerate(token_lists):
        padded[i, : len(tokens)] = tokens

    return padded, lengths


def collate_sign_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function to prepare PyTorch/NumPy batches for sequence modeling.
    """
    keypoints = [item["keypoints"] for item in batch]
    padded_kp, lengths, mask = pad_keypoint_sequences(keypoints)

    out: dict[str, Any] = {
        "sample_ids": [item["sample_id"] for item in batch],
        "language": [item["language"] for item in batch],
        "keypoints": torch.from_numpy(padded_kp).float() if HAS_TORCH else padded_kp,
        "lengths": torch.from_numpy(lengths).long() if HAS_TORCH else lengths,
        "masks": torch.from_numpy(mask).bool() if HAS_TORCH else mask,
        "texts": [item["text"] for item in batch],
        "signer_ids": [item["signer_id"] for item in batch],
    }

    if "token_ids" in batch[0]:
        token_lists = [item["token_ids"] for item in batch]
        padded_tok, tok_lengths = pad_token_sequences(token_lists)
        out["token_ids"] = torch.from_numpy(padded_tok).long() if HAS_TORCH else padded_tok
        out["token_lengths"] = torch.from_numpy(tok_lengths).long() if HAS_TORCH else tok_lengths

    return out


def create_dataloader(
    dataset: Any,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create a configured DataLoader."""
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required to instantiate DataLoader.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_sign_batch,
    )

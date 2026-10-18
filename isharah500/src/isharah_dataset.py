"""
Isharah-500 ML Dataset and DataLoader Module for PyTorch / ML Pipelines.

This module provides an ML-ready Dataset and collator for Isharah-500:
- Resolves keypoint paths from SI and US metadata CSVs.
- Handles missing hand detections (NaN coordinates) via temporal interpolation or zero-filling.
- Generates binary detection masks (T, 2) indicating hand visibility per frame.
- Provides quality filtering (exclude_flagged=True) to omit low-quality samples.
- Tokenizes sign glosses for CTC and sequence-to-sequence loss functions.
- Dynamically collates batches with variable sequence lengths.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    class Dataset:  # type: ignore
        """Fallback Dataset class when torch is not installed."""

        def __getitem__(self, index: int) -> Any:
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError


def interpolate_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate NaN keypoints along the temporal axis.

    Args:
        keypoints: Array of shape (T, 2, 63) or (T, num_hands, num_coords).

    Returns:
        Interpolated array of the same shape with NaNs filled.
    """
    out = np.copy(keypoints)
    num_frames, num_hands, num_coords = out.shape

    if num_frames == 0:
        return out

    for h in range(num_hands):
        for c in range(num_coords):
            series = out[:, h, c]
            valid_mask = ~np.isnan(series)

            if not np.any(valid_mask):
                # All values are NaN, set to 0.0
                out[:, h, c] = 0.0
                continue

            if np.all(valid_mask):
                continue

            # Linear interpolation across time
            indices = np.arange(num_frames)
            valid_indices = indices[valid_mask]
            valid_values = series[valid_mask]

            # np.interp extrapolates edge values using the left and right boundary values
            out[:, h, c] = np.interp(indices, valid_indices, valid_values)

    return out


class GlossVocabulary:
    """Vocabulary mapping for sign language gloss tokens."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    BLANK_TOKEN = "<blank>"  # Reserved for CTC loss

    def __init__(self, add_ctc_blank: bool = True):
        self.add_ctc_blank = add_ctc_blank
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        if self.add_ctc_blank:
            special_tokens.insert(0, self.BLANK_TOKEN)

        for token in special_tokens:
            self._add_token(token)

    def _add_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def build_vocab(self, gloss_sequences: list[str]) -> None:
        """Build vocabulary from a list of gloss strings."""
        for seq in gloss_sequences:
            tokens = seq.strip().split()
            for t in tokens:
                self._add_token(t)

    def encode(self, gloss_str: str, add_sos_eos: bool = False) -> list[int]:
        """Convert a gloss string to token IDs."""
        tokens = gloss_str.strip().split()
        unk_id = self.token2id[self.UNK_TOKEN]
        ids = [self.token2id.get(t, unk_id) for t in tokens]
        if add_sos_eos:
            ids = [self.token2id[self.SOS_TOKEN]] + ids + [self.token2id[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: list[int], remove_specials: bool = True) -> str:
        """Convert token IDs back to a gloss string."""
        specials = {
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.SOS_TOKEN,
            self.EOS_TOKEN,
            self.BLANK_TOKEN,
        }
        words = []
        for tid in token_ids:
            tok = self.id2token.get(tid, self.UNK_TOKEN)
            if remove_specials and tok in specials:
                continue
            words.append(tok)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.token2id)


class IsharahDataset(Dataset):
    """
    PyTorch Dataset for Isharah-500 Hand Keypoint Sequences.

    Args:
        metadata_csv: Path to split CSV (e.g., metadata/isharah500_si_train.csv).
        root_dir: Project root directory to resolve relative keypoint paths.
        exclude_flagged: If True, filters out samples marked for review in review_csv.
        review_csv: Path to reports/isharah500_review_required.csv.
        nan_strategy: 'interpolate' (smooth bridging), 'zeros' (fill with 0), or 'none'.
        flatten_hands: If True, reshapes (T, 2, 63) to (T, 126).
        vocabulary: Optional GlossVocabulary instance for gloss tokenization.
        transform: Optional callable applied to keypoints array.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        root_dir: str | Path | None = None,
        exclude_flagged: bool = False,
        review_csv: str | Path | None = None,
        nan_strategy: str = "interpolate",
        flatten_hands: bool = False,
        vocabulary: GlossVocabulary | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.root_dir = Path(root_dir) if root_dir else self.metadata_csv.resolve().parents[2]
        self.exclude_flagged = exclude_flagged
        self.nan_strategy = nan_strategy
        self.flatten_hands = flatten_hands
        self.vocabulary = vocabulary
        self.transform = transform

        # Determine flagged samples if filtering requested
        flagged_ids = set()
        if self.exclude_flagged:
            if review_csv is None:
                review_csv = self.root_dir / "isharah500/reports/isharah500_review_required.csv"
            review_path = Path(review_csv)
            if review_path.exists():
                with open(review_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("review_required", "").lower() == "true":
                            flagged_ids.add(row["sample_id"])

        # Load samples from metadata
        self.samples: list[dict[str, str]] = []
        with open(self.metadata_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id", "")
                if self.exclude_flagged and sid in flagged_ids:
                    continue
                self.samples.append(row)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples[index]
        sample_id = row["sample_id"]
        rel_path = row["keypoints_path"]
        gloss = row.get("gloss", "")
        text = row.get("text", "")

        kp_path = self.root_dir / rel_path

        # Load array if available, otherwise return empty synthetic if file missing
        if kp_path.exists():
            keypoints = np.load(kp_path).astype(np.float32)
        else:
            # File placeholder for environments without full binary arrays
            keypoints = np.full((30, 2, 63), np.nan, dtype=np.float32)

        # Compute detection mask (T, 2): True where landmarks were detected
        # A hand is detected if at least one coordinate is not NaN
        valid_mask = ~np.all(np.isnan(keypoints), axis=-1)  # Shape (T, 2)

        # Handle NaNs according to selected strategy
        if self.nan_strategy == "interpolate":
            keypoints = interpolate_keypoints(keypoints)
        elif self.nan_strategy == "zeros":
            keypoints = np.nan_to_num(keypoints, nan=0.0)

        # Apply optional custom transform
        if self.transform is not None:
            keypoints = self.transform(keypoints)

        # Optionally flatten hands: (T, 2, 63) -> (T, 126)
        if self.flatten_hands:
            keypoints = keypoints.reshape(len(keypoints), -1)

        item: dict[str, Any] = {
            "sample_id": sample_id,
            "keypoints": torch.from_numpy(keypoints).float() if HAS_TORCH else keypoints,
            "mask": torch.from_numpy(valid_mask).bool() if HAS_TORCH else valid_mask,
            "length": len(keypoints),
            "gloss": gloss,
            "text": text,
        }

        if self.vocabulary is not None:
            token_ids = self.vocabulary.encode(gloss)
            item["token_ids"] = (
                torch.tensor(token_ids, dtype=torch.long) if HAS_TORCH else token_ids
            )
            item["token_length"] = len(token_ids)

        return item


def collate_isharah_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Collate function to pad variable-length keypoints and tokens in a batch.

    Returns:
        Dictionary containing padded keypoints (B, T_max, ...), lengths (B,),
        padded masks (B, T_max, 2), sample_ids, glosses, texts,
        and optionally padded tokens (B, L_max) and token lengths (B,).
    """
    if not HAS_TORCH:
        raise RuntimeError("torch must be installed to use collate_isharah_batch")

    batch_size = len(batch)
    lengths = [item["length"] for item in batch]
    max_len = max(lengths) if lengths else 0

    sample_kp = batch[0]["keypoints"]
    kp_dims = sample_kp.shape[1:]  # e.g., (2, 63) or (126,)

    # Allocate padded keypoints & mask
    padded_kp = torch.zeros((batch_size, max_len, *kp_dims), dtype=torch.float32)
    padded_mask = torch.zeros((batch_size, max_len, 2), dtype=torch.bool)

    sample_ids = []
    glosses = []
    texts = []

    has_tokens = "token_ids" in batch[0]
    token_lengths = []

    for i, item in enumerate(batch):
        t_len = item["length"]
        padded_kp[i, :t_len] = item["keypoints"]
        padded_mask[i, :t_len] = item["mask"]
        sample_ids.append(item["sample_id"])
        glosses.append(item["gloss"])
        texts.append(item["text"])
        if has_tokens:
            token_lengths.append(item["token_length"])

    out: dict[str, Any] = {
        "keypoints": padded_kp,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "masks": padded_mask,
        "sample_ids": sample_ids,
        "glosses": glosses,
        "texts": texts,
    }

    if has_tokens:
        max_tok_len = max(token_lengths) if token_lengths else 0
        pad_id = 0
        padded_tokens = torch.full((batch_size, max_tok_len), pad_id, dtype=torch.long)
        for i, item in enumerate(batch):
            tok_len = item["token_length"]
            padded_tokens[i, :tok_len] = item["token_ids"]
        out["token_ids"] = padded_tokens
        out["token_lengths"] = torch.tensor(token_lengths, dtype=torch.long)

    return out

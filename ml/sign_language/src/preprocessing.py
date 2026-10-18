"""
SIGNOVA — Core Preprocessing & Feature Engineering Module for Sign Language.

Handles:
- Temporal normalization and sequence resampling.
- Keypoint coordinate normalization and temporal NaN-interpolation.
- Bounding box spatial cropping and coordinate mapping.
- Language-specific tokenization and vocabulary management.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def interpolate_temporal_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """
    Interpolate NaN keypoints along the temporal axis with linear interpolation.

    Args:
        keypoints: Array of shape (T, num_coords) or (T, num_hands, coords_per_hand).

    Returns:
        Interpolated array with NaNs imputed.
    """
    out = np.copy(keypoints)
    original_shape = out.shape

    if len(original_shape) == 3:
        num_frames, dim1, dim2 = original_shape
        flat = out.reshape(num_frames, dim1 * dim2)
    elif len(original_shape) == 2:
        num_frames, _ = original_shape
        flat = out
    else:
        return out

    if num_frames == 0:
        return out

    num_features = flat.shape[1]
    for f in range(num_features):
        series = flat[:, f]
        valid_mask = ~np.isnan(series)

        if not np.any(valid_mask):
            flat[:, f] = 0.0
            continue

        if np.all(valid_mask):
            continue

        indices = np.arange(num_frames)
        valid_indices = indices[valid_mask]
        valid_values = series[valid_mask]
        flat[:, f] = np.interp(indices, valid_indices, valid_values)

    return flat.reshape(original_shape)


def normalize_keypoint_coordinates(
    keypoints: np.ndarray,
    reference_point_idx: int = 0,
) -> np.ndarray:
    """
    Normalize keypoints relative to a reference joint (e.g., wrist) and scale by bounding span.

    Args:
        keypoints: (T, 2, 63) or (T, 126) array representing hand landmarks.
        reference_point_idx: Joint index used as origin.

    Returns:
        Normalized keypoints array.
    """
    norm_kp = np.copy(keypoints)
    # If 3D array (T, 2, 63) where 63 = 21 joints * 3 (x,y,z)
    if len(norm_kp.shape) == 3 and norm_kp.shape[2] == 63:
        for hand_idx in range(norm_kp.shape[1]):
            hand_data = norm_kp[:, hand_idx, :].reshape(-1, 21, 3)
            # Subtract wrist (joint 0)
            wrist = hand_data[:, reference_point_idx : reference_point_idx + 1, :]
            # Only subtract where wrist is valid
            valid_wrist = ~np.isnan(wrist)
            hand_data = np.where(valid_wrist, hand_data - wrist, hand_data)
            norm_kp[:, hand_idx, :] = hand_data.reshape(-1, 63)
    return norm_kp


def resample_sequence(
    sequence: np.ndarray,
    target_frames: int,
) -> np.ndarray:
    """
    Resample a temporal sequence to a fixed frame length via linear interpolation.

    Args:
        sequence: Array of shape (T, D).
        target_frames: Desired sequence length.

    Returns:
        Resampled array of shape (target_frames, D).
    """
    t_orig, feat_dim = sequence.shape
    if t_orig == target_frames:
        return sequence

    orig_indices = np.linspace(0, 1, t_orig)
    target_indices = np.linspace(0, 1, target_frames)

    resampled = np.zeros((target_frames, feat_dim), dtype=sequence.dtype)
    for d in range(feat_dim):
        resampled[:, d] = np.interp(target_indices, orig_indices, sequence[:, d])

    return resampled


class SignVocabulary:
    """
    Language-specific vocabulary manager for sign language annotations.
    Separates ArSL (Arabic Sign Language) and ASL (American Sign Language).
    """

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    SOS_TOKEN = "<sos>"
    EOS_TOKEN = "<eos>"
    BLANK_TOKEN = "<blank>"  # CTC blank token (id 0)

    def __init__(self, language: str, add_ctc_blank: bool = True):
        self.language = language
        self.add_ctc_blank = add_ctc_blank
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}

        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        if self.add_ctc_blank:
            special_tokens.insert(0, self.BLANK_TOKEN)

        for tok in special_tokens:
            self._register_token(tok)

    def _register_token(self, token: str) -> int:
        if token not in self.token2id:
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
            return idx
        return self.token2id[token]

    def build_from_texts(self, texts: list[str]) -> None:
        """Build vocabulary from a list of sentence or gloss strings."""
        for text in texts:
            tokens = text.strip().split()
            for t in tokens:
                self._register_token(t)

    def encode(self, text: str, add_sos_eos: bool = False) -> list[int]:
        """Convert a text or gloss string into token IDs."""
        tokens = text.strip().split()
        unk_id = self.token2id[self.UNK_TOKEN]
        ids = [self.token2id.get(tok, unk_id) for tok in tokens]
        if add_sos_eos:
            ids = [self.token2id[self.SOS_TOKEN]] + ids + [self.token2id[self.EOS_TOKEN]]
        return ids

    def decode(self, token_ids: list[int], remove_specials: bool = True) -> str:
        """Decode a list of token IDs into text."""
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

    def save(self, file_path: str | Path) -> None:
        """Save vocabulary mapping to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "language": self.language,
            "add_ctc_blank": self.add_ctc_blank,
            "token2id": self.token2id,
            "id2token": {str(k): v for k, v in self.id2token.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> SignVocabulary:
        """Load vocabulary mapping from JSON."""
        path = Path(file_path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(language=data["language"], add_ctc_blank=data.get("add_ctc_blank", True))
        vocab.token2id = data["token2id"]
        vocab.id2token = {int(k): v for k, v in data["id2token"].items()}
        return vocab

    def __len__(self) -> int:
        return len(self.token2id)

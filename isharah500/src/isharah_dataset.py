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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from isharah_features import (
    FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
    FEATURE_REPRESENTATION_RAW_63,
    construct_feature_representation,
    output_features_per_hand,
)

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


@dataclass(frozen=True)
class PreprocessingConfig:
    """Serializable preprocessing contract shared by training and inference."""

    nan_strategy: str = "interpolate"
    flatten_hands: bool = False
    missing_file_policy: str = "error"
    expected_num_hands: int = 2
    expected_features_per_hand: int = 63
    reject_infinite_values: bool = True
    coordinate_normalization: str = "none"
    normalization_epsilon: float = 1e-6
    feature_representation: str = FEATURE_REPRESENTATION_RAW_63
    global_coordinate_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.nan_strategy not in {"interpolate", "zeros", "none"}:
            raise ValueError(f"Unsupported nan_strategy: {self.nan_strategy}")
        if self.missing_file_policy not in {"error", "placeholder"}:
            raise ValueError(f"Unsupported missing_file_policy: {self.missing_file_policy}")
        if self.expected_num_hands <= 0 or self.expected_features_per_hand <= 0:
            raise ValueError("Expected keypoint dimensions must be positive")
        if self.coordinate_normalization not in {"none", "hand_centric"}:
            raise ValueError(
                f"Unsupported coordinate_normalization: {self.coordinate_normalization}"
            )
        if self.normalization_epsilon <= 0:
            raise ValueError("normalization_epsilon must be positive")
        if self.feature_representation not in {
            FEATURE_REPRESENTATION_RAW_63,
            FEATURE_REPRESENTATION_GLOBAL_POSITION_V1,
        }:
            raise ValueError(f"Unsupported feature_representation: {self.feature_representation}")
        if self.global_coordinate_scale <= 0 or not np.isfinite(self.global_coordinate_scale):
            raise ValueError("global_coordinate_scale must be finite and positive")

    @property
    def output_features_per_hand(self) -> int:
        """Feature width emitted to the model after preprocessing."""
        return output_features_per_hand(self.feature_representation)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible preprocessing settings."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> PreprocessingConfig:
        """Reconstruct a preprocessing contract from serialized settings."""
        return cls(**values)


def validate_keypoints(
    keypoints: np.ndarray,
    *,
    expected_frames: int | None = None,
    config: PreprocessingConfig | None = None,
) -> None:
    """Raise ``ValueError`` when keypoints violate the ML data contract."""
    config = config or PreprocessingConfig()
    expected_shape = (
        config.expected_num_hands,
        config.expected_features_per_hand,
    )

    if not isinstance(keypoints, np.ndarray):
        raise ValueError("Keypoints must be a NumPy array")
    if keypoints.ndim != 3 or keypoints.shape[1:] != expected_shape:
        raise ValueError(
            f"Expected keypoints shape (T, {expected_shape[0]}, "
            f"{expected_shape[1]}), got {keypoints.shape}"
        )
    if keypoints.shape[0] <= 0:
        raise ValueError("Keypoint sequences must contain at least one frame")
    if expected_frames is not None and keypoints.shape[0] != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames, got {keypoints.shape[0]}")
    if keypoints.dtype != np.float32:
        raise ValueError(f"Expected dtype float32, got {keypoints.dtype}")
    if config.reject_infinite_values and not np.all(np.isfinite(keypoints) | np.isnan(keypoints)):
        raise ValueError("Keypoints contain infinite or otherwise invalid values")


def build_training_vocabulary(
    metadata_csv: str | Path,
    *,
    add_ctc_blank: bool = True,
) -> GlossVocabulary:
    """Build a vocabulary from one selected protocol's training CSV only."""
    metadata_csv = Path(metadata_csv)
    with metadata_csv.open(encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        glosses = [row.get("gloss", "") for row in rows]
    if any(not gloss.strip() for gloss in glosses):
        raise ValueError(f"Training metadata contains an empty gloss: {metadata_csv}")
    vocabulary = GlossVocabulary(add_ctc_blank=add_ctc_blank)
    vocabulary.build_vocab(glosses)
    return vocabulary


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


def normalize_hand_centric(
    keypoints: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize each hand around its wrist using a robust palm scale.

    Each hand is reshaped into 21 landmarks with XYZ coordinates. For every
    frame, landmark 0 (wrist) is subtracted from all landmarks. The scale is
    the median, over valid frames, of the 3-D distance from the wrist
    (landmark 0) to the middle-finger MCP (landmark 9). The one scalar is
    shared across the sequence for that hand, preserving relative temporal
    movement while removing camera translation and hand-size variation.

    Missing or interpolated frames retain their caller-provided mask. Invalid
    reference measurements, zero-sized hands, and NaNs are safely converted
    to finite values without division by zero. The output shape is unchanged.
    """
    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 3:
        raise ValueError("keypoints must have shape (T, hands, features)")
    if keypoints.shape[-1] % 3 != 0:
        raise ValueError("hand features must contain XYZ triplets")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if valid_mask is not None:
        if valid_mask.shape != keypoints.shape[:2]:
            raise ValueError("valid_mask must have shape (T, hands)")

    output = np.asarray(keypoints, dtype=np.float32).copy()
    landmark_count = output.shape[-1] // 3
    if landmark_count <= 9:
        raise ValueError("hand keypoints must include landmark 9")

    for hand_index in range(output.shape[1]):
        landmarks = output[:, hand_index].reshape(output.shape[0], landmark_count, 3)
        wrist = landmarks[:, 0, :]
        reference = landmarks[:, 9, :]
        scale_values = np.linalg.norm(reference - wrist, axis=-1)
        usable = np.isfinite(scale_values) & (scale_values > epsilon)
        if valid_mask is not None:
            usable &= valid_mask[:, hand_index]
        scale = float(np.median(scale_values[usable])) if np.any(usable) else 1.0
        if not np.isfinite(scale) or scale <= epsilon:
            scale = 1.0

        normalized = (landmarks - wrist[:, None, :]) / scale
        output[:, hand_index] = np.nan_to_num(
            normalized,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).reshape(output.shape[0], -1)

    return output


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

    def oov_tokens(self, gloss_str: str) -> list[str]:
        """Return gloss tokens absent from this vocabulary."""
        return [token for token in gloss_str.strip().split() if token not in self.token2id]

    def encode(
        self,
        gloss_str: str,
        add_sos_eos: bool = False,
        oov_policy: str = "unk",
    ) -> list[int]:
        """Convert a gloss string to token IDs."""
        if oov_policy not in {"unk", "error"}:
            raise ValueError(f"Unsupported oov_policy: {oov_policy}")
        tokens = gloss_str.strip().split()
        unknown_tokens = self.oov_tokens(gloss_str)
        if unknown_tokens and oov_policy == "error":
            raise ValueError(f"Gloss contains OOV tokens: {unknown_tokens}")
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
        missing_file_policy: 'error' for ML use or 'placeholder' for legacy tests.
        strict: If True, defaults missing_file_policy to 'error'.
        preprocessing_config: Optional serialized preprocessing contract.
    """

    def __init__(
        self,
        metadata_csv: str | Path,
        root_dir: str | Path | None = None,
        exclude_flagged: bool = False,
        review_csv: str | Path | None = None,
        nan_strategy: str | None = None,
        flatten_hands: bool | None = None,
        vocabulary: GlossVocabulary | None = None,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        missing_file_policy: str | None = None,
        expected_frames: int | None = None,
        oov_policy: str = "unk",
        strict: bool = False,
        preprocessing_config: PreprocessingConfig | dict[str, Any] | None = None,
    ):
        self.metadata_csv = Path(metadata_csv)
        self.root_dir = Path(root_dir) if root_dir else self.metadata_csv.resolve().parents[2]
        self.exclude_flagged = exclude_flagged
        if preprocessing_config is None:
            default_policy = "error" if strict else "placeholder"
            config = PreprocessingConfig(
                nan_strategy=nan_strategy or "interpolate",
                flatten_hands=bool(flatten_hands) if flatten_hands is not None else False,
                missing_file_policy=missing_file_policy or default_policy,
            )
        elif isinstance(preprocessing_config, dict):
            config = PreprocessingConfig.from_dict(preprocessing_config)
        else:
            config = preprocessing_config
        if nan_strategy is not None or flatten_hands is not None or missing_file_policy is not None:
            config = PreprocessingConfig(
                **{
                    **config.to_dict(),
                    "nan_strategy": nan_strategy or config.nan_strategy,
                    "flatten_hands": (
                        flatten_hands if flatten_hands is not None else config.flatten_hands
                    ),
                    "missing_file_policy": (
                        missing_file_policy
                        if missing_file_policy is not None
                        else config.missing_file_policy
                    ),
                }
            )
        if strict and config.missing_file_policy != "error":
            raise ValueError("strict=True requires missing_file_policy='error'")
        self.preprocessing_config = config
        self.nan_strategy = config.nan_strategy
        self.flatten_hands = config.flatten_hands
        self.missing_file_policy = config.missing_file_policy
        self.expected_frames = expected_frames
        self.oov_policy = oov_policy
        self.vocabulary = vocabulary
        self.transform = transform

        # Determine flagged samples if filtering requested
        flagged_ids = set()
        if self.exclude_flagged:
            if review_csv is None:
                review_csv = self.root_dir / "isharah500/reports/isharah500_review_required.csv"
            review_path = Path(review_csv)
            if review_path.exists():
                with open(review_path, encoding="utf-8-sig") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row.get("review_required", "").lower() == "true":
                            flagged_ids.add(row["sample_id"])

        # Load samples from metadata
        self.samples: list[dict[str, str]] = []
        # The checked-in Isharah metadata uses UTF-8 with a BOM.
        with open(self.metadata_csv, encoding="utf-8-sig", newline="") as f:
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

        # Placeholder mode is retained only for backwards-compatible preprocessing tests.
        if kp_path.exists():
            try:
                keypoints = np.load(kp_path, allow_pickle=False)
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"Unable to load keypoints: {kp_path}") from exc
            validate_keypoints(
                keypoints,
                expected_frames=self.expected_frames,
                config=self.preprocessing_config,
            )
            keypoints = keypoints.astype(np.float32, copy=False)
        else:
            if self.missing_file_policy == "error":
                raise FileNotFoundError(f"Keypoint file does not exist: {kp_path}")
            keypoints = np.full((30, 2, 63), np.nan, dtype=np.float32)

        # Compute detection mask (T, 2): True where landmarks were detected
        # A hand is detected if at least one coordinate is not NaN
        valid_mask = ~np.all(np.isnan(keypoints), axis=-1)  # Shape (T, 2)

        # Handle NaNs according to selected strategy
        if self.nan_strategy == "interpolate":
            keypoints = interpolate_keypoints(keypoints)
        elif self.nan_strategy == "zeros":
            keypoints = np.nan_to_num(keypoints, nan=0.0)

        global_keypoints = keypoints.copy()

        if self.preprocessing_config.coordinate_normalization == "hand_centric":
            keypoints = normalize_hand_centric(
                keypoints,
                valid_mask,
                epsilon=self.preprocessing_config.normalization_epsilon,
            )

        keypoints = construct_feature_representation(
            keypoints,
            valid_mask,
            feature_representation=self.preprocessing_config.feature_representation,
            global_coordinate_scale=self.preprocessing_config.global_coordinate_scale,
            global_keypoints=global_keypoints,
        )

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
            oov_tokens = self.vocabulary.oov_tokens(gloss)
            token_ids = self.vocabulary.encode(gloss, oov_policy=self.oov_policy)
            item["token_ids"] = (
                torch.tensor(token_ids, dtype=torch.long) if HAS_TORCH else token_ids
            )
            item["token_length"] = len(token_ids)
            item["oov_tokens"] = oov_tokens

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


def collate_isharah_ctc_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length samples into the unpadded CTC target format."""
    if not HAS_TORCH:
        raise RuntimeError("torch must be installed to use collate_isharah_ctc_batch")
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    if any("token_ids" not in item for item in batch):
        raise ValueError("CTC collation requires a vocabulary on every dataset item")

    padded = collate_isharah_batch(batch)
    target_lengths = padded["token_lengths"]
    targets = [item["token_ids"][: item["token_length"]] for item in batch]
    target_ids = torch.cat(targets, dim=0) if targets else torch.empty(0, dtype=torch.long)

    padded["target_ids"] = target_ids
    padded["target_lengths"] = target_lengths
    padded["input_lengths"] = padded["lengths"]
    return padded

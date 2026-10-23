"""BiGRU + CTC baseline for Isharah-500 continuous sign recognition."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from isharah_dataset import GlossVocabulary
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class BiGRUCTCConfig:
    """Small model configuration suitable for a 4GB GPU."""

    num_hands: int = 2
    features_per_hand: int = 63
    projection_size: int = 128
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    zero_infinity: bool = True

    def __post_init__(self) -> None:
        if self.num_hands <= 0 or self.features_per_hand <= 0:
            raise ValueError("Input dimensions must be positive")
        if self.projection_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def input_size(self) -> int:
        return self.num_hands * self.features_per_hand

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-friendly configuration dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> BiGRUCTCConfig:
        """Reconstruct a model configuration from a dictionary."""
        return cls(**values)


def _validate_batch_inputs(
    keypoints: Tensor,
    input_lengths: Tensor,
    hand_detection_masks: Tensor | None,
    config: BiGRUCTCConfig,
) -> None:
    if keypoints.ndim != 4:
        raise ValueError(
            f"Expected keypoints with shape (B, T, {config.num_hands}, "
            f"{config.features_per_hand}), got {tuple(keypoints.shape)}"
        )
    if tuple(keypoints.shape[2:]) != (config.num_hands, config.features_per_hand):
        raise ValueError(
            f"Expected hand-keypoint shape (B, T, {config.num_hands}, "
            f"{config.features_per_hand}), got {tuple(keypoints.shape)}"
        )
    if input_lengths.ndim != 1 or input_lengths.shape[0] != keypoints.shape[0]:
        raise ValueError("input_lengths must have one value per batch item")
    if input_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("input_lengths must be an integer tensor")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > keypoints.shape[1]):
        raise ValueError("input_lengths must be in the range [1, T]")
    if not torch.isfinite(keypoints).all():
        raise ValueError("Model input contains NaN or infinite keypoints")
    if hand_detection_masks is not None:
        expected = (keypoints.shape[0], keypoints.shape[1], config.num_hands)
        if tuple(hand_detection_masks.shape) != expected:
            raise ValueError(
                f"Expected hand masks with shape {expected}, "
                f"got {tuple(hand_detection_masks.shape)}"
            )


class BiGRUCTC(nn.Module):
    """Length-aware BiGRU encoder with a CTC vocabulary projection."""

    def __init__(
        self,
        vocabulary_size: int,
        config: BiGRUCTCConfig | None = None,
    ) -> None:
        super().__init__()
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive")
        self.config = config or BiGRUCTCConfig()
        self.vocabulary_size = vocabulary_size

        self.input_projection = nn.Sequential(
            nn.Linear(self.config.input_size, self.config.projection_size),
            nn.LayerNorm(self.config.projection_size),
            nn.ReLU(),
        )
        self.encoder = nn.GRU(
            input_size=self.config.projection_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=(self.config.dropout if self.config.num_layers > 1 else 0.0),
        )
        self.output_dropout = nn.Dropout(self.config.dropout)
        self.classifier = nn.Linear(self.config.hidden_size * 2, vocabulary_size)

    def _prepare_features(
        self,
        keypoints: Tensor,
        hand_detection_masks: Tensor | None,
    ) -> Tensor:
        if hand_detection_masks is not None:
            hand_mask = hand_detection_masks.to(dtype=keypoints.dtype).unsqueeze(-1)
            keypoints = keypoints * hand_mask
        return keypoints.reshape(keypoints.shape[0], keypoints.shape[1], -1)

    def forward(
        self,
        keypoints: Tensor,
        input_lengths: Tensor,
        hand_detection_masks: Tensor | None = None,
    ) -> Tensor:
        """Return raw CTC logits with shape ``(B, T, vocabulary_size)``."""
        _validate_batch_inputs(
            keypoints,
            input_lengths,
            hand_detection_masks,
            self.config,
        )
        features = self._prepare_features(keypoints, hand_detection_masks)
        projected = self.input_projection(features)
        packed = pack_padded_sequence(
            projected,
            input_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=keypoints.shape[1],
        )
        logits = self.classifier(self.output_dropout(encoded))
        time_indices = torch.arange(
            keypoints.shape[1],
            device=keypoints.device,
        ).unsqueeze(0)
        valid_time = time_indices < input_lengths.to(device=keypoints.device).unsqueeze(1)
        return logits.masked_fill(~valid_time.unsqueeze(-1), 0.0)

    def ctc_loss(
        self,
        logits: Tensor,
        input_lengths: Tensor,
        target_ids: Tensor,
        target_lengths: Tensor,
        blank_id: int,
        *,
        target_length_weighting: str = "none",
        target_length_weight_cap: float = 2.0,
    ) -> Tensor:
        """Compute CTC loss from raw batch-first logits.

        The default ``target_length_weighting='none'`` preserves the original
        reduced CTC loss. Optional ``linear`` or ``sqrt`` weighting applies a
        capped, mean-normalized weight to unreduced per-sample losses.
        """
        return ctc_loss_from_logits(
            logits=logits,
            input_lengths=input_lengths,
            target_ids=target_ids,
            target_lengths=target_lengths,
            blank_id=blank_id,
            zero_infinity=self.config.zero_infinity,
            target_length_weighting=target_length_weighting,
            target_length_weight_cap=target_length_weight_cap,
        )


def ctc_loss_from_logits(
    logits: Tensor,
    input_lengths: Tensor,
    target_ids: Tensor,
    target_lengths: Tensor,
    blank_id: int,
    *,
    zero_infinity: bool = True,
    target_length_weighting: str = "none",
    target_length_weight_cap: float = 2.0,
) -> Tensor:
    """Compute CTC loss for logits shaped ``(B, T, C)``.

    For weighted modes, each sample's unreduced CTC loss is first divided by
    its target length, matching PyTorch ``reduction="mean"`` semantics. The
    normalized loss is then multiplied by a capped target-length weight
    relative to the current batch mean. Weights are normalized to mean one so
    the loss scale stays comparable.
    ``linear`` uses power 1.0 and ``sqrt`` uses power 0.5. The cap applies to
    the raw relative weight before mean normalization.
    """
    if target_length_weighting not in {"none", "linear", "sqrt"}:
        raise ValueError("target_length_weighting must be one of: none, linear, sqrt")
    if target_length_weight_cap <= 0:
        raise ValueError("target_length_weight_cap must be positive")
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, C)")
    if input_lengths.ndim != 1 or target_lengths.ndim != 1:
        raise ValueError("input_lengths and target_lengths must be one-dimensional")
    if input_lengths.shape[0] != logits.shape[0]:
        raise ValueError("input_lengths must match the logits batch size")
    if target_lengths.shape[0] != logits.shape[0]:
        raise ValueError("target_lengths must match the logits batch size")
    if int(target_lengths.sum().item()) != target_ids.numel():
        raise ValueError("target_lengths do not match the number of target IDs")
    if blank_id < 0 or blank_id >= logits.shape[-1]:
        raise ValueError("blank_id is outside the logits vocabulary")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > logits.shape[1]):
        raise ValueError("input_lengths are outside the logits time dimension")
    if torch.any(target_lengths < 0):
        raise ValueError("target_lengths cannot be negative")

    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    ctc_arguments = {
        "targets": target_ids,
        "input_lengths": input_lengths.to(dtype=torch.long),
        "target_lengths": target_lengths.to(dtype=torch.long),
        "blank": blank_id,
        "zero_infinity": zero_infinity,
    }
    if target_length_weighting == "none":
        return nn.functional.ctc_loss(log_probs, reduction="mean", **ctc_arguments)

    if torch.any(target_lengths <= 0):
        raise ValueError("Weighted target-length CTC requires positive target lengths")
    power = 1.0 if target_length_weighting == "linear" else 0.5
    lengths = target_lengths.to(dtype=log_probs.dtype)
    relative_lengths = lengths / lengths.mean()
    raw_weights = relative_lengths.pow(power).clamp(max=target_length_weight_cap)
    weights = raw_weights / raw_weights.mean()
    per_sample_losses = nn.functional.ctc_loss(
        log_probs,
        reduction="none",
        **ctc_arguments,
    )
    normalized_losses = per_sample_losses / lengths
    return (normalized_losses * weights).mean()


def collapse_ctc_path(token_ids: Iterable[int], blank_id: int) -> list[int]:
    """Collapse repeated CTC IDs and remove blank IDs."""
    collapsed: list[int] = []
    previous: int | None = None
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == blank_id:
            previous = token_id
            continue
        if token_id != previous:
            collapsed.append(token_id)
        previous = token_id
    return collapsed


def greedy_ctc_decode(
    logits: Tensor,
    input_lengths: Tensor,
    vocabulary: GlossVocabulary,
    *,
    blank_id: int | None = None,
) -> list[str]:
    """Decode batch-first CTC logits into gloss strings."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, C)")
    if input_lengths.ndim != 1 or input_lengths.shape[0] != logits.shape[0]:
        raise ValueError("input_lengths must have one value per batch item")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > logits.shape[1]):
        raise ValueError("input_lengths are outside the logits time dimension")
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN] if blank_id is None else blank_id
    if blank_id < 0 or blank_id >= logits.shape[-1]:
        raise ValueError("blank_id is outside the logits vocabulary")

    best_ids = logits.argmax(dim=-1)
    decoded: list[str] = []
    for row, length in zip(best_ids, input_lengths.tolist(), strict=True):
        token_ids = collapse_ctc_path(row[:length].tolist(), blank_id)
        decoded.append(vocabulary.decode(token_ids))
    return decoded


def ctc_prefix_beam_search(
    logits: Tensor,
    input_lengths: Tensor,
    vocabulary: GlossVocabulary,
    *,
    beam_width: int = 10,
) -> list[list[int]]:
    """Decode CTC logits with log-domain prefix beam search.

    Blank-ending and nonblank-ending probabilities are kept separately for
    every prefix. A repeated token can extend a prefix only after a blank,
    which preserves repeated glosses while applying standard CTC semantics.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, C)")
    if input_lengths.ndim != 1 or input_lengths.shape[0] != logits.shape[0]:
        raise ValueError("input_lengths must contain one value per batch item")
    if torch.any(input_lengths <= 0) or torch.any(input_lengths > logits.shape[1]):
        raise ValueError("input_lengths are outside the logits time dimension")
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")

    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    if blank_id < 0 or blank_id >= logits.shape[-1]:
        raise ValueError("blank_id is outside the logits vocabulary")
    log_probs = logits.log_softmax(dim=-1).detach().cpu()
    neg_inf = float("-inf")

    def logadd(first: float, second: float) -> float:
        if first == neg_inf:
            return second
        if second == neg_inf:
            return first
        larger = max(first, second)
        smaller = min(first, second)
        return larger + math.log1p(math.exp(smaller - larger))

    decoded: list[list[int]] = []
    for batch_index, length in enumerate(input_lengths.detach().cpu().tolist()):
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, neg_inf)}
        for frame in range(int(length)):
            frame_log_probs = log_probs[batch_index, frame]
            next_beams: dict[tuple[int, ...], list[float]] = {}

            def add(
                prefix: tuple[int, ...],
                blank_score: float,
                nonblank_score: float,
                target_beams: dict[tuple[int, ...], list[float]] = next_beams,
            ) -> None:
                scores = target_beams.setdefault(prefix, [neg_inf, neg_inf])
                scores[0] = logadd(scores[0], blank_score)
                scores[1] = logadd(scores[1], nonblank_score)

            for prefix, (log_blank, log_nonblank) in beams.items():
                total = logadd(log_blank, log_nonblank)
                add(prefix, total + float(frame_log_probs[blank_id]), neg_inf)
                for token_id in range(logits.shape[-1]):
                    if token_id == blank_id:
                        continue
                    token_score = float(frame_log_probs[token_id])
                    if prefix and token_id == prefix[-1]:
                        add(prefix, neg_inf, log_nonblank + token_score)
                        add(prefix + (token_id,), neg_inf, log_blank + token_score)
                    else:
                        add(prefix + (token_id,), neg_inf, total + token_score)

            beams = {
                prefix: (scores[0], scores[1])
                for prefix, scores in sorted(
                    next_beams.items(),
                    key=lambda item: logadd(item[1][0], item[1][1]),
                    reverse=True,
                )[:beam_width]
            }

        best_prefix = max(
            beams,
            key=lambda prefix: logadd(beams[prefix][0], beams[prefix][1]),
        )
        decoded.append(list(best_prefix))

    return decoded

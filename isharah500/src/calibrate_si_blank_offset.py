"""Calibrate a CTC blank-logit offset on SI train and evaluate on SI dev.

This is a read-only experiment. It never changes model parameters or
checkpoints. The default decoder remains unchanged because an offset of zero
returns the original logits unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import greedy_ctc_decode  # noqa: E402
from isharah_dataset import GlossVocabulary, IsharahDataset  # noqa: E402
from isharah_metrics import (  # noqa: E402
    align_token_sequences,
    alignment_counts,
    calculate_token_metrics,
)
from isharah_training import (  # noqa: E402
    build_si_train_dev_datasets,
    load_checkpoint,
    make_loader,
)

DEFAULT_OFFSETS = (-3.0, -2.0, -1.5, -1.0, -0.5, 0.0)


def apply_blank_logit_offset(
    logits: torch.Tensor,
    blank_id: int,
    offset: float,
) -> torch.Tensor:
    """Return logits with ``offset`` added to the CTC blank class.

    Negative values reduce blank preference and positive values increase it.
    The input tensor is never modified. An offset of zero is an exact copy of
    the baseline logits, preserving the existing decoder contract.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (B, T, C)")
    if blank_id < 0 or blank_id >= logits.shape[-1]:
        raise ValueError("blank_id is outside the logits vocabulary")
    if not torch.isfinite(logits).all():
        raise ValueError("logits contain NaN or infinite values")
    if not torch.isfinite(torch.tensor(offset, dtype=logits.dtype)):
        raise ValueError("offset must be finite")
    calibrated = logits.clone()
    calibrated[..., blank_id] += offset
    return calibrated


def _sample_limit(dataset: IsharahDataset, max_samples: int | None) -> Sequence[int]:
    if max_samples is None:
        return range(len(dataset))
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    return range(min(max_samples, len(dataset)))


def _token_sequences(
    predictions: Sequence[str],
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    vocabulary: GlossVocabulary,
) -> tuple[list[list[str]], list[list[str]]]:
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    offset = 0
    for prediction, target_length in zip(predictions, target_lengths.tolist(), strict=True):
        target = vocabulary.decode(target_ids[offset : offset + target_length].tolist())
        offset += target_length
        references.append(target.split() if target else [])
        hypotheses.append(prediction.split() if prediction else [])
    if offset != int(target_ids.numel()):
        raise ValueError("target lengths do not consume concatenated target IDs")
    return references, hypotheses


def _long_target_summary(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
    minimum_length: int = 5,
) -> dict[str, Any]:
    selected = [
        (reference, hypothesis)
        for reference, hypothesis in zip(references, hypotheses, strict=True)
        if len(reference) >= minimum_length
    ]
    substitutions = deletions = insertions = 0
    for reference, hypothesis in selected:
        counts = alignment_counts(align_token_sequences(reference, hypothesis))
        substitutions += counts["substitutions"]
        deletions += counts["deletions"]
        insertions += counts["insertions"]
    reference_tokens = sum(len(reference) for reference, _ in selected)
    return {
        "minimum_target_length": minimum_length,
        "sample_count": len(selected),
        "reference_tokens": reference_tokens,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "deletion_rate": deletions / max(reference_tokens, 1),
    }


def evaluate_logits(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    target_ids: torch.Tensor,
    target_lengths: torch.Tensor,
    vocabulary: GlossVocabulary,
    *,
    blank_id: int,
) -> dict[str, Any]:
    """Calculate canonical metrics and blank/length diagnostics for one batch."""
    predictions = greedy_ctc_decode(
        logits,
        input_lengths,
        vocabulary,
        blank_id=blank_id,
    )
    references, hypotheses = _token_sequences(
        predictions,
        target_ids.detach().cpu(),
        target_lengths.detach().cpu(),
        vocabulary,
    )
    metrics = calculate_token_metrics(references, hypotheses)
    best_ids = logits.argmax(dim=-1)
    blank_frames = 0
    total_frames = 0
    prediction_lengths: list[int] = []
    for row, length, hypothesis in zip(best_ids, input_lengths.tolist(), hypotheses, strict=True):
        path = row[:length]
        blank_frames += int((path == blank_id).sum().item())
        total_frames += length
        prediction_lengths.append(len(hypothesis))
    return {
        **metrics,
        "blank_fraction": blank_frames / max(total_frames, 1),
        "predicted_length": {
            "count": len(prediction_lengths),
            "mean": sum(prediction_lengths) / max(len(prediction_lengths), 1),
            "distribution": {
                str(length): count for length, count in sorted(Counter(prediction_lengths).items())
            },
        },
        "long_target_deletions": _long_target_summary(references, hypotheses),
    }


def evaluate_dataset(
    model: torch.nn.Module,
    dataset: IsharahDataset,
    vocabulary: GlossVocabulary,
    device: torch.device,
    *,
    blank_offset: float,
    batch_size: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate a strict real dataset without changing model parameters."""
    from torch.utils.data import Subset

    indices = _sample_limit(dataset, max_samples)
    subset = Subset(dataset, indices)
    loader = make_loader(subset, batch_size, 0, shuffle=False)
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    blank_frames = total_frames = 0
    prediction_lengths: list[int] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            input_lengths = batch["input_lengths"].to(device)
            logits = model(
                batch["keypoints"].to(device),
                input_lengths,
                batch["masks"].to(device),
            )
            calibrated = apply_blank_logit_offset(logits, blank_id, blank_offset)
            predictions = greedy_ctc_decode(
                calibrated, input_lengths, vocabulary, blank_id=blank_id
            )
            batch_references, batch_hypotheses = _token_sequences(
                predictions,
                batch["target_ids"],
                batch["target_lengths"],
                vocabulary,
            )
            references.extend(batch_references)
            hypotheses.extend(batch_hypotheses)
            best_ids = calibrated.argmax(dim=-1)
            for row, length, hypothesis in zip(
                best_ids, input_lengths.tolist(), batch_hypotheses, strict=True
            ):
                path = row[:length]
                blank_frames += int((path == blank_id).sum().item())
                total_frames += length
                prediction_lengths.append(len(hypothesis))

    metrics = calculate_token_metrics(references, hypotheses)
    return {
        **metrics,
        "blank_offset": float(blank_offset),
        "blank_fraction": blank_frames / max(total_frames, 1),
        "predicted_length": {
            "count": len(prediction_lengths),
            "mean": sum(prediction_lengths) / max(len(prediction_lengths), 1),
            "distribution": {
                str(length): count for length, count in sorted(Counter(prediction_lengths).items())
            },
        },
        "long_target_deletions": _long_target_summary(references, hypotheses),
    }


def select_blank_offset(
    model: torch.nn.Module,
    calibration_dataset: IsharahDataset,
    vocabulary: GlossVocabulary,
    device: torch.device,
    offsets: Sequence[float],
    *,
    batch_size: int,
    max_samples: int,
) -> tuple[float, list[dict[str, Any]]]:
    """Select the lowest train-calibration GER with deterministic tie-breaking."""
    if not offsets:
        raise ValueError("at least one blank offset is required")
    results = [
        evaluate_dataset(
            model,
            calibration_dataset,
            vocabulary,
            device,
            blank_offset=float(offset),
            batch_size=batch_size,
            max_samples=max_samples,
        )
        for offset in offsets
    ]
    selected = min(
        results,
        key=lambda result: (
            result["gloss_error_rate"],
            -result["exact_match"],
            abs(result["blank_offset"]),
            result["blank_offset"],
        ),
    )
    return float(selected["blank_offset"]), results


def run_calibration(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    offsets: Sequence[float] = DEFAULT_OFFSETS,
    calibration_samples: int = 100,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Calibrate on SI train and evaluate the selected offset on SI dev."""
    if calibration_samples <= 0 or batch_size <= 0:
        raise ValueError("calibration_samples and batch_size must be positive")
    device = torch.device(
        "cuda"
        if device_name == "auto" and torch.cuda.is_available()
        else device_name
        if device_name != "auto"
        else "cpu"
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model, vocabulary, preprocessing_config, _ = load_checkpoint(checkpoint_path, device=device)
    if preprocessing_config.missing_file_policy != "error":
        raise ValueError("Calibration requires strict real-data loading")
    train_set, dev_set, dev_vocabulary = build_si_train_dev_datasets(
        project_root, data_root, preprocessing_config
    )
    if dev_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI training vocabulary")
    selected_offset, calibration_results = select_blank_offset(
        model,
        train_set,
        vocabulary,
        device,
        offsets,
        batch_size=batch_size,
        max_samples=calibration_samples,
    )
    baseline = evaluate_dataset(
        model,
        dev_set,
        vocabulary,
        device,
        blank_offset=0.0,
        batch_size=batch_size,
    )
    selected = evaluate_dataset(
        model,
        dev_set,
        vocabulary,
        device,
        blank_offset=selected_offset,
        batch_size=batch_size,
    )
    return {
        "protocol": "SI",
        "calibration_split": "train",
        "evaluation_split": "dev",
        "sample_count": len(dev_set),
        "calibration_sample_count": min(calibration_samples, len(train_set)),
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "preprocessing_config": preprocessing_config.to_dict(),
        "candidate_offsets": [float(offset) for offset in offsets],
        "selected_blank_offset": selected_offset,
        "calibration_results": calibration_results,
        "baseline_dev": baseline,
        "selected_dev": selected,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=list(DEFAULT_OFFSETS),
        help="Candidate blank-logit offsets; negative values reduce blank preference.",
    )
    arguments = parser.parse_args()
    output = arguments.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {output}")
    report = run_calibration(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
        offsets=arguments.offsets,
        calibration_samples=arguments.calibration_samples,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

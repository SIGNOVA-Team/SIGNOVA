"""Read-only SI feature and logit diagnostics for a trained checkpoint.

This diagnostic samples only the SI train and dev splits. Raw statistics are
read from the keypoint files referenced by the datasets, while postprocessed
values and masks come from ``IsharahDataset`` using the checkpoint contract.
It never writes files, updates weights, or loads SI test metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import BiGRUCTC  # noqa: E402
from isharah_training import (  # noqa: E402
    build_si_train_dev_datasets,
    load_checkpoint,
    make_loader,
)


def _range_stats(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def _movement_stats(values: np.ndarray) -> dict[str, Any]:
    if values.shape[0] < 2:
        return {"frame_pairs": 0, "mean_abs_delta": None, "mean_l2_delta": None}
    delta = np.diff(values, axis=0)
    finite = np.isfinite(delta)
    abs_values = np.abs(delta)[finite]
    l2_per_coordinate_group = np.linalg.norm(delta, axis=-1)
    l2_values = l2_per_coordinate_group[np.isfinite(l2_per_coordinate_group)]
    return {
        "frame_pairs": int(values.shape[0] - 1),
        "mean_abs_delta": float(abs_values.mean()) if abs_values.size else None,
        "mean_l2_delta": float(l2_values.mean()) if l2_values.size else None,
    }


def _sample_feature_stats(dataset: Any, count: int) -> dict[str, Any]:
    raw_ranges: list[dict[str, Any]] = []
    post_ranges: list[dict[str, Any]] = []
    raw_nan_fractions: list[float] = []
    post_nan_fractions: list[float] = []
    mask_percentages: list[dict[str, Any]] = []
    movements: list[dict[str, Any]] = []
    lengths: list[int] = []

    for index in range(min(count, len(dataset))):
        row = dataset.samples[index]
        raw_path = dataset.root_dir / row["keypoints_path"]
        raw = np.load(raw_path, allow_pickle=False)
        item = dataset[index]
        post = item["keypoints"].detach().cpu().numpy()
        mask = item["mask"].detach().cpu().numpy().astype(bool)

        raw_ranges.append(_range_stats(raw))
        post_ranges.append(_range_stats(post))
        raw_nan_fractions.append(float(np.isnan(raw).mean()))
        post_nan_fractions.append(float(np.isnan(post).mean()))
        mask_percentages.append(
            {
                "overall": float(mask.mean() * 100.0),
                "hand_0": float(mask[:, 0].mean() * 100.0),
                "hand_1": float(mask[:, 1].mean() * 100.0),
            }
        )
        movements.append(_movement_stats(post))
        lengths.append(int(item["length"]))

    def _mean(values: list[float]) -> float | None:
        return float(np.mean(values)) if values else None

    def _mean_dict(values: list[dict[str, float]]) -> dict[str, float | None]:
        keys = ("overall", "hand_0", "hand_1")
        return {key: _mean([float(value[key]) for value in values]) for key in keys}

    return {
        "sample_count": len(lengths),
        "lengths": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": _mean([float(value) for value in lengths]),
        },
        "raw_coordinate_ranges": raw_ranges,
        "postprocessed_coordinate_ranges": post_ranges,
        "raw_nan_fraction": {
            "mean": _mean(raw_nan_fractions),
            "max": max(raw_nan_fractions) if raw_nan_fractions else None,
        },
        "postprocessed_nan_fraction": {
            "mean": _mean(post_nan_fractions),
            "max": max(post_nan_fractions) if post_nan_fractions else None,
        },
        "valid_hand_mask_percent": _mean_dict(mask_percentages),
        "frame_to_frame_movement": {
            "mean_abs_delta": _mean(
                [
                    float(value["mean_abs_delta"])
                    for value in movements
                    if value["mean_abs_delta"] is not None
                ]
            ),
            "mean_l2_delta": _mean(
                [
                    float(value["mean_l2_delta"])
                    for value in movements
                    if value["mean_l2_delta"] is not None
                ]
            ),
        },
    }


def _logit_stats(
    model: BiGRUCTC,
    dataset: Any,
    vocabulary: Any,
    count: int,
    device: torch.device,
) -> dict[str, Any]:
    subset = Subset(dataset, range(min(count, len(dataset))))
    loader = make_loader(
        subset, batch_size=max(1, min(count, len(dataset))), num_workers=0, shuffle=False
    )
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    blank_probabilities: list[torch.Tensor] = []
    strongest_nonblank_probabilities: list[torch.Tensor] = []
    margins: list[torch.Tensor] = []
    logits_values: list[torch.Tensor] = []
    argmax_blank = 0
    valid_frames = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            keypoints = batch["keypoints"].to(device)
            masks = batch["masks"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            logits = model(keypoints, input_lengths, masks)
            probabilities = logits.softmax(dim=-1)
            for row, length in enumerate(input_lengths.tolist()):
                row_logits = logits[row, :length]
                row_probabilities = probabilities[row, :length]
                nonblank_ids = [index for index in range(logits.shape[-1]) if index != blank_id]
                nonblank_logits = row_logits[:, nonblank_ids]
                nonblank_probabilities = row_probabilities[:, nonblank_ids]
                blank_probabilities.append(row_probabilities[:, blank_id].cpu())
                strongest_nonblank_probabilities.append(
                    nonblank_probabilities.max(dim=-1).values.cpu()
                )
                margins.append((row_logits[:, blank_id] - nonblank_logits.max(dim=-1).values).cpu())
                logits_values.append(row_logits.cpu())
                argmax_blank += int((row_logits.argmax(dim=-1) == blank_id).sum())
                valid_frames += int(length)

    all_blank = torch.cat(blank_probabilities) if blank_probabilities else torch.empty(0)
    all_nonblank = (
        torch.cat(strongest_nonblank_probabilities)
        if strongest_nonblank_probabilities
        else torch.empty(0)
    )
    all_margins = torch.cat(margins) if margins else torch.empty(0)
    all_logits = torch.cat(logits_values) if logits_values else torch.empty(0)
    bias = model.classifier.bias.detach().cpu()
    nonblank_bias = torch.cat((bias[:blank_id], bias[blank_id + 1 :]))

    def mean(value: torch.Tensor) -> float | None:
        return float(value.mean()) if value.numel() else None

    return {
        "sample_count": len(subset),
        "valid_frames": valid_frames,
        "blank_id": int(blank_id),
        "argmax_blank_fraction": argmax_blank / max(valid_frames, 1),
        "mean_blank_probability": mean(all_blank),
        "mean_strongest_nonblank_probability": mean(all_nonblank),
        "mean_blank_minus_best_nonblank_logit": mean(all_margins),
        "logit_range": _range_stats(all_logits.numpy()) if all_logits.numel() else {},
        "classifier_bias": {
            "blank": float(bias[blank_id]),
            "mean": float(bias.mean()),
            "min": float(bias.min()),
            "max": float(bias.max()),
            "best_nonblank": float(nonblank_bias.max()) if nonblank_bias.numel() else None,
        },
    }


def diagnose_features(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    sample_count: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto"
        else torch.device(device_name)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model, vocabulary, preprocessing_config, _ = load_checkpoint(checkpoint_path, device=device)
    if preprocessing_config.missing_file_policy != "error":
        raise ValueError("Refusing diagnostics with placeholder file policy")
    train_set, dev_set, train_dev_vocabulary = build_si_train_dev_datasets(
        project_root, data_root, preprocessing_config
    )
    if train_dev_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI train vocabulary")

    return {
        "protocol": "SI",
        "splits": {
            "train": _sample_feature_stats(train_set, sample_count),
            "dev": _sample_feature_stats(dev_set, sample_count),
        },
        "logits": {
            "train": _logit_stats(model, train_set, vocabulary, sample_count, device),
            "dev": _logit_stats(model, dev_set, vocabulary, sample_count, device),
        },
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "sample_count_per_split": sample_count,
        "preprocessing_config": preprocessing_config.to_dict(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root / "isharah500" / "checkpoints" / "si_pilot" / "isharah500_si_best.pt",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    try:
        result = diagnose_features(
            arguments.project_root,
            arguments.data_root,
            arguments.checkpoint,
            sample_count=arguments.samples,
            device_name=arguments.device,
        )
    except Exception as exc:
        result = {"error": type(exc).__name__, "message": str(exc)}
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()

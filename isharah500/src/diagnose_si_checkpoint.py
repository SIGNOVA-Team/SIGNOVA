"""Read-only diagnostics for the Isharah-500 SI dev checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import collapse_ctc_path, greedy_ctc_decode  # noqa: E402
from isharah_metrics import calculate_token_metrics  # noqa: E402
from isharah_training import (  # noqa: E402
    build_si_train_dev_datasets,
    load_checkpoint,
    make_loader,
)


def _summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "distribution": {}}
    counts = Counter(values)
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "distribution": {str(length): count for length, count in sorted(counts.items())},
    }


def _top_tokens(
    counts: Counter[int],
    vocabulary: Any,
    limit: int = 20,
) -> list[dict[str, Any]]:
    return [
        {
            "token_id": int(token_id),
            "token": vocabulary.id2token.get(int(token_id), f"<id:{token_id}>"),
            "count": int(count),
        }
        for token_id, count in counts.most_common(limit)
    ]


def diagnose_checkpoint(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the SI dev split without training or writing files."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

    model, vocabulary, preprocessing_config, _ = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    if preprocessing_config.missing_file_policy != "error":
        raise ValueError(
            "Checkpoint preprocessing permits placeholders; refusing real-data diagnosis"
        )

    _, dev_set, dev_vocabulary = build_si_train_dev_datasets(
        project_root,
        data_root,
        preprocessing_config,
    )
    if dev_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from the SI training vocabulary")
    if len(dev_set) != 500:
        raise ValueError(f"Expected 500 SI dev samples, found {len(dev_set)}")

    loader = make_loader(
        dev_set,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    raw_token_counts: Counter[int] = Counter()
    predicted_token_counts: Counter[int] = Counter()
    target_token_counts: Counter[int] = Counter()
    prediction_lengths: list[int] = []
    target_lengths: list[int] = []
    length_pairs: Counter[tuple[int, int]] = Counter()
    examples: list[dict[str, Any]] = []
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []
    blank_frames = 0
    total_frames = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            keypoints = batch["keypoints"].to(device)
            masks = batch["masks"].to(device)
            input_lengths = batch["input_lengths"].to(device)
            target_ids = batch["target_ids"].to(device)
            batch_target_lengths = batch["target_lengths"].to(device)

            logits = model(keypoints, input_lengths, masks)
            predictions = greedy_ctc_decode(logits, input_lengths, vocabulary)
            best_ids = logits.argmax(dim=-1)
            target_offset = 0

            for row_index, input_length in enumerate(input_lengths.tolist()):
                raw_path = best_ids[row_index, :input_length].tolist()
                collapsed_path = collapse_ctc_path(raw_path, blank_id)
                target_length = int(batch_target_lengths[row_index].item())
                target_path = (
                    target_ids[target_offset : target_offset + target_length]
                    .detach()
                    .cpu()
                    .tolist()
                )
                target_offset += target_length
                references.append(vocabulary.decode(target_path).split())
                hypotheses.append(predictions[row_index].split())

                raw_token_counts.update(raw_path)
                predicted_token_counts.update(collapsed_path)
                target_token_counts.update(target_path)
                blank_frames += sum(token_id == blank_id for token_id in raw_path)
                total_frames += len(raw_path)

                predicted_length = len(collapsed_path)
                prediction_lengths.append(predicted_length)
                target_lengths.append(target_length)
                length_pairs[(target_length, predicted_length)] += 1

                if len(examples) < 20:
                    examples.append(
                        {
                            "sample_id": batch["sample_ids"][row_index],
                            "target": vocabulary.decode(target_path),
                            "prediction": predictions[row_index],
                            "input_length": int(input_length),
                            "target_length": target_length,
                            "predicted_length": predicted_length,
                            "blank_fraction": sum(token_id == blank_id for token_id in raw_path)
                            / max(len(raw_path), 1),
                        }
                    )

    return {
        "protocol": "SI",
        "split": "dev",
        "sample_count": len(dev_set),
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "preprocessing_config": preprocessing_config.to_dict(),
        "metrics": calculate_token_metrics(references, hypotheses),
        "blank_token": {
            "id": blank_id,
            "fraction": blank_frames / max(total_frames, 1),
        },
        "predicted_lengths": _summary(prediction_lengths),
        "target_lengths": _summary(target_lengths),
        "target_predicted_length_pairs": {
            f"{target}->{predicted}": count
            for (target, predicted), count in sorted(length_pairs.items())
        },
        "token_frequency": {
            "raw_frame_tokens": _top_tokens(raw_token_counts, vocabulary),
            "predicted_collapsed_tokens": _top_tokens(
                predicted_token_counts,
                vocabulary,
            ),
            "target_tokens": _top_tokens(target_token_counts, vocabulary),
        },
        "examples": examples,
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()

    result = diagnose_checkpoint(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

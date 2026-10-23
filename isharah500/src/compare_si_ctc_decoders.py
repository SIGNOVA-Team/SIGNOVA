"""Compare greedy and prefix-beam CTC decoding on SI dev data only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import ctc_prefix_beam_search, greedy_ctc_decode  # noqa: E402
from isharah_metrics import calculate_token_metrics  # noqa: E402
from isharah_training import (  # noqa: E402
    build_si_train_dev_datasets,
    load_checkpoint,
    make_loader,
)


def _metrics(predictions: list[list[str]], targets: list[list[str]]) -> dict[str, Any]:
    prediction_lengths: list[int] = []
    for prediction in predictions:
        prediction_lengths.append(len(prediction))
    metrics = calculate_token_metrics(targets, predictions)
    return {
        "gloss_error_rate": metrics["gloss_error_rate"],
        "exact_match": metrics["exact_match"],
        "mean_prediction_length": sum(prediction_lengths) / max(len(prediction_lengths), 1),
        "sample_count": len(targets),
    }


def _select_dev_subset(dataset: Any, max_samples: int | None) -> Any:
    """Return a deterministic prefix subset without changing dataset contents."""
    if max_samples is None:
        return dataset
    if max_samples <= 0:
        raise ValueError("max_samples must be positive when provided")
    return Subset(dataset, range(min(max_samples, len(dataset))))


def _report_progress(processed: int, total: int, started_at: float) -> None:
    elapsed = time.monotonic() - started_at
    print(
        f"CTC decoder progress: processed={processed}/{total} elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def compare_decoders(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 8,
    device_name: str = "auto",
    max_samples: int | None = None,
    beam_widths: tuple[int, ...] = (5, 10),
    progress_every: int = 50,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if progress_every <= 0:
        raise ValueError("progress_every must be positive")
    if not beam_widths or any(width <= 0 for width in beam_widths):
        raise ValueError("beam_widths must contain positive values")
    if len(set(beam_widths)) != len(beam_widths):
        raise ValueError("beam_widths must not contain duplicates")
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
        raise ValueError("Refusing diagnosis with placeholder file policy")
    _, dev_set, dev_vocabulary = build_si_train_dev_datasets(
        project_root,
        data_root,
        preprocessing_config,
    )
    if dev_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI train vocabulary")

    evaluation_set = _select_dev_subset(dev_set, max_samples)
    total_samples = len(evaluation_set)
    loader = make_loader(evaluation_set, batch_size=batch_size, num_workers=0, shuffle=False)
    decoder_names = ("greedy",) + tuple(f"beam_{width}" for width in beam_widths)
    predictions: dict[str, list[list[str]]] = {name: [] for name in decoder_names}
    targets: list[list[str]] = []
    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in decoder_names}
    started_at = time.monotonic()
    processed = 0
    next_progress = progress_every

    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_lengths = batch["input_lengths"].to(device)
            logits = model(
                batch["keypoints"].to(device),
                input_lengths,
                batch["masks"].to(device),
            )
            # Existing greedy decoding is reused for the baseline string output.
            greedy_strings = greedy_ctc_decode(logits, input_lengths, vocabulary)
            beam_outputs = {
                f"beam_{width}": ctc_prefix_beam_search(
                    logits, input_lengths, vocabulary, beam_width=width
                )
                for width in beam_widths
            }
            target_offset = 0
            for row_index, target_length in enumerate(batch["target_lengths"].tolist()):
                target_ids = batch["target_ids"][
                    target_offset : target_offset + target_length
                ].tolist()
                target_offset += target_length
                target_text = vocabulary.decode(target_ids)
                target_tokens = target_text.split()
                targets.append(target_tokens)
                row_outputs = {
                    "greedy": greedy_strings[row_index].split(),
                }
                row_outputs.update(
                    {
                        name: vocabulary.decode(beam_outputs[name][row_index]).split()
                        for name in beam_outputs
                    }
                )
                for name, output in row_outputs.items():
                    predictions[name].append(output)
                    if len(examples[name]) < 20:
                        examples[name].append(
                            {
                                "sample_id": batch["sample_ids"][row_index],
                                "target": target_text,
                                "prediction": " ".join(output),
                            }
                        )
            processed += len(batch["sample_ids"])
            if processed >= next_progress or processed == total_samples:
                _report_progress(processed, total_samples, started_at)
                while next_progress <= processed:
                    next_progress += progress_every

    return {
        "protocol": "SI",
        "split": "dev",
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "sample_count": total_samples,
        "max_samples": max_samples,
        "beam_widths": list(beam_widths),
        "preprocessing_config": preprocessing_config.to_dict(),
        "decoders": {
            name: {**_metrics(predictions[name], targets), "examples": examples[name]}
            for name in decoder_names
        },
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_root
        / "isharah500"
        / "checkpoints"
        / "si_hand_centric_5ep"
        / "isharah500_si_best.pt",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate only the first deterministic SI dev samples; default is all.",
    )
    parser.add_argument(
        "--beam-widths",
        type=int,
        nargs="+",
        default=[5, 10],
        help="One or more prefix beam widths; greedy decoding is always included.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress to stderr after this many processed samples.",
    )
    arguments = parser.parse_args()
    result = compare_decoders(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
        max_samples=arguments.max_samples,
        beam_widths=tuple(arguments.beam_widths),
        progress_every=arguments.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Analyze token-level errors for an Isharah-500 SI checkpoint on dev data."""

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

COVERAGE_BUCKETS = ("0-24%", "25-49%", "50-74%", "75-99%", "100%")


def coverage_bucket(coverage: float) -> str:
    """Map valid-hand coverage in [0, 1] to a deterministic report bucket."""
    if not 0.0 <= coverage <= 1.0:
        raise ValueError("valid-hand coverage must be in [0, 1]")
    if coverage < 0.25:
        return COVERAGE_BUCKETS[0]
    if coverage < 0.50:
        return COVERAGE_BUCKETS[1]
    if coverage < 0.75:
        return COVERAGE_BUCKETS[2]
    if coverage < 1.0:
        return COVERAGE_BUCKETS[3]
    return COVERAGE_BUCKETS[4]


def _sample_record(
    sample_id: str,
    reference: Sequence[str],
    hypothesis: Sequence[str],
    valid_hand_coverage: float,
) -> dict[str, Any]:
    alignment = align_token_sequences(reference, hypothesis)
    return {
        "sample_id": sample_id,
        "reference": list(reference),
        "hypothesis": list(hypothesis),
        "target_length": len(reference),
        "prediction_length": len(hypothesis),
        "valid_hand_coverage": valid_hand_coverage,
        "coverage_bucket": coverage_bucket(valid_hand_coverage),
        "exact_match": reference == hypothesis,
        "alignment": alignment,
        **alignment_counts(alignment),
    }


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    metrics = calculate_token_metrics(
        [record["reference"] for record in records],
        [record["hypothesis"] for record in records],
    )
    prediction_lengths = Counter(record["prediction_length"] for record in records)
    return {
        **metrics,
        "mean_prediction_length": sum(record["prediction_length"] for record in records)
        / max(len(records), 1),
        "mean_target_length": sum(record["target_length"] for record in records)
        / max(len(records), 1),
        "deletion_rate": metrics["deletions"] / max(metrics["reference_tokens"], 1),
        "prediction_length_distribution": {
            str(length): count for length, count in sorted(prediction_lengths.items())
        },
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate, target-length, coverage, and confusion summaries."""
    target_groups: dict[int, list[dict[str, Any]]] = {}
    coverage_groups = {bucket: [] for bucket in COVERAGE_BUCKETS}
    substitutions: Counter[tuple[str, str]] = Counter()
    for record in records:
        target_groups.setdefault(record["target_length"], []).append(record)
        coverage_groups[record["coverage_bucket"]].append(record)
        for step in record["alignment"]:
            if step["operation"] == "substitution":
                substitutions[(str(step["reference"]), str(step["hypothesis"]))] += 1

    return {
        "aggregate": _summary(records),
        "by_target_length": {
            str(length): _summary(target_groups[length]) for length in sorted(target_groups)
        },
        "by_valid_hand_coverage": {
            bucket: _summary(coverage_groups[bucket])
            for bucket in COVERAGE_BUCKETS
            if coverage_groups[bucket]
        },
        "most_frequent_confusions": [
            {
                "reference": reference,
                "hypothesis": hypothesis,
                "count": count,
            }
            for (reference, hypothesis), count in sorted(
                substitutions.items(),
                key=lambda item: (-item[1], item[0][0], item[0][1]),
            )
        ],
    }


def _markdown_summary(report: dict[str, Any]) -> str:
    aggregate = report["analysis"]["aggregate"]
    lines = [
        "# Isharah-500 SI Error Analysis",
        "",
        f"- Split: `{report['split']}`",
        f"- Samples: `{aggregate['sample_count']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| GER | {aggregate['gloss_error_rate']:.4f} |",
        f"| Exact match | {aggregate['exact_match']:.4f} |",
        f"| Substitutions | {aggregate['substitutions']} |",
        f"| Deletions | {aggregate['deletions']} |",
        f"| Insertions | {aggregate['insertions']} |",
        f"| Mean prediction length | {aggregate['mean_prediction_length']:.3f} |",
        f"| Mean target length | {aggregate['mean_target_length']:.3f} |",
        f"| Deletion rate | {aggregate['deletion_rate']:.4f} |",
        "",
        "## Target length groups",
        "",
        "| Target length | Samples | GER | Exact match | Deletion rate |",
        "|---:|---:|---:|---:|---:|",
    ]
    for length, values in report["analysis"]["by_target_length"].items():
        lines.append(
            f"| {length} | {values['sample_count']} | {values['gloss_error_rate']:.4f} | "
            f"{values['exact_match']:.4f} | {values['deletion_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Valid-hand coverage groups",
            "",
            "| Coverage | Samples | GER | Exact match | Deletion rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for bucket, values in report["analysis"]["by_valid_hand_coverage"].items():
        lines.append(
            f"| {bucket} | {values['sample_count']} | {values['gloss_error_rate']:.4f} | "
            f"{values['exact_match']:.4f} | {values['deletion_rate']:.4f} |"
        )
    lines.extend(["", "## Most frequent aligned substitutions", ""])
    for item in report["analysis"]["most_frequent_confusions"][:20]:
        lines.append(f"- `{item['reference']}` → `{item['hypothesis']}`: {item['count']}")
    if not report["analysis"]["most_frequent_confusions"]:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def _write_utf8_text(path: Path, content: str) -> None:
    """Write report text as explicit UTF-8 without changing existing files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode("utf-8"))


def analyze_si_errors(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 8,
    device_name: str = "auto",
) -> dict[str, Any]:
    """Evaluate the checkpoint on SI dev and return a JSON-ready report."""
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
        raise ValueError("Refusing analysis with placeholder file policy")
    _, dev_set, dev_vocabulary = build_si_train_dev_datasets(
        project_root,
        data_root,
        preprocessing_config,
    )
    if dev_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI training vocabulary")
    if len(dev_set) != 500:
        raise ValueError(f"Expected 500 SI dev samples, found {len(dev_set)}")

    loader = make_loader(dev_set, batch_size=batch_size, num_workers=0, shuffle=False)
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_lengths = batch["input_lengths"].to(device)
            logits = model(
                batch["keypoints"].to(device),
                input_lengths,
                batch["masks"].to(device),
            )
            predictions = greedy_ctc_decode(logits, input_lengths, vocabulary)
            target_offset = 0
            for row_index, target_length in enumerate(batch["target_lengths"].tolist()):
                target_ids = batch["target_ids"][
                    target_offset : target_offset + target_length
                ].tolist()
                target_offset += target_length
                target_tokens = vocabulary.decode(target_ids).split()
                prediction_tokens = predictions[row_index].split()
                frame_count = int(batch["input_lengths"][row_index].item())
                coverage = float(batch["masks"][row_index, :frame_count].float().mean().item())
                records.append(
                    _sample_record(
                        batch["sample_ids"][row_index],
                        target_tokens,
                        prediction_tokens,
                        coverage,
                    )
                )

    if len(records) != len(dev_set):
        raise RuntimeError(
            f"Evaluation processed {len(records)} samples but expected {len(dev_set)}"
        )

    report = {
        "protocol": "SI",
        "split": "dev",
        "sample_count": len(records),
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "preprocessing_config": preprocessing_config.to_dict(),
        "analysis": aggregate_records(records),
        "samples": [
            {key: value for key, value in record.items() if key != "alignment"}
            for record in records
        ],
    }
    return report


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
        / "si_hand_centric_resume_15ep"
        / "isharah500_si_best.pt",
    )
    parser.add_argument(
        "--json-report",
        type=Path,
        default=project_root / "isharah500" / "reports" / "si_error_analysis.json",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=project_root / "isharah500" / "reports" / "si_error_analysis.md",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    report = analyze_si_errors(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    if arguments.json_report.exists() or arguments.markdown_report.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing error report; choose new output paths"
        )
    _write_utf8_text(
        arguments.json_report,
        json.dumps(report, ensure_ascii=False, indent=2),
    )
    _write_utf8_text(arguments.markdown_report, _markdown_summary(report))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

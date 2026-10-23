"""Run the one-time, read-only SI held-out test evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import greedy_ctc_decode  # noqa: E402
from isharah_metrics import calculate_token_metrics  # noqa: E402
from isharah_training import (  # noqa: E402
    build_si_datasets,
    load_checkpoint,
    make_loader,
    select_device,
    validate_ctc_lengths,
)

EXPECTED_SI_TEST_SAMPLES = 2000


def inspect_si_test_availability(
    project_root: str | Path,
    data_root: str | Path,
    *,
    expected_samples: int = EXPECTED_SI_TEST_SAMPLES,
) -> dict[str, Any]:
    """Check SI test metadata and keypoint paths without loading keypoints.

    This function deliberately does not call ``dataset[index]`` or ``numpy.load``.
    It verifies only metadata row count and referenced file existence.
    """
    _, _, test_set, _ = build_si_datasets(project_root, data_root)
    missing_paths = [
        str(Path(data_root) / sample["keypoints_path"])
        for sample in test_set.samples
        if not (Path(data_root) / sample["keypoints_path"]).is_file()
    ]
    return {
        "protocol": "SI",
        "split": "test",
        "metadata_samples": len(test_set),
        "expected_samples": expected_samples,
        "keypoint_files_present": len(test_set) - len(missing_paths),
        "missing_keypoint_files": missing_paths,
        "available": len(test_set) == expected_samples and not missing_paths,
    }


def _evaluate_test_dataset(
    test_set: Any,
    model: torch.nn.Module,
    vocabulary: Any,
    device: torch.device,
    *,
    batch_size: int,
) -> dict[str, Any]:
    loader = make_loader(test_set, batch_size=batch_size, num_workers=0, shuffle=False)
    references: list[list[str]] = []
    hypotheses: list[list[str]] = []

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            input_lengths = batch["input_lengths"].to(device)
            logits = model(
                batch["keypoints"].to(device),
                input_lengths,
                batch["masks"].to(device),
            )
            predictions = greedy_ctc_decode(logits, input_lengths, vocabulary)
            target_offset = 0
            for row_index, target_length in enumerate(
                batch["target_lengths"].tolist()
            ):
                target_ids = batch["target_ids"][
                    target_offset : target_offset + target_length
                ].tolist()
                target_offset += target_length
                reference = vocabulary.decode(target_ids)
                hypothesis = predictions[row_index]
                references.append(reference.split() if reference else [])
                hypotheses.append(hypothesis.split() if hypothesis else [])

    if len(references) != len(test_set):
        raise RuntimeError(
            f"Evaluation processed {len(references)} samples but expected "
            f"{len(test_set)}"
        )
    return calculate_token_metrics(references, hypotheses)


def _build_report(
    checkpoint_path: str | Path,
    device: torch.device,
    preprocessing_config: Any,
    metrics: dict[str, Any],
    availability: dict[str, Any],
    ctc_validation: dict[str, Any],
    sample_count: int,
) -> dict[str, Any]:
    """Build the stable, test-only report schema."""
    return {
        "protocol": "SI",
        "split": "test",
        "sample_count": sample_count,
        "checkpoint": str(Path(checkpoint_path)),
        "device": str(device),
        "preprocessing_config": preprocessing_config.to_dict(),
        "metrics": metrics,
        "test_input_validation": {
            "metadata_samples": availability["metadata_samples"],
            "keypoint_files_present": availability["keypoint_files_present"],
            "ctc_valid_samples": ctc_validation["valid_samples"],
            "ctc_rejected_samples": len(ctc_validation["rejected_samples"]),
        },
    }


def evaluate_si_test(
    project_root: str | Path,
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    batch_size: int = 8,
    device_name: str = "auto",
    expected_samples: int = EXPECTED_SI_TEST_SAMPLES,
) -> dict[str, Any]:
    """Evaluate one frozen checkpoint on all valid SI test samples.

    No optimizer, training loop, checkpoint save, or model-selection logic is
    reachable from this function. Any missing, invalid, OOV, or CTC-infeasible
    test sample aborts the evaluation instead of being silently excluded.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = select_device(device_name)
    model, vocabulary, preprocessing_config, payload = load_checkpoint(
        checkpoint_path,
        device=device,
    )
    if preprocessing_config.missing_file_policy != "error":
        raise ValueError("Refusing test evaluation with placeholder file policy")
    training_config = payload.get("training_config", {})
    if training_config.get("protocol", "SI") != "SI":
        raise ValueError("The selected checkpoint is not an SI checkpoint")

    train_set, dev_set, test_set, train_vocabulary = build_si_datasets(
        project_root,
        data_root,
        preprocessing_config,
    )
    if train_vocabulary.token2id != vocabulary.token2id:
        raise ValueError("Checkpoint vocabulary differs from SI training vocabulary")
    if len(train_set) != 5000 or len(dev_set) != 500:
        raise ValueError("SI train/dev metadata counts do not match the protocol")
    availability = inspect_si_test_availability(
        project_root,
        data_root,
        expected_samples=expected_samples,
    )
    if not availability["available"]:
        raise FileNotFoundError(
            "SI test availability check failed: "
            f"{availability}"
        )
    ctc_validation = validate_ctc_lengths(test_set)
    if ctc_validation["rejected_samples"]:
        raise ValueError(
            "SI test CTC validation rejected samples; evaluation was aborted: "
            f"{ctc_validation['rejected_samples']}"
        )
    metrics = _evaluate_test_dataset(
        test_set,
        model,
        vocabulary,
        device,
        batch_size=batch_size,
    )
    return _build_report(
        checkpoint_path,
        device,
        preprocessing_config,
        metrics,
        availability,
        ctc_validation,
        len(test_set),
    )


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
        default=project_root
        / "isharah500"
        / "reports"
        / "si_test_evaluation_epoch15.json",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check SI test metadata and keypoint paths without loading samples",
    )
    arguments = parser.parse_args()
    if arguments.verify_only:
        print(
            json.dumps(
                inspect_si_test_availability(
                    arguments.project_root,
                    arguments.data_root,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if arguments.json_report.exists():
        raise FileExistsError(
            "Refusing to overwrite an existing SI test report; choose a new path"
        )
    report = evaluate_si_test(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
        batch_size=arguments.batch_size,
        device_name=arguments.device,
    )
    arguments.json_report.parent.mkdir(parents=True, exist_ok=True)
    arguments.json_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

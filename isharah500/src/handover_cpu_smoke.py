"""Require a real frozen checkpoint and SI Dev sample for handover validation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_inference import load_predictor  # noqa: E402


def _resolve_keypoints(data_root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    candidates = [data_root / relative]
    if relative.parts[:2] == ("data", "processed"):
        candidates.append(data_root.joinpath(*relative.parts[2:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"SI Dev keypoint file not found; checked: {checked}")


def run_smoke(
    project_root: Path,
    data_root: Path,
    checkpoint: Path,
) -> dict[str, object]:
    """Load the frozen checkpoint on CPU and predict one real SI Dev sample."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen checkpoint not found: {checkpoint}")
    metadata = project_root / "isharah500" / "metadata" / "isharah500_si_dev.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"SI Dev metadata not found: {metadata}")

    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row is None or not row.get("keypoints_path"):
        raise ValueError("SI Dev metadata has no usable keypoint row")

    keypoints_path = _resolve_keypoints(data_root, row["keypoints_path"])
    keypoints = np.load(keypoints_path, allow_pickle=False)
    predictor = load_predictor(checkpoint, device_name="cpu")
    first = predictor.predict(keypoints).to_dict()
    second = predictor.predict(keypoints).to_dict()
    if first != second:
        raise RuntimeError("Repeated CPU predictions are not deterministic")
    if first["device"] != "cpu":
        raise RuntimeError(f"Expected CPU inference, got {first['device']}")
    if predictor.model.config.features_per_hand != 63:
        raise RuntimeError("Frozen checkpoint is not compatible with 63-feature input")
    return {
        "checkpoint": str(checkpoint),
        "si_dev_keypoints": str(keypoints_path),
        "prediction": first,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_smoke(
        arguments.project_root,
        arguments.data_root,
        arguments.checkpoint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

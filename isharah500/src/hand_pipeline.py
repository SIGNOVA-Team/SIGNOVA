"""Parallel hand-keypoint extraction and experimental tracking."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from hand_extraction import extract_sample_hand_keypoints
from hand_tracking import track_hand_slots
from keypoint_validation import is_valid_keypoint_file


def save_array_atomic(output_path, array):
    """Save an array without leaving a partial final file."""
    temporary_path = output_path.with_suffix(".tmp")

    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, array)

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def process_sample(sample_directory, model_path, raw_dir, tracked_dir):
    """Extract and track one sample, reusing completed outputs."""
    sample_id = sample_directory.name
    raw_path = raw_dir / f"{sample_id}.npy"
    tracked_path = tracked_dir / f"{sample_id}.npy"

    expected_frames = sum(1 for _ in sample_directory.glob("*.jpg"))

    raw_valid = is_valid_keypoint_file(
        raw_path,
        expected_frames=expected_frames,
    )
    tracked_valid = is_valid_keypoint_file(
        tracked_path,
        expected_frames=expected_frames,
    )

    if raw_valid and tracked_valid:
        return sample_id, "skipped"

    if raw_valid:
        keypoints = np.load(raw_path, allow_pickle=False)
    else:
        keypoints, _ = extract_sample_hand_keypoints(
            sample_directory,
            model_path,
        )
        save_array_atomic(raw_path, keypoints)

    tracked = track_hand_slots(keypoints)
    save_array_atomic(tracked_path, tracked)

    return sample_id, "processed"


def process_directory(
    input_directory,
    model_path,
    raw_dir,
    tracked_dir,
    workers=4,
    limit=None,
):
    """Process frame directories with resumable parallel workers."""
    input_directory = Path(input_directory)
    model_path = Path(model_path)
    raw_dir = Path(raw_dir)
    tracked_dir = Path(tracked_dir)

    if workers < 1:
        raise ValueError("workers must be at least 1.")

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")

    if not input_directory.is_dir():
        raise FileNotFoundError(input_directory)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    if raw_dir.resolve() == tracked_dir.resolve():
        raise ValueError("Raw and tracked directories must differ.")

    sample_directories = sorted(
        directory
        for directory in input_directory.iterdir()
        if directory.is_dir() and any(directory.glob("*.jpg"))
    )

    if limit is not None:
        sample_directories = sample_directories[:limit]

    raw_dir.mkdir(parents=True, exist_ok=True)
    tracked_dir.mkdir(parents=True, exist_ok=True)

    pending = [
        directory
        for directory in sample_directories
        if not (
            is_valid_keypoint_file(
                raw_dir / f"{directory.name}.npy",
                expected_frames=sum(1 for _ in directory.glob("*.jpg")),
            )
            and is_valid_keypoint_file(
                tracked_dir / f"{directory.name}.npy",
                expected_frames=sum(1 for _ in directory.glob("*.jpg")),
            )
        )
    ]

    processed_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_sample,
                directory,
                model_path,
                raw_dir,
                tracked_dir,
            )
            for directory in pending
        ]

        for future in as_completed(futures):
            try:
                future.result()
                processed_count += 1
            except Exception as error:
                failed_count += 1
                print(f"FAILED: {error}")

    print(f"Processed: {processed_count}")
    print(f"Skipped: {len(sample_directories) - len(pending)}")
    print(f"Failed: {failed_count}")

    return failed_count


def main():
    """Run the pipeline from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--tracked-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)

    arguments = parser.parse_args()

    failed_count = process_directory(
        input_directory=arguments.input,
        model_path=arguments.model,
        raw_dir=arguments.raw_output,
        tracked_dir=arguments.tracked_output,
        workers=arguments.workers,
        limit=arguments.limit,
    )

    if failed_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

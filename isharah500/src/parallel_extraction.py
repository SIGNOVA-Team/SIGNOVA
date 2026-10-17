"""Extract hand keypoints from multiple samples in parallel."""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hand_extraction import extract_sample_hand_keypoints


def process_sample(sample_directory, model_path, output_directory):
    """Extract one sample and save its keypoints."""
    output_path = output_directory / f"{sample_directory.name}.npy"

    if output_path.exists():
        return sample_directory.name, "skipped"

    keypoints, _ = extract_sample_hand_keypoints(
        sample_directory,
        model_path,
    )

    temporary_path = output_path.with_suffix(".tmp")

    try:
        with temporary_path.open("wb") as output_file:
            np.save(output_file, keypoints)

        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return sample_directory.name, "processed"


def process_directory(
    input_directory,
    model_path,
    output_directory,
    workers=2,
    limit=None,
):
    """Process sample folders using separate worker processes."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")

    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")

    input_directory = Path(input_directory)
    model_path = Path(model_path)
    output_directory = Path(output_directory)

    if not input_directory.is_dir():
        raise FileNotFoundError(input_directory)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    sample_directories = sorted(
        directory
        for directory in input_directory.iterdir()
        if directory.is_dir()
        and any(directory.glob("*.jpg"))
    )

    if limit is not None:
        sample_directories = sample_directories[:limit]

    output_directory.mkdir(parents=True, exist_ok=True)

    pending_samples = [
        directory
        for directory in sample_directories
        if not (output_directory / f"{directory.name}.npy").exists()
    ]

    skipped_count = len(sample_directories) - len(pending_samples)
    processed_count = 0
    failed_samples = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                directory,
                model_path,
                output_directory,
            ): directory.name
            for directory in pending_samples
        }

        for future in as_completed(futures):
            sample_id = futures[future]

            try:
                _, status = future.result()
                if status == "processed":
                    processed_count += 1
            except Exception as error:
                failed_samples.append((sample_id, str(error)))
                print(f"FAILED {sample_id}: {error}")

    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Failed: {len(failed_samples)}")

    return failed_samples


def main():
    """Run extraction from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int)

    arguments = parser.parse_args()

    failures = process_directory(
        input_directory=arguments.input,
        model_path=arguments.model,
        output_directory=arguments.output,
        workers=arguments.workers,
        limit=arguments.limit,
    )

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

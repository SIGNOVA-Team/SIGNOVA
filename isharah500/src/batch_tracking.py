"""Batch processing for hand-keypoint tracking."""
import argparse
from pathlib import Path
import numpy as np
from hand_tracking import track_hand_slots
def process_directory(input_directory, output_directory):
    """Track all keypoint files and skip existing outputs."""
    input_directory = Path(input_directory)
    output_directory = Path(output_directory)
    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )
    if input_directory.resolve() == output_directory.resolve():
        raise ValueError(
            "Input and output directories must be different."
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    processed_count = 0
    skipped_count = 0
    for input_path in sorted(input_directory.glob("*.npy")):
        output_path = output_directory / input_path.name
        if output_path.exists():
            skipped_count += 1
            continue
        keypoints = np.load(input_path, allow_pickle=False)
        tracked = track_hand_slots(keypoints)
        np.save(output_path, tracked)
        processed_count += 1
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
def main():
    """Parse command-line arguments and run tracking."""
    parser = argparse.ArgumentParser(
        description="Apply temporal tracking to hand keypoints."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    process_directory(arguments.input, arguments.output)
if __name__ == "__main__":
    main()

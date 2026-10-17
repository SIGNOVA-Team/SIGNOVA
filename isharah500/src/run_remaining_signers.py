import subprocess
import sys
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
VIDEOS_DIRECTORY = PROJECT_DIRECTORY / "data/raw/isharah500/videos"
PROCESSED_DIRECTORY = PROJECT_DIRECTORY / "data/processed/isharah500"
MODEL_PATH = PROJECT_DIRECTORY / "models/hand_landmarker.task"
PIPELINE_PATH = PROJECT_DIRECTORY / "src/hand_pipeline.py"

REMAINING_SIGNERS = {
    "train": ("03", "04", "05", "06", "07", "09", "13", "14"),
    "test": ("08", "10", "11", "12"),
}


def find_samples_directory(signer_id):
    signer_directory = VIDEOS_DIRECTORY / signer_id
    for directory in (signer_directory, signer_directory / signer_id):
        if directory.is_dir() and any(
            child.is_dir() and child.name.startswith(f"{signer_id}_")
            for child in directory.iterdir()
        ):
            return directory
    raise FileNotFoundError(f"Sample directories not found for signer {signer_id}")


def process_signer(signer_id, split_name):
    input_directory = find_samples_directory(signer_id)
    command = [
        sys.executable,
        str(PIPELINE_PATH),
        "--input",
        str(input_directory),
        "--model",
        str(MODEL_PATH),
        "--raw-output",
        str(PROCESSED_DIRECTORY / "hand_keypoints" / split_name),
        "--tracked-output",
        str(PROCESSED_DIRECTORY / "hand_keypoints_tracked" / split_name),
        "--workers",
        "4",
    ]

    print(f"\nProcessing signer {signer_id} ({split_name})", flush=True)
    subprocess.run(command, cwd=PROJECT_DIRECTORY, check=True)


def main():
    for split_name, signer_ids in REMAINING_SIGNERS.items():
        for signer_id in signer_ids:
            process_signer(signer_id, split_name)

    print("\nAll remaining signers processed successfully.")


if __name__ == "__main__":
    main()

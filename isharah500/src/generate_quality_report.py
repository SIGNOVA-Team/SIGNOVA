import csv
from pathlib import Path
import numpy as np

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

KEYPOINTS_DIRECTORY = (
    PROJECT_DIRECTORY
    / "data/processed/isharah500/hand_keypoints_tracked"
)

REPORTS_DIRECTORY = PROJECT_DIRECTORY / "reports"
SPLITS = ("train", "dev", "test")
EXPECTED_SAMPLES = 7500
MISSING_HAND_THRESHOLD = 0.30

REPORT_FIELDS = [
    "sample_id",
    "si_split",
    "total_frames",
    "two_hands_ratio",
    "one_hand_ratio",
    "no_hands_ratio",
    "missing_left_slot_ratio",
    "missing_right_slot_ratio",
    "review_required",
]

def calculate_sample_quality(keypoints_path, split_name):
    keypoints = np.load(keypoints_path, allow_pickle=False)

    if keypoints.ndim != 3 or keypoints.shape[1:] != (2, 63):
        raise ValueError(
            f"Unexpected keypoints shape: {keypoints_path}"
        )

    if keypoints.shape[0] == 0:
        raise ValueError(f"Empty sample: {keypoints_path}")

    if keypoints.dtype != np.float32:
        raise ValueError(
            f"Unexpected dtype: {keypoints_path}"
        )

    valid_landmarks = np.isfinite(keypoints)
    detected_hands = valid_landmarks.all(axis=2)
    hand_counts = detected_hands.sum(axis=1)

    two_hands_ratio = np.mean(hand_counts == 2)
    one_hand_ratio = np.mean(hand_counts == 1)
    no_hands_ratio = np.mean(hand_counts == 0)

    missing_left_slot_ratio = np.mean(~detected_hands[:, 0])
    missing_right_slot_ratio = np.mean(~detected_hands[:, 1])

    review_required = (
        no_hands_ratio > MISSING_HAND_THRESHOLD
    )

    return {
        "sample_id": keypoints_path.stem,
        "si_split": split_name,
        "total_frames": keypoints.shape[0],
        "two_hands_ratio": round(float(two_hands_ratio), 4),
        "one_hand_ratio": round(float(one_hand_ratio), 4),
        "no_hands_ratio": round(float(no_hands_ratio), 4),
        "missing_left_slot_ratio": round(
            float(missing_left_slot_ratio), 4
        ),
        "missing_right_slot_ratio": round(
            float(missing_right_slot_ratio), 4
        ),
        "review_required": review_required,
    }

def save_csv(file_path, rows):
    with file_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=REPORT_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)

def generate_quality_report():
    REPORTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    quality_rows = []
    for split_name in SPLITS:
        split_directory = KEYPOINTS_DIRECTORY / split_name
        keypoints_files = sorted(split_directory.glob("*.npy"))

        if not keypoints_files:
            raise FileNotFoundError(
                f"No keypoints found in {split_directory}"
            )

        print(
            f"Processing {split_name}: "
            f"{len(keypoints_files)} samples",
            flush=True,
        )

        for keypoints_path in keypoints_files:
            quality_rows.append(
                calculate_sample_quality(
                    keypoints_path,
                    split_name,
                )
            )

    if len(quality_rows) != EXPECTED_SAMPLES:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLES} samples, "
            f"found {len(quality_rows)}"
        )

    sample_ids = [row["sample_id"] for row in quality_rows]

    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs detected")

    review_rows = [
        row for row in quality_rows
        if row["review_required"]
    ]
    save_csv(
        REPORTS_DIRECTORY / "isharah500_quality_report.csv",
        quality_rows,
    )
    save_csv(
        REPORTS_DIRECTORY / "isharah500_review_required.csv",
        review_rows,
    )
    average_no_hands_ratio = np.mean(
        [row["no_hands_ratio"] for row in quality_rows]
    )
    print("\nQuality Report Completed")
    print(f"Total samples: {len(quality_rows)}")
    print(f"Samples requiring review: {len(review_rows)}")
    print(
        f"Average no-hands ratio: "
        f"{average_no_hands_ratio:.2%}"
    )
    print("Reports saved in: reports/")

if __name__ == "__main__":
    generate_quality_report()

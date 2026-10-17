import csv
from pathlib import Path

from keypoint_validation import is_valid_keypoint_file

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIRECTORY = PROJECT_DIRECTORY / "data/raw/isharah500/annotations/SI"

KEYPOINTS_DIRECTORY = PROJECT_DIRECTORY / "data/processed/isharah500/hand_keypoints_tracked"

METADATA_DIRECTORY = PROJECT_DIRECTORY / "metadata"
SPLITS = {
    "train": 5000,
    "dev": 500,
    "test": 2000,
}
ANNOTATION_FIELDS = ["id", "gloss", "text"]
METADATA_FIELDS = [
    "sample_id",
    "gloss",
    "text",
    "keypoints_path",
]


def prepare_split_metadata(split_name, expected_count):
    annotation_path = ANNOTATIONS_DIRECTORY / f"{split_name}.txt"
    keypoints_directory = KEYPOINTS_DIRECTORY / split_name
    metadata_rows = []
    seen_sample_ids = set()
    with annotation_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file, delimiter="|")
        if reader.fieldnames != ANNOTATION_FIELDS:
            raise ValueError(f"Unexpected columns in {annotation_path.name}: {reader.fieldnames}")
        for row in reader:
            sample_id = row["id"].strip()
            gloss = row["gloss"].strip()
            text = row["text"].strip()
            if not sample_id or not gloss or not text:
                raise ValueError(f"Missing annotation data in {split_name}: {sample_id}")
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample ID in {split_name}: {sample_id}")
            seen_sample_ids.add(sample_id)
            keypoints_path = keypoints_directory / f"{sample_id}.npy"

            if not is_valid_keypoint_file(keypoints_path):
                raise ValueError(f"Invalid keypoints: {keypoints_path}")
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "gloss": gloss,
                    "text": text,
                    "keypoints_path": keypoints_path.relative_to(PROJECT_DIRECTORY).as_posix(),
                }
            )
    available_sample_ids = {path.stem for path in keypoints_directory.glob("*.npy")}
    missing_ids = seen_sample_ids - available_sample_ids
    extra_ids = available_sample_ids - seen_sample_ids
    if missing_ids or extra_ids:
        raise ValueError(
            f"{split_name}: Missing keypoints={len(missing_ids)}, Extra keypoints={len(extra_ids)}"
        )
    if len(metadata_rows) != expected_count:
        raise ValueError(
            f"{split_name}: Expected {expected_count} samples, found {len(metadata_rows)}"
        )
    METADATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path = METADATA_DIRECTORY / f"isharah500_si_{split_name}.csv"
    with output_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=METADATA_FIELDS,
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"{split_name}: {len(metadata_rows)} samples -> {output_path.name}")

    return seen_sample_ids


def main():
    split_sample_ids = {}
    for split_name, expected_count in SPLITS.items():
        split_sample_ids[split_name] = prepare_split_metadata(
            split_name,
            expected_count,
        )
    for index, first_split in enumerate(SPLITS):
        for second_split in list(SPLITS)[index + 1 :]:
            overlapping_ids = split_sample_ids[first_split] & split_sample_ids[second_split]
            if overlapping_ids:
                raise ValueError(
                    f"Overlapping samples between "
                    f"{first_split} and {second_split}: "
                    f"{len(overlapping_ids)}"
                )
    print("\nMetadata preparation completed successfully.")
    print("Total samples: 7500")
    print("No missing, extra, or overlapping sample IDs.")


if __name__ == "__main__":
    main()

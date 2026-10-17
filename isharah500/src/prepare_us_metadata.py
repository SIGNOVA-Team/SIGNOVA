import csv
from pathlib import Path

from keypoint_validation import is_valid_keypoint_file

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent

ANNOTATIONS_DIRECTORY = PROJECT_DIRECTORY / "data/raw/isharah500/annotations/US"

KEYPOINTS_DIRECTORY = PROJECT_DIRECTORY / "data/processed/isharah500/hand_keypoints_tracked"

METADATA_DIRECTORY = PROJECT_DIRECTORY / "metadata"

SPLITS = ("train", "dev", "test")
ANNOTATION_FIELDS = ["id", "gloss", "text"]
METADATA_FIELDS = [
    "sample_id",
    "gloss",
    "text",
    "keypoints_path",
]


def build_keypoints_index():
    keypoints_index = {}
    for split_name in SPLITS:
        split_directory = KEYPOINTS_DIRECTORY / split_name

        for keypoints_path in split_directory.glob("*.npy"):
            sample_id = keypoints_path.stem

            if sample_id in keypoints_index:
                raise ValueError(f"Duplicate keypoints sample: {sample_id}")

            keypoints_index[sample_id] = keypoints_path
    if len(keypoints_index) != 7500:
        raise ValueError(f"Expected 7500 keypoint files, found {len(keypoints_index)}")

    return keypoints_index


def prepare_split(split_name, keypoints_index):
    annotation_path = ANNOTATIONS_DIRECTORY / f"{split_name}.txt"

    metadata_rows = []
    sample_ids = set()

    with annotation_path.open(
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
                raise ValueError(f"Missing annotation data: {sample_id}")

            if sample_id in sample_ids:
                raise ValueError(f"Duplicate sample ID in {split_name}: {sample_id}")
            sample_ids.add(sample_id)

            keypoints_path = keypoints_index.get(sample_id)

            if keypoints_path is None:
                raise ValueError(f"Keypoints not found for {sample_id}")

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

    return metadata_rows, sample_ids


def save_metadata(split_name, metadata_rows):
    METADATA_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = METADATA_DIRECTORY / f"isharah500_us_{split_name}.csv"

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


def main():
    keypoints_index = build_keypoints_index()
    all_sample_ids = set()

    for split_name in SPLITS:
        metadata_rows, sample_ids = prepare_split(
            split_name,
            keypoints_index,
        )
        overlapping_ids = all_sample_ids & sample_ids

        if overlapping_ids:
            raise ValueError(f"Overlapping samples in {split_name}: {len(overlapping_ids)}")

        all_sample_ids.update(sample_ids)
        save_metadata(split_name, metadata_rows)
    missing_ids = set(keypoints_index) - all_sample_ids

    if missing_ids:
        raise ValueError(f"US annotations are missing {len(missing_ids)} samples")
    print("\nUS metadata preparation completed successfully.")
    print(f"Total samples: {len(all_sample_ids)}")
    print("No missing or overlapping sample IDs.")


if __name__ == "__main__":
    main()

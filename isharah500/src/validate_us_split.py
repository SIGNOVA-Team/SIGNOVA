import csv
from pathlib import Path

PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
METADATA_DIRECTORY = PROJECT_DIRECTORY / "metadata"

SPLITS = ("train", "dev", "test")
TEXT_FIELDS = ("gloss", "text")

def load_split(split_name):
    metadata_path = (
        METADATA_DIRECTORY / f"isharah500_us_{split_name}.csv"
    )
    with metadata_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)
        required_fields = {"sample_id", *TEXT_FIELDS}

        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Missing required columns in {metadata_path.name}"
            )
        rows = list(reader)

    return rows

def normalize_sentence(sentence):
    return " ".join(sentence.split())

def validate_sentences(splits_data, field_name):
    print(f"\nChecking {field_name} overlap:")

    sentence_sets = {
        split_name: {
            normalize_sentence(row[field_name])
            for row in rows
        }
        for split_name, rows in splits_data.items()
    }

    for index, first_split in enumerate(SPLITS):
        for second_split in SPLITS[index + 1:]:
            overlap = (
                sentence_sets[first_split]
                & sentence_sets[second_split]
            )

            print(
                f"{first_split} vs {second_split}: "
                f"{len(overlap)} overlapping sentences"
            )

            if overlap:
                print(
                    f"Example: {next(iter(sorted(overlap)))}"
                )

def main():
    splits_data = {
        split_name: load_split(split_name)
        for split_name in SPLITS
    }
    for split_name, rows in splits_data.items():
        print(f"{split_name}: {len(rows)} samples")

    for field_name in TEXT_FIELDS:
        validate_sentences(splits_data, field_name)

if __name__ == "__main__":
    main()

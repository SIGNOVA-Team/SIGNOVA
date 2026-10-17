import argparse
from pathlib import Path
import pandas as pd

COLUMNS = [
    "sample_id", "signer_id", "session_id", "sign_id",
    "author_split", "qc_status", "loso_test_S01",
    "loso_test_S02", "loso_test_S03",
    "split_version", "fold", "split"
]

FOLDS = [
    ("fold_1", "S03", "S02", "S01"),
    ("fold_2", "S01", "S03", "S02"),
    ("fold_3", "S02", "S01", "S03")
]

def generate(metadata):
    columns = [
        "sample_id", "signer_id", "session_id",
        "class_label", "author_split", "qc_status", "usable"
    ]

    df = pd.read_csv(metadata, usecols=columns, low_memory=False)

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("Invalid sample IDs")

    usable = df["usable"].astype(str).str.lower().eq("true")
    df = df[df["qc_status"].eq("PASS") & usable].copy()

    coverage = df.groupby("class_label")["signer_id"].nunique()
    common_classes = coverage[coverage == 3].index

    df = df[df["class_label"].isin(common_classes)].copy()
    df = df.rename(columns={"class_label": "sign_id"})

    for signer in ["S01", "S02", "S03"]:
        df["loso_test_" + signer] = df["signer_id"].map(
            lambda value: "test" if value == signer else "train"
        )

    results = []

    for fold, train, validation, test in FOLDS:
        part = df.copy()
        part["fold"] = fold
        part["split"] = part["signer_id"].map({
            train: "train",
            validation: "validation",
            test: "test"
        })
        part["split_version"] = "v3_common_classes"
        results.append(part[COLUMNS])

    return pd.concat(results, ignore_index=True)

def validate(generated, reference):
    if list(generated.columns) != list(reference.columns):
        raise ValueError("Schema mismatch")

    keys = ["sample_id", "fold"]

    if generated.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("Duplicate sample-fold keys")

    merged = generated.merge(
        reference,
        on=keys,
        how="outer",
        suffixes=("_new", "_old"),
        indicator=True,
        validate="one_to_one"
    )

    missing = int((merged["_merge"] != "both").sum())
    differences = {}

    for column in COLUMNS:
        if column in keys:
            continue

        left = merged[column + "_new"].fillna("<NA>").astype(str)
        right = merged[column + "_old"].fillna("<NA>").astype(str)
        differences[column] = int((left != right).sum())

    passed = (
        missing == 0
        and len(generated) == len(reference)
        and all(value == 0 for value in differences.values())
    )

    return passed, missing, differences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    generated = generate(
        args.root / "data/metadata/karsl_metadata.csv"
    )

    reference = pd.read_csv(
        args.root / "data/splits/karsl_split_manifest.csv",
        low_memory=False
    )

    passed, missing, differences = validate(generated, reference)

    print("=== SPLIT GENERATION ===")
    print("Generated rows:", len(generated))
    print("Unique samples:", generated["sample_id"].nunique())
    print("Classes:", generated["sign_id"].nunique())
    print("Folds:", generated["fold"].nunique())
    print("Reference rows:", len(reference))
    print("Unmatched sample-fold keys:", missing)

    print("\n=== COLUMN COMPARISON ===")
    for name, count in differences.items():
        print(name, ":", count)

    print("\nExact manifest match:", "PASS" if passed else "FAIL")

    if not passed:
        raise SystemExit(1)

    args.output.mkdir(parents=True, exist_ok=True)
    generated.to_csv(
        args.output / "generated_karsl_split_manifest.csv",
        index=False
    )

    print("Original manifest modified: NO")

if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import pandas as pd


def generate(source, output):
    columns = [
        "sample_id",
        "class_label",
        "signer_id",
        "qc_status",
        "qc_flags",
        "qc_reason",
        "usable",
    ]

    df = pd.read_csv(source, usecols=columns, low_memory=False)

    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("Invalid or duplicate sample IDs")

    allowed = {"PASS", "REVIEW", "EXCLUDE"}
    if not set(df["qc_status"].dropna()).issubset(allowed):
        raise ValueError("Unexpected QC status")

    if df["qc_status"].isna().any():
        raise ValueError("Missing QC status")

    df["qc_flags"] = df["qc_flags"].fillna("")
    df["qc_reason"] = df["qc_reason"].fillna("")

    flagged = df[
        (df["qc_status"] != "PASS") | (df["qc_flags"] != "") | (df["qc_reason"] != "")
    ].copy()

    review = df[df["qc_status"] == "REVIEW"].copy()

    summary = (
        df.groupby("qc_status")
        .agg(
            samples=("sample_id", "count"),
            classes=("class_label", "nunique"),
            signers=("signer_id", "nunique"),
        )
        .reset_index()
    )

    output.mkdir(parents=True, exist_ok=True)

    flagged.to_csv(output / "generated_qc_flags.csv", index=False)
    review.to_csv(output / "generated_review_cases.csv", index=False)
    summary.to_csv(output / "generated_qc_summary.csv", index=False)

    print("=== QC GENERATION ===")
    print("Total samples:", len(df))
    print("Unique samples:", df["sample_id"].nunique())
    print("Flagged samples:", len(flagged))
    print("Review samples:", len(review))
    print(summary.to_string(index=False))
    print("Summary matches total:", summary["samples"].sum() == len(df))
    print("Status: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.output)


if __name__ == "__main__":
    main()

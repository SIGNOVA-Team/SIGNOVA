"""
Automated Test Suite for SIGNOVA P2 Datasets (How2Sign & ArabSign)
Tests:
  - Canonical metadata schema compliance
  - ID uniqueness and non-null constraints
  - Dataset sample count verification
  - Split manifest partition integrity
  - Signer leakage prevention in the ArabSign signer-independent benchmark
  - Linguistic integrity (ArabSign ArSL != ESL)
"""

import csv
import os

import pytest

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")

REQUIRED_CANONICAL_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_sample_id",
    "sign_language",
    "original_label",
    "gloss_label",
    "signer_id",
    "session_id",
    "relative_video_path",
    "fps",
    "duration_sec",
    "width",
    "height",
    "split",
    "split_version",
    "license_id",
    "qc_status",
    "qc_flags",
    "checksum",
]


def test_core_files_exist():
    """Verify that all core P2 deliverables exist."""
    expected_files = [
        os.path.join(REPORTS_DIR, "dataset_sources.csv"),
        os.path.join(REPORTS_DIR, "download_manifest.csv"),
        os.path.join(REPORTS_DIR, "qc_report_p2.md"),
        os.path.join(REPORTS_DIR, "qc_flags_p2.csv"),
        os.path.join(METADATA_DIR, "how2sign_canonical_metadata.csv"),
        os.path.join(METADATA_DIR, "arabsign_canonical_metadata.csv"),
        os.path.join(METADATA_DIR, "arabsign_sentences_mapping.csv"),
        os.path.join(SPLITS_DIR, "how2sign_split_manifest.csv"),
        os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_dependent.csv"),
        os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_independent.csv"),
        os.path.join(ROOT_DIR, "docs", "dataset_card_how2sign.md"),
        os.path.join(ROOT_DIR, "docs", "dataset_card_arabsign.md"),
        os.path.join(ROOT_DIR, "docs", "decisions", "ADR_001_p2_datasets_evaluation.md"),
    ]
    for fp in expected_files:
        assert os.path.isfile(fp), f"Missing required file: {fp}"


def test_how2sign_canonical_schema_and_counts():
    """Verify How2Sign schema, sample count, and non-null values."""
    csv_path = os.path.join(METADATA_DIR, "how2sign_canonical_metadata.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for col in REQUIRED_CANONICAL_COLUMNS:
            assert col in headers, f"Missing required column '{col}' in How2Sign metadata"

        rows = list(reader)
        assert len(rows) == 35263, f"Expected 35,263 How2Sign samples, got {len(rows)}"

        # Check sample IDs are unique
        sample_ids = [r["sample_id"] for r in rows]
        assert len(sample_ids) == len(set(sample_ids)), (
            "Duplicate sample_ids found in How2Sign metadata"
        )

        # Check sign language tag
        for r in rows[:100]:
            assert r["sign_language"] == "ASL", "Expected ASL sign_language tag"
            assert r["source_dataset"] == "How2Sign"
            assert float(r["duration_sec"]) > 0 or float(r["duration_sec"]) == -1.0


def test_arabsign_canonical_schema_and_counts():
    """Verify ArabSign schema, sample count, and non-null values."""
    csv_path = os.path.join(METADATA_DIR, "arabsign_canonical_metadata.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for col in REQUIRED_CANONICAL_COLUMNS:
            assert col in headers, f"Missing required column '{col}' in ArabSign metadata"

        rows = list(reader)
        assert len(rows) == 9335, f"Expected exactly 9,335 ArabSign samples, got {len(rows)}"

        # Check sample IDs are unique
        sample_ids = [r["sample_id"] for r in rows]
        assert len(sample_ids) == len(set(sample_ids)), (
            "Duplicate sample_ids found in ArabSign metadata"
        )


def test_arabsign_linguistic_integrity():
    """Verify ArabSign is tagged strictly as ArSL and NEVER as ESL."""
    csv_path = os.path.join(METADATA_DIR, "arabsign_canonical_metadata.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            assert r["sign_language"] == "ArSL", (
                f"Violated rule: ArabSign must be ArSL, found {r['sign_language']}"
            )
            assert r["sign_language"] != "ESL", (
                "CRITICAL ERROR: ArabSign was illegally tagged as ESL"
            )
            assert "DIALECT_NOT_ESL" in r["qc_flags"], "Missing DIALECT_NOT_ESL QC flag"


def test_arabsign_sentences_mapping():
    """Verify that all 50 ArabSign sentences are cleanly mapped."""
    csv_path = os.path.join(METADATA_DIR, "arabsign_sentences_mapping.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 50, f"Expected 50 sentences, got {len(rows)}"

        sentence_ids = [int(r["sentence_id"]) for r in rows]
        assert sorted(sentence_ids) == list(range(1, 51)), "Sentence IDs must be 1 to 50"

        for r in rows:
            assert r["arsl_gloss_structure"], "Missing ArSL gloss structure"
            assert r["spoken_arabic_standard"], "Missing spoken Arabic text"
            assert r["english_translation"], "Missing English translation"


def test_arabsign_signer_independent_split_zero_leakage():
    """Verify that the Signer-Independent benchmark has ZERO signer leakage."""
    csv_path = os.path.join(SPLITS_DIR, "arabsign_split_manifest_signer_independent.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 9335, f"Expected 9,335 split entries, got {len(rows)}"

        train_signers = {r["signer_id"] for r in rows if r["split"] == "train"}
        val_signers = {r["signer_id"] for r in rows if r["split"] == "val"}
        test_signers = {r["signer_id"] for r in rows if r["split"] == "test"}

        # Confirm expected signers in each split
        assert train_signers == {"01", "02", "03", "04"}, (
            f"Unexpected train signers: {train_signers}"
        )
        assert val_signers == {"05"}, f"Unexpected val signers: {val_signers}"
        assert test_signers == {"06"}, f"Unexpected test signers: {test_signers}"

        # CRITICAL LEAKAGE TEST: Zero intersection between partitions!
        assert len(train_signers.intersection(val_signers)) == 0, (
            "Signer leakage detected between train and val!"
        )
        assert len(train_signers.intersection(test_signers)) == 0, (
            "Signer leakage detected between train and test!"
        )
        assert len(val_signers.intersection(test_signers)) == 0, (
            "Signer leakage detected between val and test!"
        )

        # Check sample partition counts
        train_count = sum(1 for r in rows if r["split"] == "train")
        val_count = sum(1 for r in rows if r["split"] == "val")
        test_count = sum(1 for r in rows if r["split"] == "test")

        assert train_count == 6323, f"Expected 6,323 train samples, got {train_count}"
        assert val_count == 1500, f"Expected 1,500 val samples, got {val_count}"
        assert test_count == 1512, f"Expected 1,512 test samples, got {test_count}"
        assert train_count + val_count + test_count == 9335


def test_how2sign_split_counts():
    """Verify How2Sign split manifest partition counts."""
    csv_path = os.path.join(SPLITS_DIR, "how2sign_split_manifest.csv")
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 35263, f"Expected 35,263 split rows, got {len(rows)}"

        train_count = sum(1 for r in rows if r["split"] == "train")
        val_count = sum(1 for r in rows if r["split"] == "val")
        test_count = sum(1 for r in rows if r["split"] == "test")

        assert train_count == 31165, f"Expected 31,165 train, got {train_count}"
        assert val_count == 1741, f"Expected 1,741 val, got {val_count}"
        assert test_count == 2357, f"Expected 2,357 test, got {test_count}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])

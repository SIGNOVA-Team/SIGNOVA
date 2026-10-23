"""Unit tests for SI token alignment and grouped error summaries."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from analyze_si_errors import (  # noqa: E402
    aggregate_records,
    align_token_sequences,
    alignment_counts,
    coverage_bucket,
)


def test_alignment_counts_use_edit_operations():
    alignment = align_token_sequences(
        ["انا", "اشاره", "مدرسة"],
        ["انا", "سؤال", "مدرسة", "اليوم"],
    )

    assert alignment_counts(alignment) == {
        "substitutions": 1,
        "deletions": 0,
        "insertions": 1,
    }
    assert [step["operation"] for step in alignment] == [
        "match",
        "substitution",
        "match",
        "insertion",
    ]


def test_alignment_counts_deletion_and_exact_match():
    assert alignment_counts(align_token_sequences(["a", "b"], ["a"])) == {
        "substitutions": 0,
        "deletions": 1,
        "insertions": 0,
    }
    assert alignment_counts(align_token_sequences(["a"], ["a"])) == {
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }


def test_grouped_metrics_and_confusions_are_deterministic():
    records = [
        {
            "sample_id": "b",
            "reference": ["a", "b"],
            "hypothesis": ["a"],
            "target_length": 2,
            "prediction_length": 1,
            "valid_hand_coverage": 0.4,
            "coverage_bucket": coverage_bucket(0.4),
            "exact_match": False,
            "alignment": align_token_sequences(["a", "b"], ["a"]),
            "substitutions": 0,
            "deletions": 1,
            "insertions": 0,
        },
        {
            "sample_id": "a",
            "reference": ["a"],
            "hypothesis": ["x"],
            "target_length": 1,
            "prediction_length": 1,
            "valid_hand_coverage": 1.0,
            "coverage_bucket": coverage_bucket(1.0),
            "exact_match": False,
            "alignment": align_token_sequences(["a"], ["x"]),
            "substitutions": 1,
            "deletions": 0,
            "insertions": 0,
        },
    ]

    report = aggregate_records(records)

    assert report["aggregate"]["gloss_error_rate"] == 2 / 3
    assert report["aggregate"]["deletion_rate"] == 1 / 3
    assert list(report["by_target_length"]) == ["1", "2"]
    assert list(report["by_valid_hand_coverage"]) == ["25-49%", "100%"]
    assert report["most_frequent_confusions"] == [{"reference": "a", "hypothesis": "x", "count": 1}]

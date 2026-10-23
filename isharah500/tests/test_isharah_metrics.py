"""Regression tests for canonical token metrics."""

import sys
from pathlib import Path

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_metrics import calculate_token_metrics  # noqa: E402


def test_metrics_are_batch_size_invariant_with_partial_batch():
    references = [
        ["انا", "انا"],
        ["اشاره"],
        [],
        ["مدرسة", "لغة", "إشارة"],
        ["اليوم"],
    ]
    hypotheses = [
        ["انا"],
        ["سؤال"],
        ["زيادة"],
        ["مدرسة", "إشارة"],
        ["اليوم"],
    ]

    full = calculate_token_metrics(references, hypotheses)
    accumulated = {
        "sample_count": 0,
        "reference_tokens": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
        "edit_errors": 0,
        "exact_matches": 0,
    }
    for start in range(0, len(references), 2):
        batch = calculate_token_metrics(
            references[start : start + 2],
            hypotheses[start : start + 2],
        )
        for key in accumulated:
            accumulated[key] += batch[key]

    assert accumulated == {key: full[key] for key in accumulated}
    assert full["gloss_error_rate"] == full["edit_errors"] / full["reference_tokens"]
    assert full["exact_match"] == full["exact_matches"] / full["sample_count"]


def test_repeated_glosses_are_exact_only_when_the_whole_sequence_matches():
    exact = calculate_token_metrics([["a", "a"]], [["a", "a"]])
    collapsed = calculate_token_metrics([["a", "a"]], [["a"]])

    assert exact["exact_matches"] == 1
    assert exact["edit_errors"] == 0
    assert collapsed["exact_matches"] == 0
    assert collapsed["deletions"] == 1

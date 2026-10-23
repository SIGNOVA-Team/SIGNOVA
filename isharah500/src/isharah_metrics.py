"""Canonical token-level metrics for Isharah-500 evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def align_token_sequences(
    reference: Sequence[str],
    hypothesis: Sequence[str],
) -> list[dict[str, str | None]]:
    """Return a deterministic minimum-cost token alignment.

    Ties prefer diagonal match/substitution, then deletion, then insertion.
    """
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(1, rows):
        distances[row][0] = row
    for column in range(1, columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            distances[row][column] = min(
                distances[row - 1][column - 1]
                + int(reference[row - 1] != hypothesis[column - 1]),
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
            )

    alignment: list[dict[str, str | None]] = []
    row = len(reference)
    column = len(hypothesis)
    while row > 0 or column > 0:
        if (
            row > 0
            and column > 0
            and distances[row][column]
            == distances[row - 1][column - 1]
            + int(reference[row - 1] != hypothesis[column - 1])
        ):
            alignment.append(
                {
                    "operation": (
                        "match"
                        if reference[row - 1] == hypothesis[column - 1]
                        else "substitution"
                    ),
                    "reference": reference[row - 1],
                    "hypothesis": hypothesis[column - 1],
                }
            )
            row -= 1
            column -= 1
        elif row > 0 and distances[row][column] == distances[row - 1][column] + 1:
            alignment.append(
                {
                    "operation": "deletion",
                    "reference": reference[row - 1],
                    "hypothesis": None,
                }
            )
            row -= 1
        else:
            alignment.append(
                {
                    "operation": "insertion",
                    "reference": None,
                    "hypothesis": hypothesis[column - 1],
                }
            )
            column -= 1

    alignment.reverse()
    return alignment


def alignment_counts(alignment: Sequence[dict[str, str | None]]) -> dict[str, int]:
    """Count substitution, deletion, and insertion operations."""
    counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
    operation_keys = {
        "substitution": "substitutions",
        "deletion": "deletions",
        "insertion": "insertions",
    }
    for step in alignment:
        operation = step["operation"]
        if operation in operation_keys:
            counts[operation_keys[operation]] += 1
        elif operation != "match":
            raise ValueError(f"Unknown alignment operation: {operation}")
    return counts


def calculate_token_metrics(
    references: Sequence[Sequence[str]],
    hypotheses: Sequence[Sequence[str]],
) -> dict[str, Any]:
    """Calculate sample-denominated exact match and token-denominated GER."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal lengths")

    substitutions = 0
    deletions = 0
    insertions = 0
    reference_tokens = 0
    exact_matches = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        counts = alignment_counts(align_token_sequences(reference, hypothesis))
        substitutions += counts["substitutions"]
        deletions += counts["deletions"]
        insertions += counts["insertions"]
        reference_tokens += len(reference)
        exact_matches += int(list(reference) == list(hypothesis))

    errors = substitutions + deletions + insertions
    sample_count = len(references)
    return {
        "sample_count": sample_count,
        "reference_tokens": reference_tokens,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_errors": errors,
        "gloss_error_rate": errors / max(reference_tokens, 1),
        "exact_matches": exact_matches,
        "exact_match": exact_matches / max(sample_count, 1),
    }

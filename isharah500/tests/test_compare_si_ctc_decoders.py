"""Focused tests for the read-only SI decoder comparison helpers."""

import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from compare_si_ctc_decoders import (  # noqa: E402
    _report_progress,
    _select_dev_subset,
)


def test_select_dev_subset_uses_deterministic_prefix():
    dataset = list(range(10))

    subset = _select_dev_subset(dataset, 3)

    assert len(subset) == 3
    assert [subset[index] for index in range(len(subset))] == [0, 1, 2]


def test_select_dev_subset_defaults_to_all_samples():
    dataset = list(range(4))

    assert _select_dev_subset(dataset, None) is dataset


def test_progress_is_written_to_stderr_only(capsys):
    _report_progress(5, 20, time.monotonic() - 1.0)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "processed=5/20" in captured.err
    assert "elapsed=" in captured.err

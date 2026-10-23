"""Focused tests for the read-only SI blank-offset calibration helpers."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import greedy_ctc_decode  # noqa: E402
from isharah_dataset import GlossVocabulary  # noqa: E402
from calibrate_si_blank_offset import apply_blank_logit_offset  # noqa: E402


def _vocabulary() -> GlossVocabulary:
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha", "beta"])
    return vocabulary


def test_blank_offset_changes_only_blank_logits_without_mutating_input():
    logits = torch.zeros(1, 2, 3)
    original = logits.clone()

    calibrated = apply_blank_logit_offset(logits, blank_id=0, offset=-2.5)

    assert torch.equal(logits, original)
    assert torch.equal(calibrated[..., 1:], logits[..., 1:])
    assert torch.equal(calibrated[..., 0], torch.full((1, 2), -2.5))


def test_zero_offset_preserves_existing_greedy_decoder_behavior():
    vocabulary = _vocabulary()
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    alpha = vocabulary.token2id["alpha"]
    logits = torch.full((1, 5, len(vocabulary)), -10.0)
    logits[0, :, blank_id] = 1.0
    logits[0, 1, alpha] = 5.0
    logits[0, 3, alpha] = 5.0

    baseline = greedy_ctc_decode(
        logits, torch.tensor([5]), vocabulary, blank_id=blank_id
    )
    calibrated = greedy_ctc_decode(
        apply_blank_logit_offset(logits, blank_id, 0.0),
        torch.tensor([5]),
        vocabulary,
        blank_id=blank_id,
    )

    assert baseline == calibrated == ["alpha alpha"]


def test_negative_blank_offset_can_reveal_repeated_tokens_after_blank():
    vocabulary = _vocabulary()
    blank_id = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    alpha = vocabulary.token2id["alpha"]
    logits = torch.full((1, 5, len(vocabulary)), -3.0)
    logits[0, :, blank_id] = 2.2
    logits[0, 1, alpha] = 2.1
    logits[0, 3, alpha] = 2.1

    baseline = greedy_ctc_decode(
        logits, torch.tensor([5]), vocabulary, blank_id=blank_id
    )
    calibrated = greedy_ctc_decode(
        apply_blank_logit_offset(logits, blank_id, -1.0),
        torch.tensor([5]),
        vocabulary,
        blank_id=blank_id,
    )

    assert baseline == [""]
    assert calibrated == ["alpha alpha"]


@pytest.mark.parametrize("bad_offset", [float("nan"), float("inf")])
def test_blank_offset_rejects_nonfinite_values(bad_offset):
    with pytest.raises(ValueError, match="offset must be finite"):
        apply_blank_logit_offset(torch.zeros(1, 1, 2), 0, bad_offset)

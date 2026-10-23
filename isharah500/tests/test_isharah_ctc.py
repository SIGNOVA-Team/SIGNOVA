"""Tests for the Isharah-500 BiGRU + CTC baseline."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SOURCE_DIRECTORY = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import (  # noqa: E402
    BiGRUCTC,
    BiGRUCTCConfig,
    collapse_ctc_path,
    ctc_loss_from_logits,
    ctc_prefix_beam_search,
    greedy_ctc_decode,
)
from isharah_dataset import GlossVocabulary  # noqa: E402


def make_model(vocabulary_size):
    config = BiGRUCTCConfig(
        projection_size=16,
        hidden_size=8,
        num_layers=2,
        dropout=0.1,
    )
    return BiGRUCTC(vocabulary_size, config=config)


def test_forward_supports_keypoint_shape_and_masks():
    """Forward returns finite CTC logits for masked hand keypoints."""
    torch.manual_seed(7)
    model = make_model(9).eval()
    keypoints = torch.randn(2, 6, 2, 63)
    masks = torch.ones(2, 6, 2, dtype=torch.bool)
    masks[0, 2, 1] = False
    lengths = torch.tensor([6, 4], dtype=torch.long)

    logits = model(keypoints, lengths, masks)

    assert logits.shape == (2, 6, 9)
    assert torch.isfinite(logits).all()


def test_forward_respects_variable_lengths():
    """Packed recurrent processing keeps the padded tail out of valid output."""
    torch.manual_seed(11)
    model = make_model(7).eval()
    keypoints = torch.randn(2, 5, 2, 63)
    lengths = torch.tensor([5, 2], dtype=torch.long)

    logits = model(keypoints, lengths)

    assert logits.shape[:2] == (2, 5)
    assert torch.allclose(logits[1, 2:], torch.zeros_like(logits[1, 2:]))


def test_ctc_loss_from_logits():
    """CTC loss accepts concatenated targets and explicit lengths."""
    torch.manual_seed(13)
    logits = torch.randn(2, 7, 6, requires_grad=True)
    input_lengths = torch.tensor([7, 5], dtype=torch.long)
    target_ids = torch.tensor([1, 2, 3], dtype=torch.long)
    target_lengths = torch.tensor([2, 1], dtype=torch.long)

    loss = ctc_loss_from_logits(
        logits,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_standard_ctc_loss_is_unchanged_by_explicit_none_mode():
    """The default loss remains identical to the explicit standard mode."""
    torch.manual_seed(19)
    logits = torch.randn(3, 8, 6)
    input_lengths = torch.tensor([8, 7, 6], dtype=torch.long)
    target_ids = torch.tensor([1, 2, 1, 3, 2, 4], dtype=torch.long)
    target_lengths = torch.tensor([1, 2, 3], dtype=torch.long)

    default_loss = ctc_loss_from_logits(
        logits,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
    )
    explicit_standard_loss = ctc_loss_from_logits(
        logits,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
        target_length_weighting="none",
        target_length_weight_cap=2.0,
    )

    assert torch.equal(default_loss, explicit_standard_loss)


def test_weighted_ctc_uses_capped_normalized_variable_length_weights():
    """Weighted CTC matches its formula and produces finite gradients."""
    torch.manual_seed(23)
    logits = torch.randn(3, 8, 6, requires_grad=True)
    input_lengths = torch.tensor([8, 7, 6], dtype=torch.long)
    target_ids = torch.tensor([1, 2, 1, 3, 2, 4], dtype=torch.long)
    target_lengths = torch.tensor([1, 2, 3], dtype=torch.long)
    cap = 1.25

    loss = ctc_loss_from_logits(
        logits,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
        target_length_weighting="linear",
        target_length_weight_cap=cap,
    )
    loss.backward()

    log_probs = logits.detach().log_softmax(dim=-1).transpose(0, 1)
    per_sample = torch.nn.functional.ctc_loss(
        log_probs,
        target_ids,
        input_lengths,
        target_lengths,
        blank=0,
        reduction="none",
        zero_infinity=True,
    )
    raw_weights = (target_lengths.float() / target_lengths.float().mean()).clamp(max=cap)
    weights = raw_weights / raw_weights.mean()
    normalized_per_sample = per_sample / target_lengths.float()
    expected = (normalized_per_sample * weights).mean()

    assert torch.allclose(loss.detach(), expected)
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert raw_weights.max() <= cap


def test_weighted_unit_weights_match_standard_ctc_loss_and_gradients():
    """Unit target-length weights preserve standard mean loss and gradients."""
    torch.manual_seed(29)
    logits_standard = torch.randn(3, 8, 6, requires_grad=True)
    logits_weighted = logits_standard.detach().clone().requires_grad_(True)
    input_lengths = torch.tensor([8, 7, 6], dtype=torch.long)
    target_ids = torch.tensor([1, 2, 1, 3, 2, 4], dtype=torch.long)
    target_lengths = torch.tensor([2, 2, 2], dtype=torch.long)

    standard = ctc_loss_from_logits(
        logits_standard,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
    )
    weighted = ctc_loss_from_logits(
        logits_weighted,
        input_lengths,
        target_ids,
        target_lengths,
        blank_id=0,
        target_length_weighting="linear",
        target_length_weight_cap=2.0,
    )
    standard.backward()
    weighted.backward()

    assert torch.allclose(weighted, standard)
    assert torch.allclose(logits_weighted.grad, logits_standard.grad)


def test_weighted_ctc_rejects_nonpositive_target_lengths():
    """Weighted mode requires valid positive target lengths for weighting."""
    with pytest.raises(ValueError, match="positive target lengths"):
        ctc_loss_from_logits(
            torch.randn(2, 5, 4),
            torch.tensor([5, 5]),
            torch.tensor([1]),
            torch.tensor([1, 0]),
            blank_id=0,
            target_length_weighting="sqrt",
        )


def test_ctc_loss_rejects_inconsistent_target_lengths():
    """Flat target IDs and target lengths must agree."""
    with pytest.raises(ValueError):
        ctc_loss_from_logits(
            torch.randn(1, 4, 5),
            torch.tensor([4]),
            torch.tensor([1, 2]),
            torch.tensor([1]),
            blank_id=0,
        )


def test_greedy_ctc_decode_collapses_repeats_and_blanks():
    """Greedy decoding uses input lengths and the vocabulary blank ID."""
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha", "beta"])
    alpha = vocabulary.token2id["alpha"]
    beta = vocabulary.token2id["beta"]
    blank = vocabulary.token2id[vocabulary.BLANK_TOKEN]

    path = [alpha, alpha, blank, beta, beta, blank]
    assert collapse_ctc_path(path, blank) == [alpha, beta]

    logits = torch.full((2, 6, len(vocabulary)), -10.0)
    for time, token_id in enumerate(path):
        logits[0, time, token_id] = 10.0
    logits[1, :3, alpha] = 10.0
    logits[1, 3:, beta] = 10.0

    decoded = greedy_ctc_decode(logits, torch.tensor([6, 3]), vocabulary)

    assert decoded == ["alpha beta", "alpha"]


def test_ctc_prefix_beam_search_preserves_repeated_tokens_after_blank():
    """A blank-separated repeated token must remain repeated."""
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha", "beta"])
    alpha = vocabulary.token2id["alpha"]
    blank = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    logits = torch.full((1, 3, len(vocabulary)), -20.0)
    logits[0, 0, alpha] = 10.0
    logits[0, 1, blank] = 10.0
    logits[0, 2, alpha] = 10.0

    decoded = ctc_prefix_beam_search(
        logits,
        torch.tensor([3], dtype=torch.long),
        vocabulary,
        beam_width=5,
    )

    assert decoded == [[alpha, alpha]]


def test_ctc_prefix_beam_search_decodes_blank_only_path():
    """A blank-only path must produce an empty prefix."""
    vocabulary = GlossVocabulary()
    vocabulary.build_vocab(["alpha"])
    blank = vocabulary.token2id[vocabulary.BLANK_TOKEN]
    logits = torch.full((1, 4, len(vocabulary)), -20.0)
    logits[:, :, blank] = 10.0

    decoded = ctc_prefix_beam_search(
        logits,
        torch.tensor([4], dtype=torch.long),
        vocabulary,
        beam_width=10,
    )

    assert decoded == [[]]

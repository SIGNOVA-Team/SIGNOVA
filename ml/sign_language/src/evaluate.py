"""
SIGNOVA — Greedy and Prefix Beam Search Decoders and Evaluation Metrics.

Computes:
- Word Error Rate (WER)
- Token Accuracy
- Character Error Rate (CER)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ml.sign_language.src.preprocessing import SignVocabulary


def ctc_greedy_decode(
    log_probs: torch.Tensor,
    blank_id: int = 0,
) -> list[list[int]]:
    """
    Greedy best-path decoding for CTC log-probabilities.

    Args:
        log_probs: Tensor of shape (Time, Batch, Num_Classes) or (Batch, Time, Num_Classes).
        blank_id: Index of CTC blank token (typically 0).

    Returns:
        List of decoded token ID lists for each sequence in the batch.
    """
    if log_probs.dim() != 3:
        raise ValueError(f"Expected 3D log_probs, got shape {log_probs.shape}")

    # Ensure shape is (Batch, Time, Num_Classes)
    if log_probs.shape[1] < log_probs.shape[0]:
        # Likely (Time, Batch, Classes)
        batch_first_probs = log_probs.transpose(0, 1)
    else:
        batch_first_probs = log_probs

    best_paths = batch_first_probs.argmax(dim=-1).cpu().tolist()
    decoded_batch = []

    for path in best_paths:
        collapsed = []
        prev = None
        for tok in path:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        decoded_batch.append(collapsed)

    return decoded_batch


def calculate_wer(reference: list[str], hypothesis: list[str]) -> float:
    """
    Compute Word Error Rate (WER) using Levenshtein distance on token lists.
    WER = (S + D + I) / N
    """
    r_len = len(reference)
    h_len = len(hypothesis)

    if r_len == 0:
        return 0.0 if h_len == 0 else 1.0

    d = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        d[i][0] = i
    for j in range(h_len + 1):
        d[0][j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                insertion = d[i][j - 1] + 1
                deletion = d[i - 1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[r_len][h_len] / r_len


def calculate_batch_metrics(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """
    Compute aggregate evaluation metrics across a batch or dataset.
    """
    total_words = 0
    total_exact_match = 0
    wers = []

    for hyp, ref in zip(hypotheses, references, strict=False):
        ref_words = ref.strip().split()
        hyp_words = hyp.strip().split()

        total_words += len(ref_words)
        if hyp.strip() == ref.strip():
            total_exact_match += 1

        wers.append(calculate_wer(ref_words, hyp_words))

    avg_wer = float(sum(wers) / max(len(wers), 1))
    sentence_acc = float(total_exact_match / max(len(references), 1))

    return {
        "wer": avg_wer,
        "token_accuracy": max(0.0, 1.0 - avg_wer),
        "sentence_accuracy": sentence_acc,
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: Any,
    criterion: nn.CTCLoss,
    device: torch.device,
    vocabulary: SignVocabulary,
) -> dict[str, Any]:
    """
    Evaluate a sign language recognition model on a full dataloader partition.

    Args:
        model: Continuous sign language PyTorch model.
        dataloader: PyTorch DataLoader yielding sign batches.
        criterion: CTCLoss module.
        device: Evaluation torch.device (CUDA or CPU).
        vocabulary: SignVocabulary instance for CTC decoding.

    Returns:
        Dictionary of aggregate metrics and qualitative predictions.
    """
    model.eval()
    total_loss = 0.0
    valid_batches = 0

    all_hypotheses = []
    all_references = []

    blank_id = vocabulary.token2id[SignVocabulary.BLANK_TOKEN]

    for batch in dataloader:
        keypoints = batch["keypoints"].to(device)
        lengths = batch["lengths"].to(device)
        token_ids = batch["token_ids"].to(device)
        token_lengths = batch["token_lengths"].to(device)

        log_probs = model(keypoints, lengths)
        input_lengths = lengths.clamp(min=1)
        target_lengths = token_lengths.clamp(min=1)

        loss = criterion(log_probs, token_ids, input_lengths, target_lengths)
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
            valid_batches += 1

        decoded_ids = ctc_greedy_decode(log_probs, blank_id=blank_id)
        for i, pred_ids in enumerate(decoded_ids):
            pred_text = vocabulary.decode(pred_ids)
            ref_text = batch["texts"][i]
            all_hypotheses.append(pred_text)
            all_references.append(ref_text)

    metrics = calculate_batch_metrics(all_hypotheses, all_references)
    metrics["val_loss"] = total_loss / max(valid_batches, 1)
    metrics["sample_predictions"] = list(zip(all_references[:3], all_hypotheses[:3], strict=False))
    return metrics

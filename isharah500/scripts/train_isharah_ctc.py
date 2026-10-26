"""
SIGNOVA — Isharah-500 Baseline Training Pipeline (BiLSTM + CTC Loss).

This script provides an end-to-end training pipeline for Continuous Sign Language
Recognition on the Isharah-500 dataset:
- Loads train and dev splits with Signer-Independent (SI) or Unseen-Sentence (US) schemes.
- Handles missing hand detections (NaN interpolation).
- Filters out low-quality/degraded samples automatically (exclude_flagged=True).
- Trains a Bidirectional LSTM with Connectionist Temporal Classification (CTC) loss.
- Evaluates validation loss and greedy decoding character/token error rates.
- Saves model checkpoints and vocabulary mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "isharah500" / "src"))

from isharah500.src.isharah_dataset import (  # noqa: E402
    GlossVocabulary,
    IsharahDataset,
    collate_isharah_batch,
)


class SignLanguageBiLSTM(nn.Module):
    """Bidirectional LSTM encoder for continuous sign keypoint sequences."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Padded keypoint sequences of shape (Batch, Time, Input_Dim).
            lengths: Sequence lengths tensor of shape (Batch,).
        Returns:
            Log-probabilities of shape (Time, Batch, Num_Classes) for CTC loss.
        """
        projected = self.input_proj(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        logits = self.classifier(out)
        # CTC Loss expects (Time, Batch, Num_Classes) with log_softmax
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        return log_probs


def greedy_decode(log_probs: torch.Tensor, blank_id: int = 0) -> list[list[int]]:
    """
    Greedy argmax decoding collapsing consecutive repeats and blank tokens.
    Args:
        log_probs: Tensor of shape (Time, Batch, Num_Classes).
    Returns:
        List of decoded token id lists for each batch item.
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()  # (Batch, Time)
    batch_decoded = []
    for seq in preds:
        collapsed = []
        prev = None
        for tok in seq:
            if tok != prev and tok != blank_id:
                collapsed.append(tok)
            prev = tok
        batch_decoded.append(collapsed)
    return batch_decoded


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch in loader:
        keypoints = batch["keypoints"].to(device)
        lengths = batch["lengths"].to(device)
        token_ids = batch["token_ids"].to(device)
        token_lengths = batch["token_lengths"].to(device)

        optimizer.zero_grad()
        log_probs = model(keypoints, lengths)

        input_lengths = lengths.clamp(min=1)
        target_lengths = token_lengths.clamp(min=1)

        loss = criterion(log_probs, token_ids, input_lengths, target_lengths)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
    vocab: GlossVocabulary,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    valid_batches = 0
    correct_tokens = 0
    total_tokens = 0

    blank_id = vocab.token2id[GlossVocabulary.BLANK_TOKEN]

    for batch in loader:
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

        decoded = greedy_decode(log_probs, blank_id=blank_id)
        for i, pred_seq in enumerate(decoded):
            true_seq = token_ids[i][: token_lengths[i]].tolist()
            total_tokens += len(true_seq)
            # Count exact matches
            matches = sum(1 for p, t in zip(pred_seq, true_seq, strict=False) if p == t)
            correct_tokens += matches

    avg_loss = total_loss / max(valid_batches, 1)
    acc = correct_tokens / max(total_tokens, 1)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Model on Isharah-500")
    parser.add_argument(
        "--train-meta",
        type=str,
        default="isharah500/metadata/isharah500_si_train.csv",
        help="Path to training metadata CSV",
    )
    parser.add_argument(
        "--dev-meta",
        type=str,
        default="isharah500/metadata/isharah500_si_dev.csv",
        help="Path to dev metadata CSV",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=256, help="LSTM hidden state dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument(
        "--exclude-flagged",
        action="store_true",
        default=True,
        help="Filter out low-quality samples",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints/isharah_baseline",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build Vocabulary
    vocab = GlossVocabulary(add_ctc_blank=True)
    temp_ds = IsharahDataset(metadata_csv=ROOT_DIR / args.train_meta)
    all_glosses = [s["gloss"] for s in temp_ds.samples]
    vocab.build_vocab(all_glosses)
    print(f"Built vocabulary with {len(vocab)} unique tokens (including special tokens).")

    # Datasets & Loaders
    train_dataset = IsharahDataset(
        metadata_csv=ROOT_DIR / args.train_meta,
        exclude_flagged=args.exclude_flagged,
        nan_strategy="interpolate",
        flatten_hands=True,
        vocabulary=vocab,
    )
    dev_dataset = IsharahDataset(
        metadata_csv=ROOT_DIR / args.dev_meta,
        exclude_flagged=False,  # Keep dev evaluation uncompromised
        nan_strategy="interpolate",
        flatten_hands=True,
        vocabulary=vocab,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_isharah_batch,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_isharah_batch,
    )

    print(f"Training samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

    # Model & Criterion
    model = SignLanguageBiLSTM(
        input_dim=126,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(vocab),
    ).to(device)

    blank_id = vocab.token2id[GlossVocabulary.BLANK_TOKEN]
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    output_path = ROOT_DIR / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Save vocabulary
    with open(output_path / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {"token2id": vocab.token2id, "id2token": vocab.id2token},
            f,
            ensure_ascii=False,
            indent=2,
        )

    best_dev_loss = float("inf")

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device, vocab)

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Dev Loss: {dev_loss:.4f} "
            f"| Dev Token Acc: {dev_acc * 100:.2f}%"
        )

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), output_path / "best_model.pth")
            print(f"  --> Saved new best checkpoint to {output_path / 'best_model.pth'}")

    print(f"\nTraining Complete. Best Dev Loss: {best_dev_loss:.4f}")


if __name__ == "__main__":
    main()

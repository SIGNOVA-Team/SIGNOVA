"""
SIGNOVA — Unified Training Module for ArabSign and How2Sign Continuous Sign Models.

Provides:
- Model initialization, optimizer, scheduler, and loss function.
- Robust train and evaluation epoch loops.
- Checkpoint persistence and metric logging.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Disable cuDNN RNN kernel on Windows to prevent driver teardown bug on Blackwell architecture (RTX 50-series)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

from ml.sign_language.src.data_loader import create_dataloader  # noqa: E402
from ml.sign_language.src.dataset import SignLanguageDataset  # noqa: E402
from ml.sign_language.src.evaluate import evaluate_epoch  # noqa: E402
from ml.sign_language.src.model import build_model  # noqa: E402
from ml.sign_language.src.preprocessing import SignVocabulary  # noqa: E402


def train_one_epoch(
    model: nn.Module,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CTCLoss,
    device: torch.device,
    clip_norm: float = 5.0,
) -> float:
    model.train()
    total_loss = 0.0
    valid_batches = 0

    for batch in dataloader:
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
        optimizer.step()

        total_loss += loss.item()
        valid_batches += 1

    return total_loss / max(valid_batches, 1)


def run_training(
    dataset_name: str = "arabsign",
    config_path: str | Path | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Execute complete training workflow for ArabSign or How2Sign."""
    root_dir = Path(__file__).resolve().parents[3]
    if config_path is None:
        config_path = root_dir / "ml/sign_language/configs/config.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Device selection (CUDA / CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training starting on device: {device} (dataset: {dataset_name})")

    ds_cfg = config["data"][dataset_name]
    canonical_csv = root_dir / ds_cfg["canonical_metadata_path"]
    split_csv = root_dir / ds_cfg["split_manifest_path"]
    language = ds_cfg["language"]

    # 1. Build Vocabulary
    vocab = SignVocabulary(language=language, add_ctc_blank=True)
    temp_train_ds = SignLanguageDataset(
        language=language,
        canonical_csv=canonical_csv,
        split_manifest_csv=split_csv,
        split="train",
        max_samples=max_samples,
    )
    texts = [s.get("gloss_label") or s.get("original_label", "") for s in temp_train_ds.samples]
    vocab.build_from_texts(texts)
    print(f"Built vocabulary: {len(vocab)} unique tokens for {language}")

    # 2. Datasets & Loaders
    b_size = batch_size or config["training"]["batch_size"]
    train_dataset = SignLanguageDataset(
        language=language,
        canonical_csv=canonical_csv,
        split_manifest_csv=split_csv,
        split="train",
        vocabulary=vocab,
        max_samples=max_samples,
    )
    val_dataset = SignLanguageDataset(
        language=language,
        canonical_csv=canonical_csv,
        split_manifest_csv=split_csv,
        split="val",
        vocabulary=vocab,
        max_samples=max_samples // 4 if max_samples else None,
    )

    train_loader = create_dataloader(train_dataset, batch_size=b_size, shuffle=True)
    val_loader = create_dataloader(val_dataset, batch_size=b_size, shuffle=False)

    print(f"Loaded {len(train_dataset)} train samples and {len(val_dataset)} val samples.")

    # 3. Model, Criterion, Optimizer
    model = build_model(config, num_classes=len(vocab)).to(device)
    blank_id = vocab.token2id[SignVocabulary.BLANK_TOKEN]
    criterion = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    lr = float(config["training"]["learning_rate"])
    wd = float(config["training"]["weight_decay"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # 4. Artifact directory setup
    artifact_dir = root_dir / config["training"]["checkpoint_dir"] / dataset_name
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = artifact_dir / "vocab.json"
    vocab.save(vocab_path)

    # 5. Training loop
    num_epochs = epochs or config["training"]["max_epochs"]
    best_metric = float("inf")
    history = []

    for ep in range(1, num_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, vocab)

        val_loss = val_metrics["val_loss"]
        val_wer = val_metrics["wer"]
        val_acc = val_metrics["token_accuracy"]

        print(
            f"Epoch [{ep:02d}/{num_epochs:02d}] "
            f"| Train Loss: {tr_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Val WER: {val_wer:.4f} "
            f"| Token Acc: {val_acc * 100:.2f}%"
        )

        history.append(
            {
                "epoch": ep,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "wer": val_wer,
                "token_accuracy": val_acc,
            }
        )

        if val_loss < best_metric:
            best_metric = val_loss
            ckpt_path = artifact_dir / "best_model.pth"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "num_classes": len(vocab),
                    "language": language,
                    "epoch": ep,
                    "val_loss": val_loss,
                    "wer": val_wer,
                },
                ckpt_path,
            )
            print(f"  --> Saved new best checkpoint to {ckpt_path}")

    return {
        "dataset_name": dataset_name,
        "language": language,
        "epochs": num_epochs,
        "history": history,
        "best_val_loss": best_metric,
        "artifact_dir": str(artifact_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Train SIGNOVA Continuous Sign Model")
    parser.add_argument(
        "--dataset",
        choices=["arabsign", "how2sign"],
        default="arabsign",
        help="Dataset to train on",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Limit samples for pilot test"
    )
    args = parser.parse_args()

    run_training(
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

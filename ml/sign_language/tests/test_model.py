"""
Unit tests for model architecture and forward pass in SIGNOVA.
"""

import torch

from ml.sign_language.src.model import ContinuousSignBiLSTM, build_model


def test_model_forward():
    batch_size = 4
    seq_len = 64
    input_dim = 126
    vocab_size = 50

    model = ContinuousSignBiLSTM(
        input_dim=input_dim,
        hidden_dim=128,
        num_layers=2,
        num_classes=vocab_size,
        dropout=0.1,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    lengths = torch.tensor([64, 50, 40, 30], dtype=torch.long)

    # Forward pass returns log_probs of shape (Time, Batch, Num_Classes) for CTC
    log_probs = model(x, lengths)

    assert log_probs.shape == (seq_len, batch_size, vocab_size)
    # Log probabilities should be <= 0
    assert (log_probs <= 1e-4).all()


def test_build_model_from_config():
    cfg = {
        "model": {
            "architecture": "BiLSTM_CTC",
            "input_dim": 126,
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.0,
        }
    }
    model = build_model(cfg, num_classes=25)
    assert isinstance(model, ContinuousSignBiLSTM)
    assert model.num_classes == 25

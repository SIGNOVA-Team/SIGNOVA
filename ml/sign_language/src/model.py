"""
SIGNOVA — Deep Learning Architectures for Continuous Sign Language Recognition.

Includes:
1. ContinuousSignBiLSTM: Bidirectional LSTM with temporal linear projection and CTC head.
2. SignLanguageTransformerEncoder: Temporal Transformer encoder with sinusoidal positional encodings.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequence modeling."""

    def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (Batch, Time, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContinuousSignBiLSTM(nn.Module):
    """
    Standard Bidirectional LSTM encoder with Connectionist Temporal Classification (CTC) head.
    Robust, fast convergence, and ideal for continuous sign language gloss recognition.
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # Spatial Feature Extraction / Coordinate Projection
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Temporal Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Sequence Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Padded keypoint sequences (Batch, Time, Input_Dim)
            lengths: Sequence lengths (Batch,)
        Returns:
            Log-probabilities of shape (Time, Batch, Num_Classes) suitable for CTCLoss.
        """
        proj = self.feature_proj(x)

        # Pack padded sequence for efficient computation and correct masking
        packed = nn.utils.rnn.pack_padded_sequence(
            proj, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        logits = self.classifier(unpacked)  # (Batch, Time, Num_Classes)
        # CTC Loss requires (Time, Batch, Num_Classes)
        log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
        return log_probs


def build_model(
    config: dict[str, Any],
    num_classes: int,
) -> ContinuousSignBiLSTM:
    """Factory function to build sign language model from configuration dictionary."""
    m_cfg = config.get("model", {})
    arch = m_cfg.get("architecture", "BiLSTM_CTC")

    if arch == "BiLSTM_CTC":
        return ContinuousSignBiLSTM(
            input_dim=m_cfg.get("input_dim", 126),
            hidden_dim=m_cfg.get("hidden_dim", 256),
            num_layers=m_cfg.get("num_layers", 2),
            num_classes=num_classes,
            dropout=m_cfg.get("dropout", 0.2),
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

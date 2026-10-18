"""
SIGNOVA — Backend Inference & Prediction Interface.

Provides a clean, stable, decoupled API contract for the backend and frontend teams:
- load_model(checkpoint_path, vocab_path)
- predict(input_data)
- predict_batch(inputs)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ml.sign_language.src.evaluate import ctc_greedy_decode
from ml.sign_language.src.model import ContinuousSignBiLSTM
from ml.sign_language.src.preprocessing import (
    SignVocabulary,
    interpolate_temporal_keypoints,
    normalize_keypoint_coordinates,
)


class SignLanguagePredictor:
    """Production predictor wrapper for SIGNOVA sign language recognition models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        vocab_path: str | Path,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.vocab_path = Path(vocab_path)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Vocabulary
        self.vocabulary = SignVocabulary.load(self.vocab_path)
        self.language = self.vocabulary.language

        # 2. Load Checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        num_classes = checkpoint.get("num_classes", len(self.vocabulary))

        # 3. Instantiate Model
        self.model = ContinuousSignBiLSTM(
            input_dim=126,
            hidden_dim=256,
            num_layers=2,
            num_classes=num_classes,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.blank_id = self.vocabulary.token2id.get(SignVocabulary.BLANK_TOKEN, 0)

    def preprocess_input(self, input_features: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Validate, interpolate NaNs, normalize, and shape input into tensors.
        Accepts (T, 126) or (T, 2, 63).
        """
        arr = np.copy(input_features).astype(np.float32)
        if arr.ndim == 3 and arr.shape[1] == 2 and arr.shape[2] == 63:
            arr = arr.reshape(len(arr), 126)
        elif arr.ndim != 2 or arr.shape[1] != 126:
            raise ValueError(f"Expected input of shape (T, 126) or (T, 2, 63), got {arr.shape}")

        arr = interpolate_temporal_keypoints(arr)
        arr = normalize_keypoint_coordinates(arr)

        tensor_x = torch.from_numpy(arr).unsqueeze(0).to(self.device)  # (1, T, 126)
        length = torch.tensor([len(arr)], dtype=torch.long, device=self.device)
        return tensor_x, length

    @torch.no_grad()
    def predict(self, input_data: np.ndarray | str | Path) -> dict[str, Any]:
        """
        Predict recognized sign glosses / text for a single input sequence.

        Args:
            input_data: Either a NumPy array (T, 126) or path to a .npy keypoints file.

        Returns:
            Dictionary matching the backend API contract:
            {
                "success": bool,
                "language": str,
                "text": str,
                "tokens": list[str],
                "token_ids": list[int],
                "confidence": float,
                "latency_ms": float,
                "sequence_length": int
            }
        """
        start_time = time.perf_counter()

        # Handle file path input
        if isinstance(input_data, (str, Path)):
            p = Path(input_data)
            if not p.exists():
                return {
                    "success": False,
                    "error": f"Input file not found: {p}",
                    "language": self.language,
                }
            if p.suffix.lower() == ".npy":
                raw_features = np.load(p)
            elif p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                from ml.sign_language.src.video_extractor import extract_keypoints_from_video

                raw_features = extract_keypoints_from_video(p)
                if len(raw_features) == 0:
                    return {
                        "success": False,
                        "error": "No hand gestures or keypoints could be extracted from video.",
                        "language": self.language,
                    }
            else:
                return {
                    "success": False,
                    "error": f"Unsupported input format: {p.suffix}. Supported formats: .mp4, .avi, .mov, .npy.",
                    "language": self.language,
                }
        else:
            raw_features = input_data

        try:
            x, length = self.preprocess_input(raw_features)
            log_probs = self.model(x, length)  # (T, 1, Num_Classes)

            decoded_ids = ctc_greedy_decode(log_probs, blank_id=self.blank_id)[0]
            text = self.vocabulary.decode(decoded_ids)
            tokens = text.split()

            # Compute average probability of the predicted path
            probs = torch.exp(log_probs).squeeze(1)  # (T, C)
            max_probs, _ = probs.max(dim=-1)
            mean_conf = float(max_probs.mean().item())

            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return {
                "success": True,
                "language": self.language,
                "text": text,
                "tokens": tokens,
                "token_ids": decoded_ids,
                "confidence": round(mean_conf, 4),
                "latency_ms": round(latency_ms, 2),
                "sequence_length": int(length.item()),
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": self.language,
            }


# Top-level functional interface for backend developers
_DEFAULT_PREDICTOR: SignLanguagePredictor | None = None


def load_model(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    device: str | None = None,
) -> SignLanguagePredictor:
    """Initialize and return a persistent predictor instance."""
    global _DEFAULT_PREDICTOR
    _DEFAULT_PREDICTOR = SignLanguagePredictor(checkpoint_path, vocab_path, device=device)
    return _DEFAULT_PREDICTOR


def predict(
    input_data: np.ndarray | str | Path,
    predictor: SignLanguagePredictor | None = None,
) -> dict[str, Any]:
    """Execute prediction using the specified or globally loaded model."""
    target_predictor = predictor or _DEFAULT_PREDICTOR
    if target_predictor is None:
        raise RuntimeError("Model has not been initialized. Call load_model() first.")
    return target_predictor.predict(input_data)


def predict_video(
    video_path: str | Path,
    predictor: SignLanguagePredictor | None = None,
) -> dict[str, Any]:
    """Direct end-to-end video prediction (Video -> MediaPipe -> Continuous Sign Recognition)."""
    return predict(video_path, predictor=predictor)

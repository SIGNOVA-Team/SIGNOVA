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

            # Direct NLP Bridge Integration
            response: dict[str, Any] = {
                "success": True,
                "language": self.language,
                "text": text,
                "tokens": tokens,
                "token_ids": decoded_ids,
                "confidence": round(mean_conf, 4),
                "latency_ms": round(latency_ms, 2),
                "sequence_length": int(length.item()),
            }

            if self.language in {"ArSL", "ar", "arabic"}:
                from ml.sign_language.src.nlp_interface import get_translator

                translation_info = get_translator().translate_arabic_glosses(tokens)
                response["nlp_translation"] = translation_info
                response["spoken_arabic"] = translation_info.get("spoken_arabic", text)
                response["english_translation"] = translation_info.get("english_translation", "")
            else:
                response["nlp_translation"] = {
                    "spoken_english": text,
                    "match_type": "transcription",
                }
                response["spoken_english"] = text

            return response
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "language": self.language,
            }


# Top-level functional registry for backend developers
_PREDICTORS: dict[str, SignLanguagePredictor] = {}


def load_model(
    checkpoint_path: str | Path | None = None,
    vocab_path: str | Path | None = None,
    language: str = "ar",
    device: str | None = None,
) -> SignLanguagePredictor:
    """
    Initialize and return a persistent predictor instance for Arabic or English Sign Language.

    Args:
        checkpoint_path: Path to .pth checkpoint. Defaults to artifacts/{dataset}/best_model.pth.
        vocab_path: Path to vocab.json. Defaults to artifacts/{dataset}/vocab.json.
        language: 'ar' (default, ArabSign ArSL) or 'en' (How2Sign ASL).
        device: 'cuda', 'cpu', or None (auto-detect).
    """
    global _PREDICTORS

    lang_key = language.lower()
    dataset_name = "arabsign" if lang_key in {"ar", "arsl", "arabic"} else "how2sign"
    root_dir = Path(__file__).resolve().parents[3]

    if checkpoint_path is None:
        checkpoint_path = root_dir / f"ml/sign_language/artifacts/{dataset_name}/best_model.pth"
    else:
        checkpoint_path = Path(checkpoint_path)

    if vocab_path is None:
        vocab_path = checkpoint_path.parent / "vocab.json"
        if not vocab_path.exists():
            vocab_path = root_dir / f"ml/sign_language/artifacts/{dataset_name}/vocab.json"
    else:
        vocab_path = Path(vocab_path)

    predictor = SignLanguagePredictor(checkpoint_path, vocab_path, device=device)
    _PREDICTORS[lang_key] = predictor
    _PREDICTORS[dataset_name] = predictor
    return predictor


def predict(
    input_data: np.ndarray | str | Path,
    language: str = "ar",
    predictor: SignLanguagePredictor | None = None,
) -> dict[str, Any]:
    """
    Execute prediction on a video path (.mp4, .avi) or keypoint array.

    Args:
        input_data: Video file path (.mp4, .avi, .mov), .npy file path, or NumPy keypoint array.
        language: 'ar' for Arabic Sign Language or 'en' for American Sign Language.
        predictor: Optional pre-loaded predictor instance.
    """
    lang_key = language.lower()
    target_predictor = predictor or _PREDICTORS.get(lang_key)

    if target_predictor is None:
        target_predictor = load_model(language=language)

    return target_predictor.predict(input_data)


def predict_video(
    video_path: str | Path,
    language: str = "ar",
    predictor: SignLanguagePredictor | None = None,
) -> dict[str, Any]:
    """Direct end-to-end video prediction (Video -> MediaPipe -> Continuous Sign Recognition)."""
    return predict(video_path, language=language, predictor=predictor)

"""Reusable checkpoint-backed inference for Isharah-500 keypoint sequences."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

SOURCE_DIRECTORY = Path(__file__).resolve().parent
if str(SOURCE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIRECTORY))

from isharah_ctc import BiGRUCTC, greedy_ctc_decode  # noqa: E402
from isharah_dataset import (  # noqa: E402
    GlossVocabulary,
    PreprocessingConfig,
    interpolate_keypoints,
    normalize_hand_centric,
    validate_keypoints,
)
from isharah_features import construct_feature_representation  # noqa: E402
from isharah_training import load_checkpoint  # noqa: E402


@dataclass(frozen=True)
class Prediction:
    """Stable result returned by the backend prediction contract.

    ``gloss_tokens`` is a tuple for immutable Python use. Call ``to_dict``
    when crossing a JSON or backend serialization boundary; it converts the
    tuple to a list and preserves the documented field names.
    """

    gloss_tokens: tuple[str, ...]
    gloss_text: str
    input_length: int
    valid_hand_coverage: float
    device: str
    preprocessing_config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "gloss_tokens": list(self.gloss_tokens),
            "gloss_text": self.gloss_text,
            "input_length": self.input_length,
            "valid_hand_coverage": self.valid_hand_coverage,
            "device": self.device,
            "preprocessing_config": self.preprocessing_config,
        }


def _select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def preprocess_keypoints(
    keypoints: np.ndarray,
    preprocessing_config: PreprocessingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the saved dataset preprocessing contract to one raw sequence.

    NaNs represent missing detections and are accepted only when the configured
    preprocessing can resolve them. Infinities and nonfinite output are always
    rejected. The returned mask is the original hand-detection mask.
    """
    if not isinstance(keypoints, np.ndarray):
        raise TypeError("keypoints must be a NumPy array")
    validate_keypoints(keypoints, config=preprocessing_config)
    if np.isinf(keypoints).any():
        raise ValueError("keypoints contain infinite values")
    if preprocessing_config.flatten_hands:
        raise ValueError("flatten_hands=True is incompatible with the CTC model input contract")
    if preprocessing_config.nan_strategy == "none" and np.isnan(keypoints).any():
        raise ValueError("keypoints contain NaNs but nan_strategy='none'")

    valid_mask = ~np.all(np.isnan(keypoints), axis=-1)
    processed = keypoints.astype(np.float32, copy=True)
    if preprocessing_config.nan_strategy == "interpolate":
        processed = interpolate_keypoints(processed)
    elif preprocessing_config.nan_strategy == "zeros":
        processed = np.nan_to_num(processed, nan=0.0)
    global_keypoints = processed.copy()
    if preprocessing_config.coordinate_normalization == "hand_centric":
        processed = normalize_hand_centric(
            processed,
            valid_mask,
            epsilon=preprocessing_config.normalization_epsilon,
        )
    processed = construct_feature_representation(
        processed,
        valid_mask,
        feature_representation=preprocessing_config.feature_representation,
        global_coordinate_scale=preprocessing_config.global_coordinate_scale,
        global_keypoints=global_keypoints,
    )
    if not np.isfinite(processed).all():
        raise ValueError("preprocessed keypoints contain NaN or infinite values")
    return processed, valid_mask


class IsharahPredictor:
    """Application-scoped predictor serving repeated predictions.

    Construct this object once during application initialization with
    :func:`load_predictor`. Each call to :meth:`predict` reuses the loaded
    model, vocabulary, and preprocessing contract.
    """

    def __init__(
        self,
        model: BiGRUCTC,
        vocabulary: GlossVocabulary,
        preprocessing_config: PreprocessingConfig,
        device: torch.device,
    ) -> None:
        if preprocessing_config.expected_num_hands != model.config.num_hands:
            raise ValueError("Checkpoint and preprocessing hand dimensions differ")
        if preprocessing_config.output_features_per_hand != model.config.features_per_hand:
            raise ValueError("Checkpoint and preprocessing feature dimensions differ")
        self.model = model.eval()
        self.vocabulary = vocabulary
        self.preprocessing_config = preprocessing_config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device_name: str = "auto",
    ) -> IsharahPredictor:
        device = _select_device(device_name)
        model, vocabulary, preprocessing_config, _ = load_checkpoint(
            checkpoint_path,
            device=device,
        )
        if preprocessing_config.missing_file_policy != "error":
            raise ValueError("Inference requires missing_file_policy='error'")
        return cls(model, vocabulary, preprocessing_config, device)

    def predict(self, keypoints: np.ndarray) -> Prediction:
        """Predict glosses for one raw ``(T, 2, 63)`` sequence.

        The input must be a non-empty ``numpy.float32`` array. NaNs are
        treated as missing detections when the checkpoint's saved
        preprocessing strategy supports them; infinities are rejected.
        """
        processed, valid_mask = preprocess_keypoints(
            keypoints,
            self.preprocessing_config,
        )
        input_length = processed.shape[0]
        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.device)
        lengths = torch.tensor([input_length], dtype=torch.long, device=self.device)
        masks = torch.from_numpy(valid_mask).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            decoded = greedy_ctc_decode(
                self.model(tensor, lengths, masks),
                lengths,
                self.vocabulary,
            )[0]
        tokens = tuple(decoded.split()) if decoded else ()
        return Prediction(
            gloss_tokens=tokens,
            gloss_text=decoded,
            input_length=input_length,
            valid_hand_coverage=float(valid_mask.mean()),
            device=str(self.device),
            preprocessing_config=self.preprocessing_config.to_dict(),
        )


def load_predictor(
    checkpoint_path: str | Path,
    *,
    device_name: str = "auto",
) -> IsharahPredictor:
    """Load one reusable predictor instance from a checkpoint."""
    return IsharahPredictor.from_checkpoint(
        checkpoint_path,
        device_name=device_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--keypoints", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    keypoints = np.load(arguments.keypoints, allow_pickle=False)
    predictor = load_predictor(arguments.checkpoint, device_name=arguments.device)
    print(json.dumps(predictor.predict(keypoints).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

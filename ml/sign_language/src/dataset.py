"""
SIGNOVA — Dataset Adapters for ArabSign (ArSL) and How2Sign (ASL).

Provides unified sample iteration, split separation, modality adapters,
and robust handling of missing files with synthetic feature simulation for offline development.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ml.sign_language.src.preprocessing import (
    SignVocabulary,
    interpolate_temporal_keypoints,
    normalize_keypoint_coordinates,
)


class SignLanguageDataset:
    """
    Modular Sign Language Dataset supporting ArabSign and How2Sign.
    Ensures strict separation between ASL and ArSL vocabularies and evaluation protocols.
    """

    def __init__(
        self,
        language: str,  # 'ArSL' or 'ASL'
        canonical_csv: str | Path,
        split_manifest_csv: str | Path,
        split: str = "train",  # 'train', 'val', 'test'
        keypoints_dir: str | Path | None = None,
        vocabulary: SignVocabulary | None = None,
        nan_strategy: str = "interpolate",
        normalize_coords: bool = True,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
        max_samples: int | None = None,
    ):
        self.language = language
        self.canonical_csv = Path(canonical_csv)
        self.split_manifest_csv = Path(split_manifest_csv)
        self.split = split
        self.keypoints_dir = Path(keypoints_dir) if keypoints_dir else None
        self.vocabulary = vocabulary
        self.nan_strategy = nan_strategy
        self.normalize_coords = normalize_coords
        self.transform = transform

        # 1. Load split IDs from manifest to guarantee 0% data leakage
        valid_sample_ids = set()
        with open(self.split_manifest_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split") == self.split:
                    valid_sample_ids.add(row["sample_id"])

        # 2. Load canonical metadata matching the split
        self.samples: list[dict[str, Any]] = []
        with open(self.canonical_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id")
                if sid in valid_sample_ids:
                    self.samples.append(row)
                    if max_samples and len(self.samples) >= max_samples:
                        break

    def __len__(self) -> int:
        return len(self.samples)

    def _generate_deterministic_mock_keypoints(
        self, sample_id: str, duration_sec: float
    ) -> np.ndarray:
        """
        Generate reproducible, physically plausible continuous hand keypoints (T, 126)
        when raw binary files are not locally present.
        Uses sample_id hash as seed to guarantee reproducibility across workers.
        """
        seed = int(hashlib.md5(sample_id.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)

        fps = 30.0 if self.language == "ArSL" else 25.0
        num_frames = max(int(duration_sec * fps), 15)

        # Baseline coordinates with smooth sinusoidal movement + noise
        t = np.linspace(0, 4 * np.pi, num_frames)[:, None]
        base_motion = np.sin(t) * 0.15 + 0.5
        features = np.tile(base_motion, (1, 126)) + rng.normal(0, 0.02, (num_frames, 126))

        # Inject realistic occasional NaN hand loss (e.g. 8% of frames)
        missing_mask = rng.rand(num_frames) < 0.08
        features[missing_mask] = np.nan
        return features.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        sample_id = row["sample_id"]
        gloss_or_text = row.get("gloss_label") or row.get("original_label", "")
        signer_id = row.get("signer_id", "unknown")
        duration = float(row.get("duration_sec", 3.0))

        # Load real keypoints if present, else fallback deterministically
        loaded_real = False
        if self.keypoints_dir:
            kp_path = self.keypoints_dir / f"{sample_id}.npy"
            if kp_path.exists():
                keypoints = np.load(kp_path).astype(np.float32)
                loaded_real = True

        if not loaded_real:
            keypoints = self._generate_deterministic_mock_keypoints(sample_id, duration)

        # Preprocessing: Handle missing values (NaNs)
        if self.nan_strategy == "interpolate":
            keypoints = interpolate_temporal_keypoints(keypoints)
        elif self.nan_strategy == "zeros":
            keypoints = np.nan_to_num(keypoints, nan=0.0)

        # Preprocessing: Spatial Normalization
        if self.normalize_coords:
            keypoints = normalize_keypoint_coordinates(keypoints)

        # Preprocessing: Transform
        if self.transform is not None:
            keypoints = self.transform(keypoints)

        item: dict[str, Any] = {
            "sample_id": sample_id,
            "language": self.language,
            "keypoints": keypoints,
            "length": len(keypoints),
            "text": gloss_or_text,
            "signer_id": signer_id,
        }

        # Tokenization
        if self.vocabulary is not None:
            token_ids = self.vocabulary.encode(gloss_or_text)
            item["token_ids"] = token_ids
            item["token_length"] = len(token_ids)

        return item

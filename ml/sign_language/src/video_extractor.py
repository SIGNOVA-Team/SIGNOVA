"""
SIGNOVA — Video Feature Extractor using Google MediaPipe HandLandmarker.

Transforms raw video files (.mp4, .avi, .mov, etc.) or OpenCV frame generators
into continuous (T, 126) hand keypoint coordinate sequences for ML inference.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class VideoKeypointExtractor:
    """Extracts continuous dual-hand landmarks (21 joints * 3 coords * 2 hands = 126 features) from video."""

    def __init__(self, model_asset_path: str | Path | None = None):
        root_dir = Path(__file__).resolve().parents[3]
        if model_asset_path is None:
            model_asset_path = root_dir / "ml/sign_language/artifacts/hand_landmarker.task"

        self.model_asset_path = Path(model_asset_path)
        if not self.model_asset_path.exists():
            self._download_model_asset()

        base_options = python.BaseOptions(model_asset_path=str(self.model_asset_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _download_model_asset(self) -> None:
        """Download official MediaPipe HandLandmarker task model asset."""
        import urllib.request

        self.model_asset_path.parent.mkdir(parents=True, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            urllib.request.urlretrieve(url, self.model_asset_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to auto-download MediaPipe hand landmarker model: {e}"
            ) from e

    def extract_from_video(
        self,
        video_path: str | Path,
        max_frames: int | None = None,
        target_fps: float | None = None,
    ) -> np.ndarray:
        """
        Extract continuous hand keypoints from a video file.

        Args:
            video_path: Path to video file.
            max_frames: Optional upper limit on processed frames.
            target_fps: Optional sampling rate.

        Returns:
            np.ndarray of shape (T, 126) with (x, y, z) coordinates for Left & Right hands.
        """
        v_path = Path(video_path)
        if not v_path.exists():
            raise FileNotFoundError(f"Video file not found: {v_path}")

        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {v_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = (
            int(round(orig_fps / target_fps)) if target_fps and target_fps < orig_fps else 1
        )

        all_keypoints: list[np.ndarray] = []
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval != 0:
                    frame_count += 1
                    continue

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                detection_result = self.detector.detect(mp_image)
                frame_kp = self._process_detection(detection_result)
                all_keypoints.append(frame_kp)

                frame_count += 1
                if max_frames and len(all_keypoints) >= max_frames:
                    break
        finally:
            cap.release()

        if not all_keypoints:
            return np.empty((0, 126), dtype=np.float32)

        return np.vstack(all_keypoints).astype(np.float32)

    def _process_detection(self, detection_result: vision.HandLandmarkerResult) -> np.ndarray:
        """
        Convert detection result into a single 126-D array:
        [Hand 0 (63 coords), Hand 1 (63 coords)]
        Missing hands are populated with NaNs for downstream temporal interpolation.
        """
        hand_0 = np.full((21, 3), np.nan, dtype=np.float32)
        hand_1 = np.full((21, 3), np.nan, dtype=np.float32)

        if detection_result.hand_landmarks:
            num_detected = len(detection_result.hand_landmarks)
            if num_detected >= 1:
                landmarks = detection_result.hand_landmarks[0]
                hand_0 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
            if num_detected >= 2:
                landmarks = detection_result.hand_landmarks[1]
                hand_1 = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        combined = np.concatenate([hand_0.flatten(), hand_1.flatten()])
        return combined.reshape(1, 126)


_DEFAULT_EXTRACTOR: VideoKeypointExtractor | None = None


def extract_keypoints_from_video(video_path: str | Path) -> np.ndarray:
    """Convenience module function for direct video-to-keypoints extraction."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = VideoKeypointExtractor()
    return _DEFAULT_EXTRACTOR.extract_from_video(video_path)

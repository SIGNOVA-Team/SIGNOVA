"""
SIGNOVA — Computer Vision (CV) Streaming & Window Buffer Interface.

Enables real-time, online continuous sign language recognition on live camera feeds:
- Sliding temporal keypoint windowing (e.g. 60 frames, stride 15 frames)
- Live motion/activity detection to skip idle/resting hands
- Decoupled frame-by-frame ingestion from OpenCV camera streams
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class StreamingKeypointBuffer:
    """
    Sliding window temporal buffer for real-time video stream ingestion.
    Buffers incoming (126,) keypoint vectors and produces (window_size, 126) sequence tensors.
    """

    def __init__(
        self,
        window_size: int = 60,
        stride: int = 15,
        activity_threshold: float = 0.005,
    ):
        """
        Args:
            window_size: Number of frames per sequence (e.g. 60 frames ~ 2.0s at 30fps).
            stride: Number of new frames needed to trigger a new prediction window.
            activity_threshold: Minimum keypoint coordinate variance/movement to trigger prediction.
        """
        self.window_size = window_size
        self.stride = stride
        self.activity_threshold = activity_threshold

        self.buffer: deque[np.ndarray] = deque(maxlen=window_size)
        self.frames_since_last_prediction = 0

    def add_keypoint_frame(self, frame_keypoints: np.ndarray) -> np.ndarray | None:
        """
        Add a single frame's 126-D keypoint array to the buffer.

        Args:
            frame_keypoints: Array of shape (126,) or (1, 126).

        Returns:
            np.ndarray of shape (window_size, 126) if a full stride is ready and active;
            otherwise None.
        """
        arr = np.asarray(frame_keypoints, dtype=np.float32).flatten()
        if len(arr) != 126:
            raise ValueError(f"Expected frame keypoint array of size 126, got {len(arr)}")

        self.buffer.append(arr)
        self.frames_since_last_prediction += 1

        if (
            len(self.buffer) == self.window_size
            and self.frames_since_last_prediction >= self.stride
        ):
            window_arr = np.array(self.buffer, dtype=np.float32)  # (window_size, 126)
            self.frames_since_last_prediction = 0

            # Check for active movement (skip stationary / resting hands)
            if self.is_active_gesture(window_arr):
                return window_arr

        return None

    def is_active_gesture(self, sequence: np.ndarray) -> bool:
        """Determine whether sequence contains significant hand movement."""
        valid_vals = sequence[~np.isnan(sequence)]
        if len(valid_vals) < 20:
            return False
        # Calculate temporal variance
        variance = float(np.nanvar(sequence))
        return variance >= self.activity_threshold

    def clear(self) -> None:
        """Reset the buffer."""
        self.buffer.clear()
        self.frames_since_last_prediction = 0


class RealtimeCVStreamProcessor:
    """
    End-to-end live camera processor:
    Frame (BGR) -> MediaPipe Landmark Extractor -> Streaming Buffer -> ML Predictor -> NLP Translation.
    """

    def __init__(
        self,
        language: str = "ar",
        window_size: int = 60,
        stride: int = 15,
    ):
        from ml.sign_language.src.inference import load_model
        from ml.sign_language.src.video_extractor import VideoKeypointExtractor

        self.language = language
        self.extractor = VideoKeypointExtractor()
        self.buffer = StreamingKeypointBuffer(window_size=window_size, stride=stride)
        self.predictor = load_model(language=language)

    def process_frame(self, bgr_frame: np.ndarray) -> dict[str, Any] | None:
        """
        Process a single OpenCV BGR camera frame.

        Args:
            bgr_frame: Image from cv2.VideoCapture (H, W, 3).

        Returns:
            Prediction response dictionary if a prediction window fired, else None.
        """
        import cv2
        import mediapipe as mp

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = self.extractor.detector.detect(mp_image)
        frame_kp = self.extractor._process_detection(detection_result).flatten()

        window = self.buffer.add_keypoint_frame(frame_kp)
        if window is not None:
            return self.predictor.predict(window)

        return None

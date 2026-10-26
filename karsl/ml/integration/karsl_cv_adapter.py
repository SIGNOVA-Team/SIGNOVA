from pathlib import Path
from types import SimpleNamespace

import numpy as np

from karsl.ml.src.preprocessing import PREPROCESSING_VERSION, preprocess_pose


def extract_karsl_landmarks(rgb_frames):
    frames = list(rgb_frames)
    if not frames or len(frames) > 300:
        raise ValueError("Expected 1 to 300 RGB frames from one isolated sign")
    if any(
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        for rgb in frames
    ):
        raise ValueError("Each frame must be an RGB uint8 array of shape (height, width, 3)")
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise ImportError("Install the separate CV environment with mediapipe==0.10.21.") from exc
    if mp.__version__ != "0.10.21":
        raise ValueError("This research adapter was checked only with mediapipe==0.10.21.")
    data = np.full((len(frames), 1, 543, 3), np.nan, dtype=np.float32)
    confidence = np.zeros((len(frames), 1, 543), dtype=np.float32)
    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as detector:
        for frame_index, rgb in enumerate(frames):
            result = detector.process(rgb)
            for landmarks, offset, count in (
                (result.pose_landmarks, 0, 33),
                (result.left_hand_landmarks, 501, 21),
                (result.right_hand_landmarks, 522, 21),
            ):
                if landmarks is None:
                    continue
                for index, point in enumerate(landmarks.landmark[:count]):
                    data[frame_index, 0, offset + index] = [
                        point.x * 1000,
                        point.y * 1000,
                        point.z * 1000,
                    ]
                    confidence[frame_index, 0, offset + index] = (
                        float(point.visibility) if offset == 0 else 1.0
                    )
    return SimpleNamespace(body=SimpleNamespace(data=data, confidence=confidence))


def features_from_rgb_frames(rgb_frames):
    return preprocess_pose(extract_karsl_landmarks(rgb_frames))


def predict_rgb_frames(predictor, rgb_frames, top_k=5):
    features = features_from_rgb_frames(rgb_frames)
    return predictor.predict_features(
        features, preprocessing_version=PREPROCESSING_VERSION, top_k=top_k
    )


def read_rgb_video(path, max_frames=300):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("Install OpenCV in the separate CV environment.") from exc
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        raise ValueError("Cannot open the supplied video")
    frames = []
    try:
        while len(frames) <= max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames or len(frames) > max_frames:
        raise ValueError("Expected 1 to 300 decodable frames from one isolated sign")
    return frames

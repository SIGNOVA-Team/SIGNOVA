"""Extract hand landmarks from ordered video frames."""

from pathlib import Path

import mediapipe as mp
import numpy as np

HAND_COUNT = 2
LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_FEATURE_COUNT = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
HAND_LABEL_TO_INDEX = {
    "Left": 0,
    "Right": 1,
}


def extract_sample_hand_keypoints(sample_directory, model_path, fps=30):
    """Extract ordered hand landmarks and per-frame detection counts."""
    sample_directory = Path(sample_directory)
    model_path = Path(model_path)
    if fps <= 0:
        raise ValueError("fps must be greater than zero.")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    frame_paths = sorted(sample_directory.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No JPG frames found in {sample_directory}")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=HAND_COUNT,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    keypoint_sequences = []
    detection_counts = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        for frame_index, frame_path in enumerate(frame_paths):
            image = mp.Image.create_from_file(str(frame_path))
            timestamp_ms = round(frame_index * 1000 / fps)
            result = landmarker.detect_for_video(
                image,
                timestamp_ms,
            )
            frame_keypoints = np.full(
                (HAND_COUNT, HAND_FEATURE_COUNT),
                np.nan,
                dtype=np.float32,
            )
            for landmarks, handedness in zip(
                result.hand_landmarks,
                result.handedness,
                strict=False,
            ):
                hand_label = handedness[0].category_name
                hand_index = HAND_LABEL_TO_INDEX.get(hand_label)
                if hand_index is None:
                    continue
                coordinates = [
                    coordinate
                    for landmark in landmarks
                    for coordinate in (
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    )
                ]
                frame_keypoints[hand_index] = coordinates
            keypoint_sequences.append(frame_keypoints)
            detection_counts.append(len(result.hand_landmarks))
    return (
        np.stack(keypoint_sequences),
        np.asarray(detection_counts, dtype=np.int32),
    )

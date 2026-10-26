import numpy as np
PREPROCESSING_VERSION = 'pose_v1_32x50_xy_visibility_shoulder_conf05'
BODY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24]
HAND_INDICES = list(range(501, 543))
LANDMARK_INDICES = BODY_INDICES + HAND_INDICES
NUM_FRAMES = 32
INPUT_DIM = 150

def preprocess_pose(pose_object):
    data = np.asarray(pose_object.body.data, dtype=np.float32)
    confidence = np.asarray(pose_object.body.confidence, dtype=np.float32)
    if data.ndim != 4 or data.shape[1:] != (1, 543, 3):
        raise ValueError(f'Unexpected pose shape: {data.shape}')
    if confidence.shape != data.shape[:3]:
        raise ValueError(f'Unexpected confidence shape: {confidence.shape}')
    if not len(data):
        raise ValueError('Empty sequence')
    coordinates = data[:, 0, :, :2]
    scores = confidence[:, 0, :]
    valid_sh = (scores[:, 11] > 0.5) & (scores[:, 12] > 0.5) & np.isfinite(coordinates[:, 11]).all(axis=1) & np.isfinite(coordinates[:, 12]).all(axis=1)
    if not valid_sh.any():
        raise ValueError('No valid shoulder reference found')
    left = coordinates[valid_sh, 11]
    right = coordinates[valid_sh, 12]
    center = np.median((left + right) / 2, axis=0)
    widths = np.linalg.norm(left - right, axis=1)
    good_widths = widths[widths > 1e-06]
    if not len(good_widths):
        raise ValueError('Invalid shoulder width')
    scale = float(np.median(good_widths))
    selected = coordinates[:, LANDMARK_INDICES, :]
    selected_scores = scores[:, LANDMARK_INDICES]
    valid = (selected_scores > 0.5) & np.isfinite(selected).all(axis=2)
    normalized = np.clip((selected - center) / scale, -5.0, 5.0)
    normalized = np.where(valid[..., None], normalized, 0.0)
    features = np.concatenate([normalized, valid[..., None].astype(np.float32)], axis=2)
    indices = np.rint(np.linspace(0, len(features) - 1, NUM_FRAMES)).astype(int)
    return features[indices].reshape(NUM_FRAMES, INPUT_DIM).astype(np.float32)

def load_pose_file(path):
    try:
        from pose_format import Pose
    except ImportError as exc:
        raise ImportError('Install pose-format to read .pose files') from exc
    from pathlib import Path
    with Path(path).open('rb') as stream:
        return Pose.read(stream.read())

def validate_features(features, batch=False):
    a = np.asarray(features, dtype=np.float32)
    required = (None, NUM_FRAMES, INPUT_DIM) if batch else (NUM_FRAMES, INPUT_DIM)
    if batch:
        if a.ndim != 3 or a.shape[1:] != required[1:] or len(a) == 0:
            raise ValueError(f'Expected batch (N,32,150), got {a.shape}')
    elif a.shape != required:
        raise ValueError(f'Expected (32,150), got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError('Input contains NaN or infinity')
    return a

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MODEL_SHA256 = 'f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254'
CALIBRATION_SHA256 = '448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4'
LABELS_SHA256 = '3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a'
PREPROCESSING_SHA256 = '4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c'
MODEL_MODULE_SHA256 = 'f6e0f60a817de984fc356b7bbd7edcaff72ffbc63e046fe09175564cc6633b5e'
FINGERPRINT = '5d0d43950dec9c69bef2766e48550463a027b49a44748631b976f211c77e7c4a'
PREPROCESSING_VERSION = 'pose_v1_32x50_xy_visibility_shoulder_conf05'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_python(path, module_name, expected_sha256):
    require(sha256(path) == expected_sha256, f'Changed Python dependency: {path}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KArSL502Predictor:
    def __init__(self, base=None, repo_root=None, device='cpu'):
        checkout = Path(__file__).resolve().parents[4]
        self.repo_root = Path(repo_root) if repo_root is not None else checkout
        self.base = Path(base) if base is not None else self.repo_root / 'karsl/ml/ml_expansion'
        self.device = torch.device(device)
        if self.device.type == 'cuda':
            require(torch.cuda.is_available(), 'CUDA is not available.')
        scripts = self.base / 'scripts'
        preprocess_path = self.repo_root / 'karsl/ml/src/preprocessing.py'
        release = self.repo_root / 'karsl/ml/artifacts'
        model_path = release / 'cnn/best_cnn_502_pass_v1.pt'
        calibration_path = release / 'cnn/calibration_config.json'
        labels_path = release / 'labels_mapping.json'
        if not model_path.is_file() and not calibration_path.is_file() and not labels_path.is_file():
            model_path = self.base / 'experiments/cnn_502_pass_v1/best_cnn_502_pass_v1.pt'
            calibration_path = self.base / 'experiments/cnn_502_pass_v1/calibration_v1/calibration_config.json'
            labels_path = self.base / 'source_metadata/labels_mapping.json'
        require(sha256(model_path) == MODEL_SHA256, 'Selected CNN checkpoint changed.')
        require(sha256(calibration_path) == CALIBRATION_SHA256, 'Frozen calibration config changed.')
        require(sha256(labels_path) == LABELS_SHA256, 'Source labels changed.')
        self.preprocessing = load_python(preprocess_path, 'signova_frozen_preprocessing', PREPROCESSING_SHA256)
        common = load_python(scripts / 'signova_compare_cnn_502.py', 'signova_frozen_model', MODEL_MODULE_SHA256)
        require(self.preprocessing.PREPROCESSING_VERSION == PREPROCESSING_VERSION,
                'Preprocessing implementation version mismatch.')
        config = json.loads(calibration_path.read_text(encoding='utf-8'))
        require(config['model_name'] == 'SignCNN502' and config['model_version'] == 'karsl_cnn_502_pass_v1'
                and config['calibration_version'] == 'cnn502_validation_temperature_v1'
                and config['checkpoint_sha256'] == MODEL_SHA256
                and config['preprocessing_version'] == PREPROCESSING_VERSION
                and config['source_fingerprint'] == FINGERPRINT
                and config['class_ids'] == list(range(1, 503)), 'Calibration configuration mismatch.')
        require(config['acceptance_enabled'] is False and config['threshold'] is None and
                config['candidate_threshold'] is None and not config['oov_detection_validated'] and
                not config['production_ready'], 'Frozen uncertain-result policy mismatch.')
        self.config = config
        self.temperature = float(config['temperature'])
        require(np.isfinite(self.temperature) and self.temperature > 0, 'Invalid temperature.')
        labels = json.loads(labels_path.read_text(encoding='utf-8'))
        require(set(labels) == {str(i) for i in range(1, 503)}, 'Source label IDs changed.')
        for class_id in range(1, 503):
            record = labels[str(class_id)]
            require(record['sign_index'] == class_id - 1 and isinstance(record['sign_english'], str)
                    and isinstance(record['sign_arabic'], str), 'Source label mapping mismatch.')
        self.labels = labels
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        require(checkpoint['model_name'] == 'SignCNN502' and checkpoint['model_config'] == {'num_classes': 502}
                and checkpoint['classes'] == list(range(1, 503)) and checkpoint['input_shape'] == [32, 150]
                and checkpoint['source_fingerprint'] == FINGERPRINT
                and checkpoint['preprocessing_version'] == PREPROCESSING_VERSION and
                checkpoint['best_epoch'] == 12 and not checkpoint['test_evaluated'],
                'CNN checkpoint architecture or source mismatch.')
        self.model = common.SignCNN502(**checkpoint['model_config']).to(self.device).eval()
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)

    def _check_features(self, features):
        array = self.preprocessing.validate_features(features)
        frames = array.reshape(32, 50, 3)
        valid = frames[:, :, 2]
        require(bool(np.all((valid == 0.0) | (valid == 1.0))), 'Visibility entries must be zero or one.')
        require(bool(np.all(np.abs(frames[:, :, :2]) <= 5.000001)),
                'Coordinates must match the frozen clipped representation.')
        require(bool(np.all(frames[:, :, :2][valid == 0] == 0)),
                'Invisible landmark coordinates must be zero.')
        return np.ascontiguousarray(array, dtype=np.float32)

    def predict_features(self, features, preprocessing_version, top_k=5):
        require(preprocessing_version == PREPROCESSING_VERSION, 'Preprocessing version mismatch.')
        require(type(top_k) is int and 1 <= top_k <= 502, 'top_k must be an integer in 1..502.')
        array = self._check_features(features)
        with torch.inference_mode():
            input_tensor = torch.from_numpy(array).unsqueeze(0).to(self.device)
            logits = self.model(input_tensor)
            require(bool(torch.isfinite(logits).all()), 'Non-finite model output.')
            probability = (logits / self.temperature).softmax(dim=1)[0].cpu().numpy()
        ranking = np.argsort(-probability, kind='stable')[:top_k]
        candidates = []
        for index in ranking:
            class_id = int(index) + 1
            label = self.labels[str(class_id)]
            candidates.append({'class_id': class_id, 'label': label['sign_english'],
                               'label_arabic': label['sign_arabic'], 'score': round(float(probability[index]), 6),
                               'label_review_status': 'unverified',
                               'label_conflict': class_id in (27, 28)})
        confidence = float(probability.max())
        accepted = bool(self.config['acceptance_enabled'] and confidence >= self.config['threshold'])
        winner = candidates[0] if accepted else None
        return {'status': 'accepted' if accepted else 'uncertain', 'uncertain': not accepted,
                'class_id': winner['class_id'] if winner else None,
                'label': winner['label'] if winner else None,
                'label_arabic': winner['label_arabic'] if winner else None,
                'confidence': round(confidence, 6), 'top_k': candidates,
                'model_version': self.config['model_version'],
                'calibration_version': self.config['calibration_version']}

    def predict_pose_object(self, pose_object, top_k=5):
        features = self.preprocessing.preprocess_pose(pose_object)
        return self.predict_features(features, PREPROCESSING_VERSION, top_k)

    def predict_pose_bytes(self, pose_bytes, top_k=5):
        require(isinstance(pose_bytes, bytes) and bool(pose_bytes), 'Expected nonempty Pose bytes.')
        try:
            from pose_format import Pose
        except ImportError as exc:
            raise ImportError('Install pose-format to decode .pose files.') from exc
        return self.predict_pose_object(Pose.read(pose_bytes), top_k)

    def predict_pose_file(self, path, top_k=5):
        return self.predict_pose_bytes(Path(path).read_bytes(), top_k)


def synthetic_pose():
    frames = 17
    points = np.zeros((frames, 1, 543, 3), dtype=np.float32)
    scores = np.zeros((frames, 1, 543), dtype=np.float32)
    points[:, 0, 11, 0] = -1
    points[:, 0, 12, 0] = 1
    scores[:, 0, [11, 12]] = 1
    scores[:, 0, 501:543] = 1
    points[:, 0, 501:543, 0] = np.linspace(-0.2, 0.2, 42)
    return SimpleNamespace(body=SimpleNamespace(data=points, confidence=scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path)
    parser.add_argument('--repo-root', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--pose-file', type=Path)
    args = parser.parse_args()
    predictor = KArSL502Predictor(args.base, args.repo_root, args.device)
    example = synthetic_pose()
    features = predictor.preprocessing.preprocess_pose(example)
    feature_result = predictor.predict_features(features, PREPROCESSING_VERSION, top_k=3)
    object_result = predictor.predict_pose_object(example, top_k=3)
    require(feature_result == object_result and feature_result['uncertain'] and
            feature_result['class_id'] is None and len(feature_result['top_k']) == 3,
            'Pose preprocessing and prediction parity failed.')
    print('Model loading: PASS')
    print('Frozen preprocessing fingerprint: PASS')
    print('Synthetic Pose-to-features-to-prediction parity: PASS')
    print('Contract: PASS')
    print(json.dumps(feature_result, ensure_ascii=False, indent=2))
    if args.pose_file is not None:
        result = predictor.predict_pose_file(args.pose_file, top_k=3)
        print('Real Pose decoding and prediction: PASS')
        print(json.dumps(result, ensure_ascii=False, indent=2))
    print('Status: INFERENCE VERIFICATION COMPLETED')


if __name__ == '__main__':
    main()

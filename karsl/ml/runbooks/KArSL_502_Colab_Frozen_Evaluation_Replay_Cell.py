from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

import torch

drive.mount('/content/drive')

base = Path('/content/drive/MyDrive/SIGNOVA/KArSL')
archive = base / 'ml_delivery_packages/SIGNOVA_KArSL_502_BRANCH_CANDIDATE_CODE_ONLY_EN.zip'
features = base / 'ml_expansion/artifacts/private/features_502_v1'
original = base / 'ml_expansion/experiments/cnn_502_pass_v1'
original_labels = base / 'ml_expansion/source_metadata/labels_mapping.json'
workspace = Path('/content/signova_karsl_502_branch_candidate')
project = workspace / 'SIGNOVA_KArSL_502_REVIEW'
expected_cnn = 'f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254'
expected_calibration = '448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4'
assert archive.is_file() and features.is_dir() and original_labels.is_file()
assert (features / 'shards.json').is_file()
assert (features / 'sample_qc.csv').is_file()
assert torch.cuda.is_available(), 'Select a GPU runtime in Colab.'

with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    bundle.extractall(workspace)
    manifest = json.loads(bundle.read('SIGNOVA_KArSL_502_REVIEW/manifest.json'))

for relative, expected in manifest.items():
    path = project / relative
    assert path.stat().st_size == expected['size']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], relative

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run_base = Path('/content') / f'signova_502_frozen_eval_replay_{stamp}'
destination = base / 'ml_expansion/experiments' / f'cnn_502_frozen_eval_replay_{stamp}'
assert not run_base.exists() and not destination.exists()
(run_base / 'artifacts/private').mkdir(parents=True)
(run_base / 'artifacts/private/features_502_v1').symlink_to(features, target_is_directory=True)
labels_copy = run_base / 'source_metadata/labels_mapping.json'
labels_copy.parent.mkdir(parents=True)
shutil.copy2(original_labels, labels_copy)
experiment = run_base / 'experiments/cnn_502_pass_v1'
experiment.mkdir(parents=True)
for filename in ('best_cnn_502_pass_v1.pt', 'result.json', 'experiment.json'):
    shutil.copy2(original / filename, experiment / filename)
assert hashlib.sha256((experiment / 'best_cnn_502_pass_v1.pt').read_bytes()).hexdigest() == expected_cnn

calibration_dir = experiment / 'calibration_v1'
calibration_dir.mkdir()
shutil.copy2(original / 'calibration_v1/protocol.json', calibration_dir / 'protocol.json')
scripts = project / 'karsl/ml/ml_expansion/scripts'
cache_validation = Path('/content/signova_502_frozen_calibration_cache')
print('Replaying the original validation-only calibration policy.', flush=True)
subprocess.run([
    sys.executable, '-u', str(scripts / 'signova_calibrate_cnn_502.py'),
    '--base', str(run_base), '--cache', str(cache_validation),
], check=True)

calibration_path = calibration_dir / 'calibration_config.json'
assert hashlib.sha256(calibration_path.read_bytes()).hexdigest() == expected_calibration, (
    'Calibration differs from the frozen configuration; test was not opened.'
)
calibration = json.loads(calibration_path.read_text(encoding='utf-8'))
assert calibration['acceptance_enabled'] is False and calibration['threshold'] is None
assert calibration['candidate_threshold'] is None
print('Frozen calibration SHA-256: PASS', flush=True)

test_dir = experiment / 'test_evaluation_v1'
test_dir.mkdir()
shutil.copy2(original / 'test_evaluation_v1/test_protocol.json', test_dir / 'test_protocol.json')
cache_test = Path('/content/signova_502_frozen_test_cache')
print('Replaying the already frozen test protocol without tuning.', flush=True)
subprocess.run([
    sys.executable, '-u', str(scripts / 'signova_evaluate_test_502.py'),
    '--base', str(run_base), '--cache', str(cache_test),
], check=True)

report = json.loads((test_dir / 'test_report.json').read_text(encoding='utf-8'))
primary = report['primary']
assert primary['samples'] == 24536 and primary['evaluated_true_classes'] == 500
assert primary['model_output_classes'] == 502
assert abs(primary['accuracy'] - 0.5251874796217803) < 1e-12
assert abs(primary['macro_f1'] - 0.48520812104660715) < 1e-12
assert abs(primary['top5_accuracy'] - 0.7429083795239648) < 1e-12
assert report['model_unchanged'] and report['calibration_unchanged']

destination.mkdir(parents=True)
shutil.copytree(calibration_dir, destination / 'calibration_v1')
shutil.copytree(test_dir, destination / 'test_evaluation_v1')
shutil.copy2(experiment / 'best_cnn_502_pass_v1.pt', destination / 'best_cnn_502_pass_v1.pt')
assert hashlib.sha256((destination / 'best_cnn_502_pass_v1.pt').read_bytes()).hexdigest() == expected_cnn
print('Frozen evaluation replay: PASS', flush=True)
print('Primary accuracy:', primary['accuracy'], flush=True)
print('Primary macro F1:', primary['macro_f1'], flush=True)
print('Primary top-5:', primary['top5_accuracy'], flush=True)
print('Active-policy coverage:', primary['active_policy']['coverage'], flush=True)
print('Saved output:', destination, flush=True)

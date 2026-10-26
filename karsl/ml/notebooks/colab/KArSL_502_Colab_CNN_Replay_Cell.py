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
reference_gru = base / 'ml_expansion/experiments/gru_502_pass_v1/best_gru_502_pass_v1.pt'
workspace = Path('/content/signova_karsl_502_branch_candidate')
project = workspace / 'SIGNOVA_KArSL_502_REVIEW'
expected_gru_sha256 = 'ba1797040c21de7b5f05039d72bac82e7fdce8e7860df02e445281af6ae577da'
assert torch.cuda.is_available(), 'Select a GPU runtime in Colab.'
assert archive.is_file() and features.is_dir() and reference_gru.is_file()
assert (features / 'shards.json').is_file()

with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    bundle.extractall(workspace)
    manifest = json.loads(bundle.read('SIGNOVA_KArSL_502_REVIEW/manifest.json'))

for relative, expected in manifest.items():
    path = project / relative
    assert path.stat().st_size == expected['size']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], relative

assert hashlib.sha256(reference_gru.read_bytes()).hexdigest() == expected_gru_sha256
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run_base = Path('/content') / f'signova_cnn_502_clean_replay_{stamp}'
output = run_base / 'experiments/cnn_502_pass_v1'
assert not run_base.exists()
(run_base / 'artifacts/private').mkdir(parents=True)
(run_base / 'artifacts/private/features_502_v1').symlink_to(features, target_is_directory=True)
gru_copy = run_base / 'experiments/gru_502_pass_v1/best_gru_502_pass_v1.pt'
gru_copy.parent.mkdir(parents=True)
shutil.copy2(reference_gru, gru_copy)
assert hashlib.sha256(gru_copy.read_bytes()).hexdigest() == expected_gru_sha256

script = project / 'karsl/ml/ml_expansion/scripts/signova_compare_cnn_502.py'
cache = Path('/content/signova_502_cnn_clean_replay_cache')
destination = base / 'ml_expansion/experiments' / f'cnn_502_clean_replay_{stamp}'
assert not destination.exists()
print('CUDA device:', torch.cuda.get_device_name(0), flush=True)
print('Frozen features:', features, flush=True)
print('Original reference GRU:', reference_gru, flush=True)
print('New CNN output:', destination, flush=True)
print('Test split will not be read by this script.', flush=True)

subprocess.run([
    sys.executable, '-u', str(script),
    '--base', str(run_base),
    '--cache', str(cache),
], check=True)

result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
assert result['test_evaluated'] is False
assert result['reference_gru_unchanged'] is True
destination.mkdir(parents=True)
for filename in (
    'best_cnn_502_pass_v1.pt', 'resume_last.pt', 'history.csv',
    'result.json', 'experiment.json', 'runtime_sessions.json',
    'gru_reload_verification.json', 'validation_comparison.csv',
):
    shutil.copy2(output / filename, destination / filename)

copied_checkpoint = destination / 'best_cnn_502_pass_v1.pt'
assert hashlib.sha256(copied_checkpoint.read_bytes()).hexdigest() == result['checkpoint_sha256']
print('CNN replay:', result['status'], flush=True)
print('Best epoch:', result['best_epoch'], flush=True)
print('Validation metrics:', result['validation_metrics'], flush=True)
print('Checkpoint SHA-256:', result['checkpoint_sha256'], flush=True)
print('Saved output:', destination, flush=True)
print('Test data accessed: NO', flush=True)

from google.colab import drive
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import zipfile

import torch

drive.mount('/content/drive')

base = Path('/content/drive/MyDrive/SIGNOVA/KArSL')
archive = base / 'ml_delivery_packages/SIGNOVA_KArSL_502_BRANCH_CANDIDATE_CODE_ONLY_EN.zip'
features = base / 'ml_expansion/artifacts/private/features_502_v1'
workspace = Path('/content/signova_karsl_502_branch_candidate')
project = workspace / 'SIGNOVA_KArSL_502_REVIEW'
assert archive.is_file(), f'Missing code archive: {archive}'
assert (features / 'shards.json').is_file(), f'Missing frozen feature manifest: {features}'
assert (features / 'preprocessing_report.json').is_file()
assert torch.cuda.is_available(), 'Select a GPU runtime in Colab before starting full training.'

with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None, 'The archive failed its integrity check.'
    bundle.extractall(workspace)
    manifest = json.loads(bundle.read('SIGNOVA_KArSL_502_REVIEW/manifest.json'))

for relative, expected in manifest.items():
    path = project / relative
    assert path.stat().st_size == expected['size']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected['sha256'], relative

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = base / 'ml_expansion/experiments' / f'gru_502_clean_replay_{stamp}'
cache = Path('/content/signova_502_gru_clean_replay_cache')
script = project / 'karsl/ml/training/train_gru_502.py'
print('CUDA device:', torch.cuda.get_device_name(0), flush=True)
print('Frozen features:', features, flush=True)
print('New output:', output, flush=True)
print('Test split will not be read by this script.', flush=True)

subprocess.run([
    sys.executable, '-u', str(script),
    '--features-root', str(features),
    '--cache', str(cache),
    '--output', str(output),
    '--device', 'cuda',
], check=True)

result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
assert result['test_evaluated'] is False
print('Full GRU replay:', result['status'], flush=True)
print('Best epoch:', result['best_epoch'], flush=True)
print('Validation metrics:', result['validation_metrics'], flush=True)
print('Checkpoint SHA-256:', result['checkpoint_sha256'], flush=True)
print('Output directory:', output, flush=True)

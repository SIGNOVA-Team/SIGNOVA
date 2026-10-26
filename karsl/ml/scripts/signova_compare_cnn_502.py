import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

FINGERPRINT = '5d0d43950dec9c69bef2766e48550463a027b49a44748631b976f211c77e7c4a'
PREPROCESSING = 'pose_v1_32x50_xy_visibility_shoulder_conf05'
GRU_SHA256 = 'ba1797040c21de7b5f05039d72bac82e7fdce8e7860df02e445281af6ae577da'
CLASSES = list(range(1, 503))
EXPERIMENT = 'cnn_502_pass_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_csv(path, rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(output.getvalue(), encoding='utf-8')
    os.replace(temporary, path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class SignGRU502(nn.Module):
    def __init__(self, hidden_size=128, num_classes=502):
        super().__init__()
        self.gru = nn.GRU(150, hidden_size, num_layers=2, batch_first=True,
                          bidirectional=True, dropout=0.25)
        self.head = nn.Sequential(nn.LayerNorm(hidden_size * 4), nn.Dropout(0.30),
                                  nn.Linear(hidden_size * 4, num_classes))

    def forward(self, x):
        sequence, _ = self.gru(x)
        return self.head(torch.cat((sequence.mean(dim=1), sequence.amax(dim=1)), dim=1))


class SignCNN502(nn.Module):
    def __init__(self, num_classes=502):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(150, 128, 5, padding=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Conv1d(128, 192, 3, padding=1, bias=False),
            nn.BatchNorm1d(192), nn.ReLU(), nn.Dropout(0.20),
            nn.Conv1d(192, 256, 3, padding=1, bias=False),
            nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.LayerNorm(512), nn.Dropout(0.30),
                                  nn.Linear(512, num_classes))

    def forward(self, x):
        sequence = self.encoder(x.transpose(1, 2))
        return self.head(torch.cat((sequence.mean(dim=2), sequence.amax(dim=2)), dim=1))


def load_pass_split(source, cache, manifest, split):
    require(split in ('train', 'validation'), 'Only train and validation are allowed.')
    records = [item for item in manifest['shards'] if item['split'] == split]
    require(bool(records), f'No shards for {split}.')
    xs, ys, paths, signer_ids, raw_hashes, feature_hashes = [], [], [], [], [], []
    qc = Counter()
    seen_shards = set()
    for number, item in enumerate(records, 1):
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                len(relative.parts) == 2 and relative.parts[0] == split and
                relative.suffix == '.npz', 'Unsafe or unexpected shard path.')
        require(str(relative) not in seen_shards, 'Repeated shard in manifest.')
        seen_shards.add(str(relative))
        original, local = source / relative, cache / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.is_file() or sha256(local) != item['sha256']:
            temporary = local.with_suffix('.copying')
            shutil.copyfile(original, temporary)
            require(sha256(temporary) == item['sha256'], f'Changed source shard: {relative}')
            os.replace(temporary, local)
        with np.load(local, allow_pickle=False) as part:
            require(str(part['fingerprint'].item()) == FINGERPRINT, 'Wrong shard fingerprint.')
            require(str(part['preprocessing_version'].item()) == PREPROCESSING,
                    'Wrong preprocessing version.')
            require(str(part['split'].item()) == split, 'Wrong embedded split.')
            audit = json.loads(str(part['audit_json'].item()))
            require(len(audit) == item['expected_samples'], 'Audit length mismatch.')
            require(all(r['status'] in ('PASS', 'REVIEW', 'FAILED') for r in audit),
                    'Unknown QC status.')
            require(all(r['split'] == split for r in audit), 'Audit split mismatch.')
            qc.update(r['status'] for r in audit)
            prepared = [r for r in audit if r['status'] != 'FAILED']
            x, labels, sample_paths = part['X'], part['labels'], part['paths'].astype(str)
            require(x.shape == (item['prepared_samples'], 32, 150) and
                    x.dtype == np.float32 and np.isfinite(x).all(), 'Invalid feature tensor.')
            require(labels.shape == sample_paths.shape == (len(prepared),),
                    'Prepared sample count mismatch.')
            require(sample_paths.tolist() == [r['pose_file'] for r in prepared] and
                    labels.tolist() == [r['sign_id'] for r in prepared],
                    'Feature/label/path alignment mismatch.')
            keep = np.array([r['status'] == 'PASS' for r in prepared], dtype=bool)
            kept = [r for r in prepared if r['status'] == 'PASS']
            xs.append(x[keep])
            ys.append(labels[keep].astype(np.int64))
            paths.extend(sample_paths[keep].tolist())
            signer_ids.extend(r['signer_id'] for r in kept)
            raw_hashes.extend(r['raw_sha256'] for r in kept)
            feature_hashes.extend(r['feature_sha256'] for r in kept)
        if number % 10 == 0 or number == len(records):
            print(f'{split}: verified {number}/{len(records)} shards', flush=True)
    x, labels = np.concatenate(xs), np.concatenate(ys)
    require(len(x) == len(labels) == len(paths) == len(set(paths)), 'Duplicate sample path.')
    require(set(labels.tolist()) == set(CLASSES), f'{split}: missing or unexpected classes.')
    identity = hashlib.sha256(json.dumps(list(zip(paths, labels.tolist())),
                                         separators=(',', ':')).encode()).hexdigest()
    info = {'samples': len(labels), 'qc_counts': dict(qc), 'signers': sorted(set(signer_ids)),
            'sample_order_sha256': identity}
    return x, labels - 1, paths, set(raw_hashes), set(feature_hashes), info


def make_loader(x, y, batch_size, shuffle, device):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


def evaluate(model, loader, device, num_classes=502):
    model.eval()
    truth, probabilities = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device, non_blocking=True))
            require(bool(torch.isfinite(logits).all()), 'Non-finite model output.')
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            truth.append(y.numpy())
    y, probs = np.concatenate(truth), np.concatenate(probabilities)
    labels = np.arange(num_classes)
    return {'accuracy': float(accuracy_score(y, probs.argmax(axis=1))),
            'macro_f1': float(f1_score(y, probs.argmax(axis=1), average='macro',
                                      labels=labels, zero_division=0)),
            'top5_accuracy': float(top_k_accuracy_score(y, probs, k=5, labels=labels))}


def check_metrics(actual, expected, tolerance=1e-4):
    differences = {key: abs(actual[key] - expected[key]) for key in actual}
    require(all(value <= tolerance for value in differences.values()),
            f'Checkpoint validation mismatch: {differences}')
    return differences


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def get_rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def set_rng_state(state):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        require(torch.cuda.is_available(), 'CUDA is required to resume this run.')
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpoint_payload(best_state, best_epoch, metrics, config):
    return {'model_state_dict': best_state, 'model_name': 'SignCNN502',
            'model_config': config['model_config'], 'input_shape': [32, 150],
            'classes': CLASSES, 'source_fingerprint': FINGERPRINT,
            'preprocessing_version': PREPROCESSING, 'selection_metric': 'validation_macro_f1',
            'best_epoch': best_epoch, 'validation_metrics': metrics,
            'test_evaluated': False, 'experiment_config': config}


def train(model, train_loader, val_loader, device, output, config):
    output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = output / 'resume_last.pt', output / 'best_cnn_502_pass_v1.pt'
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    history, best_state, best_metrics = [], None, None
    best_epoch, stale, elapsed = 0, 0, 0.0
    if last_path.exists():
        saved = torch.load(last_path, map_location='cpu', weights_only=True)
        require(saved['config'] == config, 'Resume configuration differs from the saved run.')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        history, best_state, best_metrics = saved['history'], saved['best_state'], saved['best_metrics']
        best_epoch, stale, elapsed = saved['best_epoch'], saved['stale_epochs'], saved['training_seconds']
        set_rng_state(saved['rng_state'])
        print(f'Restored completed epoch: {len(history)}', flush=True)
    else:
        require(not best_path.exists() and not (output / 'history.csv').exists(),
                'Existing CNN outputs have no resume checkpoint; refusing to overwrite.')
    if history:
        save_torch(best_path, checkpoint_payload(best_state, best_epoch, best_metrics, config))
        save_csv(output / 'history.csv', history)
    for epoch in range(len(history) + 1, config['epochs_max'] + 1):
        if stale >= config['patience']:
            break
        started = time.perf_counter()
        model.train()
        total_loss, samples = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            require(bool(torch.isfinite(loss)), 'Non-finite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            samples += len(y)
        metrics = evaluate(model, val_loader, device, config['model_config']['num_classes'])
        row = {'epoch': epoch, 'train_loss': total_loss / samples, **metrics}
        history.append(row)
        if best_metrics is None or metrics['macro_f1'] > best_metrics['macro_f1'] + config['min_delta']:
            best_epoch, stale, best_metrics, best_state = epoch, 0, metrics, cpu_state(model)
        else:
            stale += 1
        elapsed += time.perf_counter() - started
        state = {'config': config, 'model_state_dict': cpu_state(model),
                 'optimizer_state_dict': optimizer.state_dict(), 'history': history,
                 'best_state': best_state, 'best_metrics': best_metrics, 'best_epoch': best_epoch,
                 'stale_epochs': stale, 'training_seconds': elapsed, 'rng_state': get_rng_state()}
        save_torch(last_path, state)
        save_torch(best_path, checkpoint_payload(best_state, best_epoch, best_metrics, config))
        save_csv(output / 'history.csv', history)
        print(f"Epoch {epoch:02d}: loss={row['train_loss']:.4f}, val_acc={metrics['accuracy']:.4f}, "
              f"val_macro_f1={metrics['macro_f1']:.4f}, val_top5={metrics['top5_accuracy']:.4f}", flush=True)
    require(bool(history), 'Training produced no epochs.')
    return {'epochs_run': len(history), 'best_epoch': best_epoch, 'validation_metrics': best_metrics,
            'training_seconds': elapsed, 'early_stopped': stale >= config['patience']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path,
                        default=Path('/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion'))
    parser.add_argument('--cache', type=Path, default=Path('/content/signova_502_train_cache'))
    args = parser.parse_args()
    require(torch.cuda.is_available(), 'A Colab GPU runtime with CUDA-enabled PyTorch is required.')
    device = torch.device('cuda')
    torch.set_num_threads(2)
    seed_all(42)
    source = args.base / 'artifacts/private/features_502_v1'
    output = args.base / 'experiments' / EXPERIMENT
    gru_path = args.base / 'experiments/gru_502_pass_v1/best_gru_502_pass_v1.pt'
    require(sha256(gru_path) == GRU_SHA256, 'The reference GRU checkpoint has changed.')
    report = json.loads((source / 'preprocessing_report.json').read_text())
    manifest = json.loads((source / 'shards.json').read_text())
    require(report['fingerprint'] == manifest['fingerprint'] == FINGERPRINT, 'Wrong frozen source.')
    require(report['expected_samples'] == 75515 and report['source_classes'] == 502 and
            report['feature_shape'] == [32, 150] and len(manifest['shards']) == 149,
            'Source report does not match this experiment.')
    require(report['raw_cross_split_duplicate_groups'] == 0, 'Cross-split raw duplicates need review.')
    print('Loading the frozen train/validation PASS samples.', flush=True)
    xt, yt, pt, rt, ft, train_info = load_pass_split(source, args.cache, manifest, 'train')
    xv, yv, pv, rv, fv, val_info = load_pass_split(source, args.cache, manifest, 'validation')
    require(train_info['qc_counts'] == {'PASS': 25032, 'REVIEW': 55} and
            val_info['qc_counts'] == {'PASS': 25254, 'REVIEW': 58, 'FAILED': 4}, 'QC counts changed.')
    require(train_info['signers'] == [3] and val_info['signers'] == [2], 'Signer split changed.')
    require(set(pt).isdisjoint(pv) and rt.isdisjoint(rv) and ft.isdisjoint(fv),
            'Train/validation sample or content overlap.')
    val_loader = make_loader(xv, yv, 256, False, device)
    checkpoint = torch.load(gru_path, map_location='cpu', weights_only=True)
    require(checkpoint['model_name'] == 'SignGRU502' and checkpoint['classes'] == CLASSES and
            checkpoint['source_fingerprint'] == FINGERPRINT and
            checkpoint['preprocessing_version'] == PREPROCESSING and
            checkpoint['input_shape'] == [32, 150] and not checkpoint['test_evaluated'],
            'GRU checkpoint metadata mismatch.')
    gru = SignGRU502(**checkpoint['model_config']).to(device)
    gru.load_state_dict(checkpoint['model_state_dict'], strict=True)
    gru_metrics = evaluate(gru, val_loader, device)
    metric_differences = check_metrics(gru_metrics, checkpoint['validation_metrics'])
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / 'gru_reload_verification.json', {
        'checkpoint_sha256': GRU_SHA256, 'best_epoch': checkpoint['best_epoch'],
        'saved_metrics': checkpoint['validation_metrics'], 'reloaded_metrics': gru_metrics,
        'absolute_differences': metric_differences, 'absolute_tolerance': 1e-4,
        'validation_samples': len(yv), 'test_evaluated': False, 'status': 'PASS'})
    print('GRU checkpoint reload and full validation: PASS', flush=True)
    print(json.dumps(gru_metrics, indent=2), flush=True)
    del gru, checkpoint
    torch.cuda.empty_cache()
    seed_all(42)
    cnn = SignCNN502().to(device)
    train_loader = make_loader(xt, yt, 128, True, device)
    config = {'experiment': EXPERIMENT, 'model_name': 'SignCNN502',
              'model_config': {'num_classes': 502}, 'source_fingerprint': FINGERPRINT,
              'preprocessing_version': PREPROCESSING, 'classes': CLASSES, 'input_shape': [32, 150],
              'seed': 42, 'batch_size': 128, 'validation_batch_size': 256, 'epochs_max': 25,
              'patience': 6, 'min_delta': 1e-6, 'learning_rate': 0.001, 'weight_decay': 0.001,
              'optimizer': 'AdamW', 'gradient_clip_norm': 1.0, 'selection_metric': 'validation_macro_f1',
              'augmentation': None, 'train': train_info, 'validation': val_info,
              'reference_gru_sha256': GRU_SHA256, 'script_sha256': sha256(Path(__file__)),
              'test_evaluated': False, 'label_semantics_verified': False,
              'known_test_issue': '50 duplicate pairs have conflicting labels 27 and 28; historical exposure of 450 test samples.',
              'deterministic_algorithms': True}
    runtime = {'started_utc': datetime.now(timezone.utc).isoformat(),
               'torch': str(torch.__version__), 'numpy': np.__version__,
               'python': platform.python_version(), 'cuda_build': str(torch.version.cuda),
               'gpu': torch.cuda.get_device_name(0), 'cudnn': torch.backends.cudnn.version()}
    config_path = output / 'experiment.json'
    if config_path.exists():
        require(json.loads(config_path.read_text()) == config,
                'Existing experiment configuration differs; no training outputs were overwritten.')
    else:
        save_json(config_path, config)
    sessions_path = output / 'runtime_sessions.json'
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append(runtime)
    save_json(sessions_path, sessions)
    print(f'Train PASS: {len(yt)} | Validation PASS: {len(yv)} | Classes: 502', flush=True)
    print(f'Model: SignCNN502 | Parameters: {sum(p.numel() for p in cnn.parameters())}', flush=True)
    print('Device: cuda | Test features loaded: NO', flush=True)
    result = train(cnn, train_loader, val_loader, device, output, config)
    best_path = output / 'best_cnn_502_pass_v1.pt'
    best = torch.load(best_path, map_location='cpu', weights_only=True)
    cnn.load_state_dict(best['model_state_dict'], strict=True)
    cnn_metrics = evaluate(cnn, val_loader, device)
    check_metrics(cnn_metrics, best['validation_metrics'])
    require(sha256(gru_path) == GRU_SHA256, 'Reference GRU changed during the run.')
    rows = [{'model': 'GRU', 'experiment': 'gru_502_pass_v1', 'classes': 502,
             'validation_samples': len(yv), 'best_epoch': 17, **gru_metrics},
            {'model': 'CNN', 'experiment': EXPERIMENT, 'classes': 502,
             'validation_samples': len(yv), 'best_epoch': best['best_epoch'], **cnn_metrics}]
    save_csv(output / 'validation_comparison.csv', rows)
    preferred = 'CNN' if cnn_metrics['macro_f1'] > gru_metrics['macro_f1'] + 1e-6 else 'GRU'
    save_json(output / 'result.json', {**result, 'comparison': rows, 'preferred_by_validation_macro_f1': preferred,
              'tie_policy': 'Keep GRU when CNN macro F1 improvement is at most 1e-6.',
              'checkpoint_sha256': sha256(best_path), 'reference_gru_unchanged': True,
              'cnn_checkpoint_reload': 'PASS', 'test_evaluated': False, 'calibration_performed': False,
              'completed_utc': datetime.now(timezone.utc).isoformat(), 'status': 'COMPARISON_COMPLETED'})
    print('\n=== VALIDATION COMPARISON ===', flush=True)
    for row in rows:
        print(f"{row['model']}: Accuracy={row['accuracy']:.4%}, Macro F1={row['macro_f1']:.6f}, "
              f"Top-5={row['top5_accuracy']:.4%}, Best epoch={row['best_epoch']}", flush=True)
    print(f'Preferred by validation macro F1: {preferred}', flush=True)
    print(f'CNN checkpoint: {best_path}', flush=True)
    print(f'Resume checkpoint: {output / "resume_last.pt"}', flush=True)
    print(f'Comparison report: {output / "validation_comparison.csv"}', flush=True)
    print('Original GRU checkpoint: UNCHANGED\nTest evaluated: NO\nStatus: COMPARISON COMPLETED', flush=True)


if __name__ == '__main__':
    main()

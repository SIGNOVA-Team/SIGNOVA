import argparse
import hashlib
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


MODEL_SOURCE_SHA256 = 'f6e0f60a817de984fc356b7bbd7edcaff72ffbc63e046fe09175564cc6633b5e'
EXPERIMENT = 'gru_502_pass_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    p = Path(path)
    data = p.read_bytes()
    if p.suffix in {'.py', '.json', '.txt', '.csv', '.md'}:
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def load_frozen_model_source(path):
    require(sha256(path) == MODEL_SOURCE_SHA256, 'Frozen model/loader source has changed.')
    spec = importlib.util.spec_from_file_location('signova_frozen_502_models', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def checkpoint_payload(model, best_epoch, metrics, frozen, config):
    return {
        'model_state_dict': frozen.cpu_state(model),
        'model_name': 'SignGRU502',
        'model_config': {'hidden_size': 128, 'num_classes': 502},
        'input_shape': [32, 150],
        'classes': frozen.CLASSES,
        'source_fingerprint': frozen.FINGERPRINT,
        'preprocessing_version': frozen.PREPROCESSING,
        'selection_metric': 'validation_macro_f1',
        'best_epoch': best_epoch,
        'validation_metrics': metrics,
        'test_evaluated': False,
        'experiment_config': config,
    }


def train_gru(frozen, train_x, train_y, val_x, val_y, output, device, epochs=25, patience=6):
    require(epochs >= 1 and patience >= 1, 'Epochs and patience must be positive.')
    require(train_x.ndim == val_x.ndim == 3 and train_x.shape[1:] == val_x.shape[1:] == (32, 150),
            'Wrong feature shape.')
    require(len(train_x) == len(train_y) and len(val_x) == len(val_y) and
            len(train_x) > 0 and len(val_x) > 0, 'Empty or unaligned split.')
    require(np.isfinite(train_x).all() and np.isfinite(val_x).all(), 'Nonfinite features.')
    require(np.issubdtype(train_y.dtype, np.integer) and
            np.issubdtype(val_y.dtype, np.integer) and
            bool(np.all((0 <= train_y) & (train_y < 502))) and
            bool(np.all((0 <= val_y) & (val_y < 502))), 'Invalid encoded labels.')
    output = Path(output)
    require(not output.exists(), 'Output directory already exists; refusing to overwrite an experiment.')
    output.mkdir(parents=True)
    model = frozen.SignGRU502(hidden_size=128, num_classes=502).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    train_loader = frozen.make_loader(train_x, train_y, 128, True, device)
    val_loader = frozen.make_loader(val_x, val_y, 256, False, device)
    config = {
        'experiment': EXPERIMENT,
        'source_fingerprint': frozen.FINGERPRINT,
        'preprocessing_version': frozen.PREPROCESSING,
        'classes': frozen.CLASSES,
        'input_shape': [32, 150],
        'seed': 42,
        'batch_size': 128,
        'validation_batch_size': 256,
        'epochs_max': epochs,
        'patience': patience,
        'min_delta': 1e-6,
        'optimizer': 'AdamW',
        'learning_rate': 0.001,
        'weight_decay': 0.001,
        'gradient_clip_norm': 1.0,
        'selection_metric': 'validation_macro_f1',
        'train_pass': len(train_y),
        'validation_pass': len(val_y),
        'test_accessed': False,
    }
    save_json(output / 'experiment.json', config)
    history = []
    best_f1 = -1.0
    stale = 0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            require(bool(torch.isfinite(loss)), 'Nonfinite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_total += float(loss.detach()) * len(y)
        metrics = frozen.evaluate(model, val_loader, device)
        row = {'epoch': epoch, 'train_loss': loss_total / len(train_y), **metrics}
        history.append(row)
        frozen.save_csv(output / 'history.csv', history)
        print(f"Epoch {epoch:02d}: loss={row['train_loss']:.4f}, "
              f"val_acc={row['accuracy']:.4f}, val_macro_f1={row['macro_f1']:.4f}, "
              f"val_top5={row['top5_accuracy']:.4f}", flush=True)
        if metrics['macro_f1'] > best_f1 + 1e-6:
            best_f1, stale = metrics['macro_f1'], 0
            save_torch(output / 'best_gru_502_pass_v1.pt',
                       checkpoint_payload(model, epoch, metrics, frozen, config))
        else:
            stale += 1
            if stale >= patience:
                break
    checkpoint_path = output / 'best_gru_502_pass_v1.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    reloaded_metrics = frozen.evaluate(model, val_loader, device)
    frozen.check_metrics(reloaded_metrics, checkpoint['validation_metrics'])
    result = {
        'status': 'COMPLETE',
        'best_epoch': checkpoint['best_epoch'],
        'epochs_run': len(history),
        'validation_metrics': reloaded_metrics,
        'training_seconds': time.perf_counter() - start,
        'checkpoint_sha256': sha256(checkpoint_path),
        'test_evaluated': False,
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'numpy': np.__version__, 'device': str(device),
                    'completed_utc': datetime.now(timezone.utc).isoformat()},
    }
    save_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--frozen-model-source', type=Path,
                        default=Path(__file__).resolve().parents[1] /
                        'scripts/signova_compare_cnn_502.py')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.device == 'cuda':
        require(torch.cuda.is_available(), 'CUDA is unavailable; pass --device cpu if appropriate.')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    frozen = load_frozen_model_source(args.frozen_model_source)
    frozen.seed_all(42)
    report = json.loads((args.features_root / 'preprocessing_report.json').read_text())
    manifest = json.loads((args.features_root / 'shards.json').read_text())
    require(report['fingerprint'] == manifest['fingerprint'] == frozen.FINGERPRINT,
            'Wrong frozen source.')
    require(report['expected_samples'] == 75515 and report['source_classes'] == 502 and
            report['feature_shape'] == [32, 150] and len(manifest['shards']) == 149 and
            report['raw_cross_split_duplicate_groups'] == 0, 'Source report changed.')
    x_train, y_train, train_paths, train_raw, train_features, train_info = frozen.load_pass_split(
        args.features_root, args.cache, manifest, 'train')
    x_val, y_val, val_paths, val_raw, val_features, val_info = frozen.load_pass_split(
        args.features_root, args.cache, manifest, 'validation')
    require(train_info['qc_counts'] == {'PASS': 25032, 'REVIEW': 55} and
            val_info['qc_counts'] == {'PASS': 25254, 'REVIEW': 58, 'FAILED': 4},
            'Frozen QC counts changed.')
    require(train_info['signers'] == [3] and val_info['signers'] == [2],
            'Signer split changed.')
    require(set(train_paths).isdisjoint(val_paths) and train_raw.isdisjoint(val_raw) and
            train_features.isdisjoint(val_features), 'Train/validation overlap.')
    result = train_gru(frozen, x_train, y_train, x_val, y_val, args.output, device)
    print(json.dumps(result, indent=2), flush=True)
    print('Test data accessed: NO', flush=True)


if __name__ == '__main__':
    main()

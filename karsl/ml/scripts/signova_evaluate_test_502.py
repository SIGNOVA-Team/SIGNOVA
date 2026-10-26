import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score

CHECKPOINT_SHA256 = 'f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254'
CALIBRATION_CONFIG_SHA256 = '448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4'
QC_SHA256 = '25c2345329cd8a510ae522a88e51236f779150f9d43024f41c43d32fee6178eb'
LABELS_SHA256 = '3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a'
MODULE_HASHES = {
    'signova_compare_cnn_502.py': 'f6e0f60a817de984fc356b7bbd7edcaff72ffbc63e046fe09175564cc6633b5e',
    'signova_calibrate_cnn_502.py': 'a83f47c0de8d3d0192cbc18dacbfc7f5cee5fc95bbe3b0ae26e1b42d0a5fbbad',
}
EXCLUDED_CLASS_IDS = {27, 28}
EVALUATED_CLASS_IDS = [i for i in range(1, 503) if i not in EXCLUDED_CLASS_IDS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_module(name):
    path = Path(__file__).with_name(name)
    require(sha256(path) == MODULE_HASHES[name], f'Changed dependency: {name}')
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_bool(value):
    text = str(value).strip().lower()
    require(text in ('true', 'false'), f'Invalid Boolean value: {value}')
    return text == 'true'


def read_qc(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['sign_id'] = int(row['sign_id'])
        row['signer_id'] = int(row['signer_id'])
        row['sample_id'] = int(row['sample_id'])
        row['previously_evaluated_test'] = parse_bool(row['previously_evaluated_test'])
        require(row['split'] in ('train', 'validation', 'test'), 'Unexpected source split.')
        require(row['status'] in ('PASS', 'REVIEW', 'FAILED'), 'Unexpected source QC status.')
    require(len(rows) == len({row['pose_file'] for row in rows}), 'Duplicate source sample identity.')
    return rows


def build_policy(rows):
    test_rows = [row for row in rows if row['split'] == 'test']
    require(bool(test_rows), 'No test audit rows.')
    audit = []
    for row in test_rows:
        valid = row['status'] == 'PASS'
        ambiguous = row['sign_id'] in EXCLUDED_CLASS_IDS
        historical = row['previously_evaluated_test']
        reasons = []
        if not valid:
            reasons.append('QC_' + row['status'])
        if ambiguous:
            reasons.append('UNRESOLVED_CLASS_27_28')
        if historical:
            reasons.append('HISTORICAL_TEST_EXPOSURE')
        audit.append({'pose_file': row['pose_file'], 'class_id': row['sign_id'],
                      'signer_id': row['signer_id'], 'qc_status': row['status'],
                      'historically_evaluated': historical,
                      'primary_included': valid and not ambiguous and not historical,
                      'secondary_included': valid and not ambiguous,
                      'primary_exclusion_reasons': '|'.join(reasons)})
    primary = {row['pose_file'] for row in audit if row['primary_included']}
    secondary = {row['pose_file'] for row in audit if row['secondary_included']}
    class_counts = Counter(row['class_id'] for row in audit if row['primary_included'])
    counts = {'test_source_samples': len(test_rows), 'qc_counts': dict(Counter(row['status'] for row in test_rows)),
              'excluded_class_samples': sum(row['sign_id'] in EXCLUDED_CLASS_IDS for row in test_rows),
              'historically_evaluated_samples': sum(row['previously_evaluated_test'] for row in test_rows),
              'historical_exclusions_after_qc_and_class_rules': len(secondary - primary),
              'primary_samples': len(primary), 'secondary_samples': len(secondary),
              'primary_classes': len(class_counts), 'primary_minimum_class_support': min(class_counts.values()),
              'primary_classes_with_support_below_10': sum(count < 10 for count in class_counts.values())}
    return audit, primary, secondary, counts


def verify_leakage(rows, selected_paths):
    groups = {split: [row for row in rows if row['split'] == split and row['status'] == 'PASS']
              for split in ('train', 'validation', 'test')}
    for field in ('pose_file', 'raw_sha256', 'feature_sha256', 'signer_id'):
        values = {split: {row[field] for row in group} for split, group in groups.items()}
        for left, right in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            require(values[left].isdisjoint(values[right]), f'Cross-split overlap in {field}: {left}/{right}')
    selected = [row for row in groups['test'] if row['pose_file'] in selected_paths]
    for field in ('raw_sha256', 'feature_sha256'):
        require(len(selected) == len({row[field] for row in selected}),
                f'Duplicate {field} within the selected test subset requires review.')


def load_test_features(source, cache, manifest, qc_rows, selected_paths, common):
    records = [item for item in manifest['shards'] if item['split'] == 'test']
    require(len(records) == 50, 'Unexpected number of test shards.')
    expected = {row['pose_file']: row for row in qc_rows if row['split'] == 'test'}
    xs, ys, paths, seen = [], [], [], set()
    shard_names = set()
    for number, item in enumerate(records, 1):
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and
                len(relative.parts) == 2 and relative.parts[0] == 'test' and relative.suffix == '.npz',
                'Unsafe test shard path.')
        require(str(relative) not in shard_names, 'Repeated test shard.')
        shard_names.add(str(relative))
        local = cache / relative
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.is_file() or sha256(local) != item['sha256']:
            temporary = local.with_suffix('.copying')
            shutil.copyfile(source / relative, temporary)
            require(sha256(temporary) == item['sha256'], 'Test source shard checksum mismatch.')
            os.replace(temporary, local)
        with np.load(local, allow_pickle=False) as part:
            require(str(part['split'].item()) == 'test' and str(part['fingerprint'].item()) == common.FINGERPRINT
                    and str(part['preprocessing_version'].item()) == common.PREPROCESSING,
                    'Test shard version mismatch.')
            audit = json.loads(str(part['audit_json'].item()))
            require(len(audit) == item['expected_samples'], 'Test shard audit count mismatch.')
            for row in audit:
                path = row['pose_file']
                require(path in expected and path not in seen, 'Unexpected or repeated test sample.')
                seen.add(path)
                reference = expected[path]
                for field in ('sign_id', 'signer_id', 'sample_id', 'split', 'status', 'raw_sha256', 'feature_sha256'):
                    require(row[field] == reference[field], f'Test audit mismatch: {path}, {field}')
                require(parse_bool(row['previously_evaluated_test']) == reference['previously_evaluated_test'],
                        'Historical test exposure flag mismatch.')
            prepared = [row for row in audit if row['status'] != 'FAILED']
            x, labels, sample_paths = part['X'], part['labels'], part['paths'].astype(str)
            require(x.shape == (item['prepared_samples'], 32, 150) and x.dtype == np.float32
                    and np.isfinite(x).all(), 'Invalid test feature tensor.')
            require(labels.shape == sample_paths.shape == (len(prepared),) and
                    labels.tolist() == [row['sign_id'] for row in prepared] and
                    sample_paths.tolist() == [row['pose_file'] for row in prepared], 'Test feature alignment mismatch.')
            keep = np.array([path in selected_paths for path in sample_paths], dtype=bool)
            xs.append(x[keep])
            ys.append(labels[keep].astype(np.int64) - 1)
            paths.extend(sample_paths[keep].tolist())
        if number % 10 == 0 or number == len(records):
            print(f'test: verified {number}/{len(records)} shards', flush=True)
    require(seen == set(expected), 'Incomplete test shard coverage.')
    require(set(paths) == selected_paths and len(paths) == len(selected_paths), 'Selected test samples changed.')
    return np.concatenate(xs), np.concatenate(ys), paths


def compute_metrics(logits, y, temperature, calibration, evaluated_ids=EVALUATED_CLASS_IDS):
    evaluated_indices = np.array(evaluated_ids, dtype=np.int64) - 1
    require(set(y.tolist()) == set(evaluated_indices.tolist()), 'Missing or unexpected evaluated class support.')
    prediction = logits.argmax(axis=1)
    before, _ = calibration.calibration_metrics(logits, y, 1.0)
    after, _ = calibration.calibration_metrics(logits, y, temperature)
    confidence = softmax(logits.astype(np.float64) / temperature, axis=1).max(axis=1)
    return {'samples': len(y), 'evaluated_true_classes': len(evaluated_indices),
            'model_output_classes': logits.shape[1], 'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, average='macro', labels=evaluated_indices, zero_division=0)),
            'top5_accuracy': float(top_k_accuracy_score(y, logits, k=5, labels=np.arange(logits.shape[1]))),
            'predictions_into_excluded_classes': int(np.isin(prediction + 1, list(EXCLUDED_CLASS_IDS)).sum()),
            'calibration_before': before, 'calibration_after': after,
            'active_policy': calibration.selective_metrics(confidence, prediction == y, None)}


def export_class_reports(output, logits, y, paths, temperature, labels, calibration):
    all_indices = np.arange(502)
    prediction = logits.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=all_indices, zero_division=0)
    rows = []
    for index in all_indices:
        class_id = int(index + 1)
        evaluated = class_id not in EXCLUDED_CLASS_IDS
        rows.append({'class_id': class_id, 'label_english': labels[str(class_id)]['sign_english'],
                     'label_arabic': labels[str(class_id)]['sign_arabic'],
                     'status': 'EVALUATED' if evaluated else 'EXCLUDED_UNRESOLVED_LABELS',
                     'support': int(support[index]), 'support_below_10': bool(evaluated and support[index] < 10),
                     'precision': float(precision[index]) if evaluated else None,
                     'recall': float(recall[index]) if evaluated else None, 'f1': float(f1[index]) if evaluated else None})
    calibration.write_csv(output / 'test_primary_per_class.csv', rows, list(rows[0]))
    matrix = confusion_matrix(y, prediction, labels=all_indices)
    rows = [dict(true_class_id=int(i + 1), **{f'pred_{j + 1}': int(matrix[i, j]) for j in all_indices}) for i in all_indices]
    calibration.write_csv(output / 'test_primary_confusion_matrix.csv', rows, list(rows[0]))
    pairs = [(int(matrix[i, j]), int(i), int(j)) for i, j in zip(*np.nonzero(matrix)) if i != j]
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    fields = ['true_class_id', 'true_label_english', 'predicted_class_id', 'predicted_label_english', 'errors']
    rows = [{'true_class_id': i + 1, 'true_label_english': labels[str(i + 1)]['sign_english'],
             'predicted_class_id': j + 1, 'predicted_label_english': labels[str(j + 1)]['sign_english'],
             'errors': count} for count, i, j in pairs[:100]]
    calibration.write_csv(output / 'test_primary_top_confusions.csv', rows, fields)
    confidence = softmax(logits.astype(np.float64) / temperature, axis=1).max(axis=1)
    wrong = np.flatnonzero(prediction != y).tolist()
    wrong.sort(key=lambda i: (-confidence[i], paths[i]))
    fields = ['pose_file', 'true_class_id', 'predicted_class_id', 'calibrated_confidence', 'status']
    rows = [{'pose_file': paths[i], 'true_class_id': int(y[i] + 1), 'predicted_class_id': int(prediction[i] + 1),
             'calibrated_confidence': float(confidence[i]), 'status': 'uncertain'} for i in wrong[:200]]
    calibration.write_csv(output / 'test_primary_high_confidence_errors.csv', rows, fields)


def benchmark(model, device, temperature, warmup=20, repeats=100, batch_sizes=(1, 32)):
    model = model.to(device).eval()
    generator = np.random.default_rng(42)
    rows = []
    for batch_size in batch_sizes:
        array = generator.normal(0, 0.1, size=(batch_size, 32, 50, 3)).astype(np.float32)
        array[:, :, :, 2] = 1
        batch = torch.from_numpy(array.reshape(batch_size, 32, 150))

        def run():
            output = model(batch.to(device))
            scores, indices = (output / temperature).softmax(dim=1).topk(5, dim=1)
            return scores.cpu().numpy(), indices.cpu().numpy()

        durations = []
        with torch.inference_mode():
            for _ in range(warmup):
                run()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            for _ in range(repeats):
                start = time.perf_counter()
                scores, indices = run()
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                durations.append((time.perf_counter() - start) * 1000)
        require(scores.shape == indices.shape == (batch_size, 5) and np.isfinite(scores).all(),
                'Benchmark output shape or values are invalid.')
        mean = float(np.mean(durations))
        rows.append({'device': str(device), 'batch_size': batch_size, 'warmup': warmup, 'repeats': repeats,
                     'mean_batch_ms': mean, 'p50_batch_ms': float(np.quantile(durations, 0.50)),
                     'p95_batch_ms': float(np.quantile(durations, 0.95)), 'p99_batch_ms': float(np.quantile(durations, 0.99)),
                     'mean_ms_per_sample': mean / batch_size, 'samples_per_second': 1000 * batch_size / mean})
    return rows


def print_result(summary, output):
    metrics = summary['primary']
    print('\n=== TEST EVALUATION RESULT ===', flush=True)
    print('Model: SignCNN502 | Output classes: 502 | Evaluated true classes: 500', flush=True)
    print(f"Primary samples: {metrics['samples']}", flush=True)
    print(f"Accuracy: {metrics['accuracy']:.4%}", flush=True)
    print(f"Macro F1: {metrics['macro_f1']:.6f}", flush=True)
    print(f"Top-5 Accuracy: {metrics['top5_accuracy']:.4%}", flush=True)
    print(f"ECE before: {metrics['calibration_before']['ece_15_equal_width']:.6f}", flush=True)
    print(f"ECE after: {metrics['calibration_after']['ece_15_equal_width']:.6f}", flush=True)
    print('Active-policy coverage: 0.00% | Accepted accuracy: N/A', flush=True)
    print('Abstention policy: UNCHANGED; all predictions remain uncertain.', flush=True)
    print(f"Checkpoint size: {summary['benchmark']['checkpoint_mib']:.3f} MiB", flush=True)
    for row in summary['benchmark']['measurements']:
        print(f"Benchmark {row['device']} batch={row['batch_size']}: mean={row['mean_batch_ms']:.3f} ms, p95={row['p95_batch_ms']:.3f} ms", flush=True)
    print(f'Reports: {output}', flush=True)
    print('Model and calibration: UNCHANGED\nOOV detection: NOT VALIDATED\nProduction ready: NO', flush=True)
    print('Status: TEST_EVALUATION_AND_BENCHMARK_COMPLETED', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion'))
    parser.add_argument('--cache', type=Path, default=Path('/content/signova_502_test_cache'))
    args = parser.parse_args()
    common = load_module('signova_compare_cnn_502.py')
    calibration = load_module('signova_calibrate_cnn_502.py')
    common.seed_all(42)
    torch.set_num_threads(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = args.base / 'artifacts/private/features_502_v1'
    experiment = args.base / 'experiments/cnn_502_pass_v1'
    calibration_dir = experiment / 'calibration_v1'
    output = experiment / 'test_evaluation_v1'
    checkpoint_path = experiment / 'best_cnn_502_pass_v1.pt'
    calibration_path = calibration_dir / 'calibration_config.json'
    qc_path = source / 'sample_qc.csv'
    labels_path = args.base / 'source_metadata/labels_mapping.json'
    require(sha256(checkpoint_path) == CHECKPOINT_SHA256, 'Selected CNN checkpoint changed.')
    require(sha256(calibration_path) == CALIBRATION_CONFIG_SHA256, 'Frozen calibration settings changed.')
    require(sha256(qc_path) == QC_SHA256, 'Frozen sample QC changed.')
    require(sha256(labels_path) == LABELS_SHA256, 'Source label mapping changed.')
    calibration_report = calibration.load_completed_report(calibration_dir)
    require(calibration_report is not None, 'Calibration is incomplete.')
    config = json.loads(calibration_path.read_text())
    require(config['acceptance_enabled'] is False and config['threshold'] is None and
            config['candidate_threshold'] is None and config['checkpoint_sha256'] == CHECKPOINT_SHA256 and
            config['source_fingerprint'] == common.FINGERPRINT and config['class_ids'] == common.CLASSES,
            'Frozen uncertainty policy does not match this evaluation.')
    temperature = float(config['temperature'])
    manifest = json.loads((source / 'shards.json').read_text())
    calibration_protocol = json.loads((calibration_dir / 'protocol.json').read_text())
    require(manifest['fingerprint'] == common.FINGERPRINT and len(manifest['shards']) == 149 and
            sha256(source / 'shards.json') == calibration_protocol['shards_manifest_sha256'],
            'Feature shard manifest changed.')
    rows = read_qc(qc_path)
    require(len(rows) == 75515, 'Unexpected QC sample count.')
    audit, primary_paths, secondary_paths, counts = build_policy(rows)
    require(counts['test_source_samples'] == 25112 and counts['qc_counts'] == {'PASS': 25106, 'REVIEW': 6}
            and counts['excluded_class_samples'] == 150 and counts['historically_evaluated_samples'] == 450
            and counts['primary_samples'] == 24536 and counts['secondary_samples'] == 24956
            and counts['primary_classes'] == 500 and counts['primary_minimum_class_support'] == 2,
            'Test subset counts differ from the pre-evaluation audit.')
    require({row['signer_id'] for row in rows if row['split'] == 'test'} == {1}, 'Test signer changed.')
    verify_leakage(rows, secondary_paths)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        'protocol_version': 'cnn502_test_exclusions_v1', 'model_version': 'karsl_cnn_502_pass_v1',
        'checkpoint_sha256': CHECKPOINT_SHA256, 'calibration_config_sha256': CALIBRATION_CONFIG_SHA256,
        'sample_qc_sha256': QC_SHA256, 'labels_sha256': LABELS_SHA256,
        'shards_manifest_sha256': sha256(source / 'shards.json'), 'source_fingerprint': common.FINGERPRINT,
        'script_sha256': sha256(Path(__file__)), 'module_sha256': MODULE_HASHES,
        'primary': 'Test PASS samples, excluding all class IDs 27 and 28 and every historically evaluated test path.',
        'secondary': 'Test PASS samples excluding all class IDs 27 and 28; includes historically evaluated samples and is descriptive only.',
        'exclusion_rationale': 'Six technical REVIEW samples are excluded. All 150 samples in classes 27/28 are excluded conservatively because 50 raw duplicate pairs carry conflicting labels; remaining class-27 examples lack an independent semantic resolution. No source files or labels are modified.',
        'exclusions_fixed_before_predictions': True, 'counts': counts,
        'model_output_classes': 502, 'evaluated_class_ids': EVALUATED_CLASS_IDS,
        'macro_f1_labels': 'The fixed 500 evaluated true classes. Predictions into excluded classes remain wrong and reduce recall/F1.',
        'ranking': 'Unmodified 502-way logits. No class masking, renormalization, or output-head changes.',
        'temperature': temperature, 'acceptance_enabled': False, 'threshold': None,
        'calibration_policy': 'Use the frozen validation temperature and disabled acceptance. Never select a threshold or retrain from test results.',
        'scope': 'Terminal research evaluation of this frozen checkpoint; not a clean full-502-class benchmark or production approval.',
        'limitations': [
            'The underlying test pool has historical exposure. Removing 450 previously evaluated paths does not erase all prior study-design influence.',
            'Only one held-out signer is represented; samples within the signer/session are correlated.',
            'Some primary classes have only two examples after exclusions; per-class estimates are unstable.',
            'Validation was reused for model selection and calibration analysis.',
            'Source label semantics, pose/video provenance, and release rights remain unapproved.',
            'Unknown/OOV detection has not been validated.',
        ],
        'benchmark': {'warmup': 20, 'repeats': 100, 'batch_sizes': [1, 32], 'torch_cpu_threads': 2,
                      'inputs': 'Fixed synthetic normalized features; all visibility channels equal one.',
                      'timed_scope': 'Preconstructed CPU tensor -> device transfer -> model -> temperature softmax -> top-5 -> CPU NumPy arrays.',
                      'excluded': 'Model loading, input validation, JSON construction, pose/video preprocessing, network, backend, and UI.'},
    }
    protocol_path = output / 'test_protocol.json'
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol, 'A different test protocol already exists.')
    else:
        common.save_json(protocol_path, protocol)
    completion_path = output / 'completion.json'
    if completion_path.exists():
        completion = json.loads(completion_path.read_text())
        for name, digest in completion['files'].items():
            require(Path(name).name == name and name not in ('.', '..'), 'Unsafe completed report path.')
            require(sha256(output / name) == digest, f'Completed test artifact changed: {name}')
        print('Completed test evaluation verified; reusing frozen results without new predictions.', flush=True)
        print_result(json.loads((output / 'test_report.json').read_text()), output)
        return
    calibration.write_csv(output / 'test_manifest_audit.csv', audit, list(audit[0]))
    common.save_json(output / 'subset_counts.json', counts)
    policy_text = (
        '# Frozen test evaluation policy\n\n'
        'This protocol was written before generating predictions for the 502-class CNN on test.\n\n'
        'Primary: 24,536 technically PASS samples across 500 true classes. Exclude six REVIEW samples, all 150 samples in classes 27/28, and 420 additional historically evaluated samples. '
        'The historical set has 450 samples in total; 30 already fall within the excluded classes.\n\n'
        'Secondary: 24,956 PASS samples across the same 500 classes, including the 420 historically evaluated samples. This comparison is descriptive only.\n\n'
        'The model retains all 502 outputs. Predictions into classes 27/28 on other true classes are errors. Macro F1 averages the fixed 500 evaluated true classes. '
        'No labels are automatically corrected. Exclusions are based on metadata/QC, not prediction correctness.\n\n'
        'Acceptance remains disabled. Forced-choice classification accuracy is a research metric; accepted-prediction coverage is zero and accepted accuracy is undefined.\n\n'
        'The protocol does not resolve semantic review, historical test exposure, data rights, OOV evaluation, or limited signer diversity. '
        'Future model improvement must not tune against these test results.\n'
    )
    (output / 'test_policy.md').write_text(policy_text, encoding='utf-8')
    print('Test protocol: FROZEN before predictions', flush=True)
    print('Primary: 24536 samples | True classes: 500 | Model outputs: 502', flush=True)
    print('Secondary: 24956 samples | Contains historical test exposure', flush=True)
    print('Acceptance: DISABLED | Calibration and model: FROZEN', flush=True)
    x, y, paths = load_test_features(source, args.cache, manifest, rows, secondary_paths, common)
    primary_mask = np.array([path in primary_paths for path in paths], dtype=bool)
    require(int(primary_mask.sum()) == 24536 and len(y) == 24956, 'Loaded subset counts changed.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(checkpoint['model_name'] == 'SignCNN502' and checkpoint['classes'] == common.CLASSES and
            checkpoint['input_shape'] == [32, 150] and checkpoint['source_fingerprint'] == common.FINGERPRINT and
            checkpoint['preprocessing_version'] == common.PREPROCESSING and checkpoint['best_epoch'] == 12,
            'Selected checkpoint metadata mismatch.')
    model = common.SignCNN502(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    loader = common.make_loader(x, y, 256, False, device)
    logits, actual_y = calibration.collect_logits(model, loader, device)
    require(np.array_equal(y, actual_y), 'Test inference order changed.')
    logits_path = args.base / 'artifacts/private/cnn_502_test_logits_v1.npz'
    temporary = logits_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, logits=logits, labels=y, paths=np.asarray(paths), primary_mask=primary_mask,
                            checkpoint_sha256=CHECKPOINT_SHA256, calibration_sha256=CALIBRATION_CONFIG_SHA256,
                            protocol_sha256=sha256(protocol_path), fingerprint=common.FINGERPRINT)
    os.replace(temporary, logits_path)
    primary_logits, primary_y = logits[primary_mask], y[primary_mask]
    primary_names = [path for path, keep in zip(paths, primary_mask) if keep]
    primary = compute_metrics(primary_logits, primary_y, temperature, calibration)
    secondary = compute_metrics(logits, y, temperature, calibration)
    export_class_reports(output, primary_logits, primary_y, primary_names, temperature,
                         json.loads(labels_path.read_text()), calibration)
    before_metrics, before_bins = calibration.calibration_metrics(primary_logits, primary_y, 1.0)
    after_metrics, after_bins = calibration.calibration_metrics(primary_logits, primary_y, temperature)
    reliability = [{'stage': stage, **row} for stage, bins in [('before', before_bins), ('after', after_bins)] for row in bins]
    calibration.write_csv(output / 'test_primary_reliability_bins.csv', reliability, list(reliability[0]))
    del x, loader
    print('Test predictions complete. Running fixed inference benchmarks.', flush=True)
    measurements = []
    if device.type == 'cuda':
        measurements.extend(benchmark(model, device, temperature))
    measurements.extend(benchmark(model, torch.device('cpu'), temperature))
    benchmark_report = {'checkpoint_bytes': checkpoint_path.stat().st_size,
                        'checkpoint_mib': checkpoint_path.stat().st_size / 1024 ** 2,
                        'parameters': sum(p.numel() for p in model.parameters()),
                        'measurements': measurements, 'method': protocol['benchmark']}
    common.save_json(output / 'benchmark.json', benchmark_report)
    calibration.write_csv(output / 'benchmark.csv', measurements, list(measurements[0]))
    require(sha256(checkpoint_path) == CHECKPOINT_SHA256 and sha256(calibration_path) == CALIBRATION_CONFIG_SHA256,
            'Frozen model or calibration changed during evaluation.')
    summary = {'status': 'TEST_EVALUATION_AND_BENCHMARK_COMPLETED', 'model': 'SignCNN502',
               'best_epoch': 12, 'primary': primary, 'secondary_with_historical_exposure': secondary,
               'counts': counts, 'benchmark': benchmark_report,
               'checkpoint_sha256': CHECKPOINT_SHA256, 'calibration_config_sha256': CALIBRATION_CONFIG_SHA256,
               'protocol_sha256': sha256(protocol_path), 'test_logits_sha256': sha256(logits_path),
               'test_evaluated': True, 'model_unchanged': True, 'calibration_unchanged': True,
               'production_ready': False, 'oov_detection_validated': False,
               'limitations': protocol['limitations'], 'completed_utc': datetime.now(timezone.utc).isoformat(),
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'numpy': np.__version__, 'platform': platform.platform(),
                           'device': str(device), 'cpu_threads': torch.get_num_threads(),
                           'cuda_build': str(torch.version.cuda), 'cudnn': torch.backends.cudnn.version(),
                           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    common.save_json(output / 'test_report.json', summary)
    metrics_rows = []
    for name, metrics in [('primary_previously_unevaluated', primary), ('secondary_includes_historical', secondary)]:
        metrics_rows.append({'subset': name, 'samples': metrics['samples'], 'evaluated_true_classes': 500,
                             'model_output_classes': 502, 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
                             'top5_accuracy': metrics['top5_accuracy'],
                             'ece_before': metrics['calibration_before']['ece_15_equal_width'],
                             'ece_after': metrics['calibration_after']['ece_15_equal_width'],
                             'accepted_coverage': 0.0, 'accepted_accuracy': None})
    calibration.write_csv(output / 'test_metrics.csv', metrics_rows, list(metrics_rows[0]))
    report_text = (
        '# Frozen CNN research test evaluation\n\n'
        'The model has 502 outputs; the primary evaluation covers 500 true classes after documented exclusions. '
        'These figures must not be described as a clean full-502-class test result.\n\n'
        f"Primary samples: {primary['samples']}. Accuracy: {primary['accuracy']:.4%}; macro F1: {primary['macro_f1']:.6f}; top-5: {primary['top5_accuracy']:.4%}.\n\n"
        f"ECE: {before_metrics['ece_15_equal_width']:.6f} before and {after_metrics['ece_15_equal_width']:.6f} after applying the frozen validation temperature.\n\n"
        'The active prediction policy abstains on every input. Accepted coverage is 0%; accepted accuracy is undefined, not zero or 100%. '
        'Forced-choice metrics describe the underlying classifier and do not indicate reliable translation.\n\n'
        f"Secondary samples including historical exposure: {secondary['samples']}; accuracy: {secondary['accuracy']:.4%}; macro F1: {secondary['macro_f1']:.6f}; top-5: {secondary['top5_accuracy']:.4%}. This is descriptive only.\n\n"
        f"Primary minimum class support: {counts['primary_minimum_class_support']}; classes below ten samples: {counts['primary_classes_with_support_below_10']}. See per-class results before interpreting aggregate metrics.\n\n"
        f"Checkpoint size: {benchmark_report['checkpoint_mib']:.3f} MiB. Benchmark timing includes model execution, temperature softmax, top-5, and device transfers; excludes pose/video preprocessing and application overhead.\n\n"
        '## Limitations\n\n' + '\n'.join('- ' + item for item in protocol['limitations']) + '\n\n'
        'Model parameters, class order, and calibration remain unchanged. No test-driven retraining or threshold adjustment occurred.\n'
    )
    (output / 'test_report.md').write_text(report_text, encoding='utf-8')
    names = ['test_protocol.json', 'test_manifest_audit.csv', 'subset_counts.json', 'test_policy.md',
             'test_primary_per_class.csv', 'test_primary_confusion_matrix.csv', 'test_primary_top_confusions.csv',
             'test_primary_high_confidence_errors.csv', 'test_primary_reliability_bins.csv', 'benchmark.json',
             'benchmark.csv', 'test_report.json', 'test_metrics.csv', 'test_report.md']
    common.save_json(completion_path, {'files': {name: sha256(output / name) for name in names}})
    print_result(summary, output)


if __name__ == '__main__':
    main()

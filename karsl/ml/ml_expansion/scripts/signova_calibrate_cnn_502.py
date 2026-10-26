import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
import torch

COMMON_SHA256 = 'f6e0f60a817de984fc356b7bbd7edcaff72ffbc63e046fe09175564cc6633b5e'
CHECKPOINT_SHA256 = 'f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254'
LABELS_SHA256 = '3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a'
VERSION = 'cnn502_validation_temperature_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_common():
    path = Path(__file__).with_name('signova_compare_cnn_502.py')
    require(sha256(path) == COMMON_SHA256, 'Comparison module checksum mismatch.')
    spec = importlib.util.spec_from_file_location('signova_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, fields):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(output.getvalue(), encoding='utf-8')
    os.replace(temporary, path)


def partition_validation(paths, y):
    require(len(paths) == len(y) == len(set(paths)), 'Invalid validation sample identities.')
    names = np.empty(len(y), dtype='<U16')
    for label in np.unique(y):
        members = np.flatnonzero(y == label).tolist()
        require(len(members) >= 4, 'Too few samples for the calibration partition.')
        members.sort(key=lambda index: (hashlib.sha256(
            ('SIGNOVA_CALIBRATION_V1|42|' + paths[index]).encode()).hexdigest(), paths[index]))
        n_fit, n_select = len(members) // 2, len(members) // 4
        names[members[:n_fit]] = 'temperature_fit'
        names[members[n_fit:n_fit + n_select]] = 'threshold_select'
        names[members[n_fit + n_select:]] = 'policy_audit'
    return names


def negative_log_likelihood(logits, y, temperature):
    require(np.isfinite(temperature) and temperature > 0, 'Invalid temperature.')
    scores = np.asarray(logits, dtype=np.float64) / temperature
    return float(np.mean(logsumexp(scores, axis=1) - scores[np.arange(len(y)), y]))


def fit_temperature(logits, y):
    require(len(logits) == len(y) > 0 and np.isfinite(logits).all(), 'Invalid calibration inputs.')
    before = negative_log_likelihood(logits, y, 1.0)
    result = minimize_scalar(lambda beta: negative_log_likelihood(logits, y, 1.0 / beta),
                             bounds=(0.02, 20.0), method='bounded',
                             options={'xatol': 1e-7, 'maxiter': 150})
    require(result.success and np.isfinite(result.fun), 'Temperature optimization failed.')
    candidates = [(1.0, before)]
    candidates.extend((1.0 / beta, negative_log_likelihood(logits, y, 1.0 / beta))
                      for beta in (0.02, float(result.x), 20.0))
    temperature, after = min(candidates, key=lambda item: item[1])
    return temperature, {'nll_before': before, 'nll_after': after,
                         'temperature_bounds': [0.05, 50.0],
                         'boundary_solution': temperature in (0.05, 50.0),
                         'objective': 'Mean negative log likelihood on temperature_fit only.'}


def reliability_table(confidence, correct, bins=15):
    require(len(confidence) == len(correct) > 0 and np.isfinite(confidence).all(),
            'Invalid reliability inputs.')
    require(bool(np.all((confidence >= 0) & (confidence <= 1))), 'Confidence outside [0, 1].')
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    rows, ece = [], 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else None
        mean_confidence = float(confidence[mask].mean()) if count else None
        gap = abs(accuracy - mean_confidence) if count else 0.0
        ece += count / len(confidence) * gap
        rows.append({'bin': index, 'lower': index / bins, 'upper': (index + 1) / bins,
                     'upper_inclusive': index == bins - 1, 'count': count,
                     'accuracy': accuracy, 'mean_confidence': mean_confidence, 'absolute_gap': gap})
    return float(ece), rows


def classification_metrics(logits, y):
    labels = np.arange(logits.shape[1])
    prediction = logits.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, prediction)),
            'macro_f1': float(f1_score(y, prediction, average='macro', labels=labels, zero_division=0)),
            'top5_accuracy': float(top_k_accuracy_score(y, logits, k=5, labels=labels))}


def calibration_metrics(logits, y, temperature):
    probs = softmax(np.asarray(logits, dtype=np.float64) / temperature, axis=1)
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y
    ece, bins = reliability_table(confidence, correct)
    brier = np.mean(np.square(probs).sum(axis=1) - 2 * probs[np.arange(len(y)), y] + 1)
    return {'samples': len(y), 'accuracy': float(correct.mean()),
            'nll': negative_log_likelihood(logits, y, temperature),
            'ece_15_equal_width': ece, 'multiclass_brier_sum': float(brier)}, bins


def wilson_lower(correct, count):
    if count == 0:
        return None
    z = 1.959963984540054
    proportion = correct / count
    denominator = 1 + z * z / count
    return float((proportion + z * z / (2 * count) -
                  z * math.sqrt(proportion * (1 - proportion) / count + z * z / (4 * count ** 2))) /
                 denominator)


def selective_metrics(confidence, correct, threshold):
    accepted = np.zeros(len(confidence), dtype=bool) if threshold is None else confidence >= threshold
    count = int(accepted.sum())
    successes = int(correct[accepted].sum())
    return {'samples': len(confidence), 'threshold': threshold, 'accepted': count,
            'coverage': count / len(confidence), 'accepted_correct': successes,
            'accepted_accuracy': successes / count if count else None,
            'accepted_error_rate': 1 - successes / count if count else None,
            'wilson_95_lower': wilson_lower(successes, count)}


def meets_target(stats, target=0.80, minimum=100):
    return stats['accepted'] >= minimum and stats['wilson_95_lower'] is not None and stats['wilson_95_lower'] >= target


def choose_threshold(confidence, correct, target=0.80, minimum=100):
    rows = [selective_metrics(confidence, correct, float(value)) for value in np.linspace(0, 1, 101)]
    eligible = [row for row in rows if meets_target(row, target, minimum)]
    selected = max(eligible, key=lambda row: (row['coverage'], row['accepted_accuracy'], -row['threshold'])) if eligible else None
    return selected, rows


def collect_logits(model, loader, device):
    model.eval()
    parts, truth = [], []
    with torch.inference_mode():
        for batch_x, batch_y in loader:
            logits = model(batch_x.to(device, non_blocking=True))
            require(bool(torch.isfinite(logits).all()), 'Non-finite CNN output.')
            parts.append(logits.cpu().numpy())
            truth.append(batch_y.numpy())
    return np.concatenate(parts), np.concatenate(truth)


def load_completed_report(output):
    marker = output / 'completion.json'
    if not marker.exists():
        return None
    completed = json.loads(marker.read_text())
    for name, digest in completed['files'].items():
        require(Path(name).name == name and name not in ('.', '..'), 'Unsafe report path.')
        require(sha256(output / name) == digest, f'Completed calibration artifact changed: {name}')
    return json.loads((output / 'calibration_report.json').read_text())


def print_summary(summary, output):
    before = summary['partitions']['policy_audit']['before']
    after = summary['partitions']['policy_audit']['after']
    print('\n=== CALIBRATION RESULT ===', flush=True)
    print(f"Temperature: {summary['temperature']:.6f}", flush=True)
    print(f"Audit ECE before: {before['ece_15_equal_width']:.6f}", flush=True)
    print(f"Audit ECE after: {after['ece_15_equal_width']:.6f}", flush=True)
    print(f"Candidate threshold: {summary['candidate_threshold']}", flush=True)
    print(f"Acceptance enabled: {summary['acceptance_enabled']} | Policy status: {summary['policy_status']}", flush=True)
    print(f"Candidate policy audit: {json.dumps(summary['candidate_policy_audit'])}", flush=True)
    print(f'Class reports and calibration files: {output}', flush=True)
    print('Checkpoint: UNCHANGED\nTest evaluated: NO\nOOV detection: NOT VALIDATED', flush=True)
    print('Status: CALIBRATION_AND_ERROR_ANALYSIS_COMPLETED', flush=True)


def export_errors(output, logits, y, paths, partitions, confidence, label_map, enabled, threshold):
    prediction = logits.argmax(axis=1)
    labels = np.arange(logits.shape[1])
    precision, recall, f1, support = precision_recall_fscore_support(y, prediction, labels=labels, zero_division=0)
    rows = [{'class_id': int(i + 1), 'label_english': label_map[str(i + 1)]['sign_english'],
             'label_arabic': label_map[str(i + 1)]['sign_arabic'], 'support': int(support[i]),
             'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
            for i in labels]
    write_csv(output / 'validation_per_class.csv', rows, list(rows[0]))
    cm = confusion_matrix(y, prediction, labels=labels)
    cm_rows = [dict(true_class_id=int(i + 1), **{f'pred_{j + 1}': int(cm[i, j]) for j in labels}) for i in labels]
    write_csv(output / 'validation_confusion_matrix.csv', cm_rows, list(cm_rows[0]))
    pairs = [(int(cm[i, j]), int(i), int(j)) for i, j in zip(*np.nonzero(cm)) if i != j]
    pairs.sort(key=lambda row: (-row[0], row[1], row[2]))
    fields = ['true_class_id', 'true_label_english', 'predicted_class_id', 'predicted_label_english', 'errors', 'true_class_support', 'class_error_fraction']
    confusion_rows = [{'true_class_id': i + 1, 'true_label_english': label_map[str(i + 1)]['sign_english'],
                       'predicted_class_id': j + 1, 'predicted_label_english': label_map[str(j + 1)]['sign_english'],
                       'errors': count, 'true_class_support': int(support[i]),
                       'class_error_fraction': count / int(support[i])} for count, i, j in pairs[:100]]
    write_csv(output / 'validation_top_confusions.csv', confusion_rows, fields)
    wrong = np.flatnonzero(prediction != y).tolist()
    wrong.sort(key=lambda i: (-confidence[i], paths[i]))
    fields = ['pose_file', 'partition', 'true_class_id', 'true_label_english', 'predicted_class_id', 'predicted_label_english', 'calibrated_confidence', 'accepted_by_active_policy']
    errors = [{'pose_file': paths[i], 'partition': str(partitions[i]), 'true_class_id': int(y[i] + 1),
               'true_label_english': label_map[str(y[i] + 1)]['sign_english'],
               'predicted_class_id': int(prediction[i] + 1),
               'predicted_label_english': label_map[str(prediction[i] + 1)]['sign_english'],
               'calibrated_confidence': float(confidence[i]),
               'accepted_by_active_policy': bool(enabled and confidence[i] >= threshold)} for i in wrong[:200]]
    write_csv(output / 'validation_high_confidence_errors.csv', errors, fields)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, default=Path('/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion'))
    parser.add_argument('--cache', type=Path, default=Path('/content/signova_502_train_cache'))
    args = parser.parse_args()
    common = load_common()
    torch.set_num_threads(2)
    common.seed_all(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = args.base / 'artifacts/private/features_502_v1'
    experiment = args.base / 'experiments/cnn_502_pass_v1'
    checkpoint_path = experiment / 'best_cnn_502_pass_v1.pt'
    labels_path = args.base / 'source_metadata/labels_mapping.json'
    require(sha256(checkpoint_path) == CHECKPOINT_SHA256, 'Selected CNN checkpoint checksum mismatch.')
    require(sha256(labels_path) == LABELS_SHA256, 'Source label mapping changed.')
    result = json.loads((experiment / 'result.json').read_text())
    require(result['status'] == 'COMPARISON_COMPLETED' and result['preferred_by_validation_macro_f1'] == 'CNN'
            and result['checkpoint_sha256'] == CHECKPOINT_SHA256 and not result['test_evaluated'],
            'Model comparison or selection does not match this calibration run.')
    manifest = json.loads((source / 'shards.json').read_text())
    require(manifest['fingerprint'] == common.FINGERPRINT, 'Source fingerprint mismatch.')
    labels = json.loads(labels_path.read_text())
    require(set(labels) == {str(i) for i in common.CLASSES}, 'Missing or unexpected source labels.')
    for key, value in labels.items():
        require(value['sign_index'] == int(key) - 1 and isinstance(value['sign_english'], str)
                and isinstance(value['sign_arabic'], str), 'Inconsistent source label mapping.')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(checkpoint['model_name'] == 'SignCNN502' and checkpoint['best_epoch'] == 12 and
            checkpoint['classes'] == common.CLASSES and checkpoint['source_fingerprint'] == common.FINGERPRINT
            and checkpoint['preprocessing_version'] == common.PREPROCESSING and checkpoint['input_shape'] == [32, 150]
            and not checkpoint['test_evaluated'],
            'Selected checkpoint metadata mismatch.')
    output = experiment / 'calibration_v1'
    output.mkdir(parents=True, exist_ok=True)
    protocol = {'version': VERSION, 'model_version': 'karsl_cnn_502_pass_v1',
                'checkpoint_sha256': CHECKPOINT_SHA256, 'source_fingerprint': common.FINGERPRINT,
                'shards_manifest_sha256': sha256(source / 'shards.json'),
                'script_sha256': sha256(Path(__file__)), 'comparison_module_sha256': COMMON_SHA256,
                'labels_sha256': LABELS_SHA256, 'partition_method': 'Per-class SHA256 path order, seed 42; floor(n/2) fit, floor(n/4) selection, remainder audit.',
                'temperature_fit_objective': 'Mean NLL on temperature_fit only; bounded temperature 0.05 to 50.',
                'threshold_grid': {'start': 0.0, 'stop': 1.0, 'step': 0.01},
                'minimum_accepted_samples': 100, 'target_wilson_95_lower_accuracy': 0.80,
                'threshold_selection': 'Maximum coverage satisfying the target on threshold_select.',
                'audit_rule': 'Apply the selected threshold once to policy_audit without retuning; disable acceptance if the same support/lower-bound rule fails.',
                'unmet_target_behavior': 'Return uncertain for every prediction; retain top-k candidates for research.',
                'target_status': 'Exploratory engineering criterion; not a team-approved service guarantee.',
                'validation_reuse': 'All partitions previously contributed to model/epoch selection. This is post-selection validation analysis, not an independent test.',
                'statistical_limit': 'Wilson bounds are descriptive, not simultaneous selection-adjusted guarantees; repeated samples from one signer/session may be correlated.',
                'oov_detection_validated': False, 'test_evaluated': False}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol, 'An incompatible calibration protocol already exists.')
    else:
        common.save_json(protocol_path, protocol)
    completed = load_completed_report(output)
    if completed is not None:
        print('Completed calibration verified; reusing frozen results without refitting.', flush=True)
        print_summary(completed, output)
        return
    print(f'Selected model: SignCNN502 | Epoch: 12 | Device: {device}', flush=True)
    print('Protocol frozen: validation only; no test feature loading.', flush=True)
    x, y, paths, _, _, info = common.load_pass_split(source, args.cache, manifest, 'validation')
    require(info == checkpoint['experiment_config']['validation'], 'Validation sample identity or QC changed.')
    partition = partition_validation(paths, y)
    partition_rows = [{'pose_file': path, 'class_id': int(label + 1), 'partition': str(part)}
                      for path, label, part in zip(paths, y, partition)]
    write_csv(output / 'validation_partitions.csv', partition_rows, list(partition_rows[0]))
    for name in ('temperature_fit', 'threshold_select', 'policy_audit'):
        print(f'{name}: {int((partition == name).sum())} samples', flush=True)
    model = common.SignCNN502(**checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    loader = common.make_loader(x, y, 256, False, device)
    logits, actual_y = collect_logits(model, loader, device)
    require(np.array_equal(y, actual_y), 'Prediction/sample alignment changed.')
    metrics = classification_metrics(logits, y)
    common.check_metrics(metrics, checkpoint['validation_metrics'])
    print('CNN checkpoint reload and validation metrics: PASS', flush=True)
    logits_path = args.base / 'artifacts/private/cnn_502_validation_logits_v1.npz'
    temporary = logits_path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, logits=logits, labels=y, paths=np.asarray(paths), partitions=partition,
                            checkpoint_sha256=CHECKPOINT_SHA256, fingerprint=common.FINGERPRINT,
                            preprocessing_version=common.PREPROCESSING)
    os.replace(temporary, logits_path)
    fit_mask = partition == 'temperature_fit'
    temperature, fit_result = fit_temperature(logits[fit_mask], y[fit_mask])
    require(fit_result['nll_after'] <= fit_result['nll_before'] + 1e-10, 'Temperature fitting increased NLL.')
    calibrated = softmax(logits.astype(np.float64) / temperature, axis=1)
    require(np.array_equal(calibrated.argmax(axis=1), logits.argmax(axis=1)), 'Calibration changed class ranking.')
    confidence = calibrated.max(axis=1)
    correct = logits.argmax(axis=1) == y
    select_mask, audit_mask = partition == 'threshold_select', partition == 'policy_audit'
    selected, sweep = choose_threshold(confidence[select_mask], correct[select_mask])
    write_csv(output / 'threshold_search.csv', sweep, list(sweep[0]))
    candidate_threshold = selected['threshold'] if selected else None
    audit = selective_metrics(confidence[audit_mask], correct[audit_mask], candidate_threshold)
    enabled = selected is not None and meets_target(audit)
    policy_status = 'AUDIT_TARGET_MET' if enabled else ('NO_THRESHOLD_MET_SELECTION_TARGET' if selected is None else 'AUDIT_TARGET_NOT_MET')
    active_threshold = candidate_threshold if enabled else None
    groups, reliability_rows, metric_rows = {}, [], []
    for name in ('temperature_fit', 'threshold_select', 'policy_audit', 'all_validation'):
        mask = np.ones(len(y), dtype=bool) if name == 'all_validation' else partition == name
        before, before_bins = calibration_metrics(logits[mask], y[mask], 1.0)
        after, after_bins = calibration_metrics(logits[mask], y[mask], temperature)
        require(before['accuracy'] == after['accuracy'], 'Temperature changed prediction accuracy.')
        groups[name] = {'before': before, 'after': after,
                        'active_policy': selective_metrics(confidence[mask], correct[mask], active_threshold)}
        for stage, values, bins in [('before', before, before_bins), ('after', after, after_bins)]:
            metric_rows.append({'partition': name, 'stage': stage, **values})
            reliability_rows.extend({'partition': name, 'stage': stage, **row} for row in bins)
    write_csv(output / 'calibration_metrics.csv', metric_rows, list(metric_rows[0]))
    write_csv(output / 'reliability_bins.csv', reliability_rows, list(reliability_rows[0]))
    config = {'calibration_version': VERSION, 'model_version': 'karsl_cnn_502_pass_v1',
              'model_name': 'SignCNN502', 'checkpoint_sha256': CHECKPOINT_SHA256,
              'preprocessing_version': common.PREPROCESSING, 'source_fingerprint': common.FINGERPRINT,
              'temperature': temperature, 'acceptance_enabled': enabled, 'threshold': active_threshold,
              'candidate_threshold': candidate_threshold, 'policy_status': policy_status,
              'comparison': 'accept only when acceptance_enabled and max(softmax(logits / temperature)) >= threshold',
              'uncertain_output': {'status': 'uncertain', 'uncertain': True, 'class_id': None, 'label': None},
              'class_ids': common.CLASSES, 'index_mapping': 'class_id = output_index + 1',
              'oov_detection_validated': False, 'production_ready': False,
              'test_evaluated': False, 'label_semantics_verified': False,
              'validation_reuse': protocol['validation_reuse'], 'engineering_target': protocol['target_status']}
    common.save_json(output / 'calibration_config.json', config)
    export_errors(output, logits, y, paths, partition, confidence, labels, enabled, active_threshold)
    common.save_json(output / 'label_map_502.json', {'source_sha256': LABELS_SHA256,
                    'source': 'Kaggle labels_mapping.json, pinned dataset version 1',
                    'semantics_expert_verified': False, 'classes': [dict(class_id=i, **labels[str(i)]) for i in common.CLASSES]})
    require(sha256(checkpoint_path) == CHECKPOINT_SHA256, 'CNN checkpoint changed during calibration.')
    summary = {'status': 'CALIBRATION_AND_ERROR_ANALYSIS_COMPLETED', 'model': 'SignCNN502',
               'best_epoch': 12, 'validation_metrics': metrics, 'temperature': temperature,
               'temperature_fit': fit_result, 'policy_status': policy_status,
               'acceptance_enabled': enabled, 'candidate_threshold': candidate_threshold,
               'active_threshold': active_threshold, 'threshold_selection': selected,
               'candidate_policy_audit': audit, 'partitions': groups,
               'validation_sample_order_sha256': info['sample_order_sha256'],
               'validation_logits_sha256': sha256(logits_path), 'protocol_sha256': sha256(protocol_path),
               'checkpoint_unchanged': True, 'test_evaluated': False, 'oov_detection_validated': False,
               'validation_reuse': protocol['validation_reuse'],
               'known_test_issue': '50 exact duplicate pairs have conflicting labels 27/28; 450 test samples were used in historical experiments.',
               'completed_utc': datetime.now(timezone.utc).isoformat(),
               'runtime': {'python': platform.python_version(), 'torch': str(torch.__version__),
                           'numpy': np.__version__, 'scipy': scipy.__version__, 'device': str(device)}}
    common.save_json(output / 'calibration_report.json', summary)
    audit_before, audit_after = groups['policy_audit']['before'], groups['policy_audit']['after']
    report_text = (
        '# CNN 502-class validation calibration and error analysis\n\n'
        f'Selected checkpoint: epoch 12; SHA256 `{CHECKPOINT_SHA256}`.\n\n'
        f"Validation accuracy: {metrics['accuracy']:.4%}; macro F1: {metrics['macro_f1']:.6f}; top-5: {metrics['top5_accuracy']:.4%}.\n\n"
        f'Temperature: {temperature:.6f}. Temperature scaling preserves class ranking.\n\n'
        f"Policy-audit ECE: {audit_before['ece_15_equal_width']:.6f} before and {audit_after['ece_15_equal_width']:.6f} after calibration.\n\n"
        f'Acceptance policy status: `{policy_status}`. Active threshold: `{active_threshold}`.\n\n'
        'The fixed exploratory target requires at least 100 accepted samples and a descriptive 95% Wilson lower bound of at least 80% accuracy on both the selection and audit subsets. '
        'If either check fails, acceptance is disabled and all predictions remain uncertain. This criterion is not a service guarantee or team approval.\n\n'
        + protocol['validation_reuse'] + '\n\n' + protocol['statistical_limit'] + '\n\n'
        'No test predictions were generated. No unknown/OOV detection claim is supported. Test label conflicts and historical test exposure remain unresolved. '
        'Source labels are copied without automatic correction and have not received expert semantic review.\n\n'
        'Outputs: calibration_config.json; calibration_report.json; calibration_metrics.csv; reliability_bins.csv; threshold_search.csv; validation_partitions.csv; '
        'validation_per_class.csv; validation_confusion_matrix.csv; validation_top_confusions.csv; validation_high_confidence_errors.csv; label_map_502.json.\n'
    )
    (output / 'validation_report.md').write_text(report_text, encoding='utf-8')
    names = ['protocol.json', 'validation_partitions.csv', 'threshold_search.csv', 'calibration_metrics.csv',
             'reliability_bins.csv', 'calibration_config.json', 'validation_per_class.csv',
             'validation_confusion_matrix.csv', 'validation_top_confusions.csv',
             'validation_high_confidence_errors.csv', 'label_map_502.json', 'calibration_report.json',
             'validation_report.md']
    common.save_json(output / 'completion.json', {'files': {name: sha256(output / name) for name in names}})
    print_summary(summary, output)


if __name__ == '__main__':
    main()

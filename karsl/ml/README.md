# KArSL-502 ML research branch candidate

This directory contains the isolated-sign KArSL-502 research baseline. It is a separate Arabic Sign Language reference, not an Egyptian Sign Language recognizer or continuous translator. The current frozen CNN returns `uncertain` for every prediction. Keep advisory `top_k` candidates separate from confirmed translation text.

## Layout

`src/preprocessing.py` is the exact source used to create the 32 × 150 features. `ml_expansion/scripts/` contains the frozen feature preparation, CNN training/comparison, calibration, evaluation and prediction scripts. `training/train_gru_502.py` provides a standalone GRU training entry point using the same validated train/validation shard loader and architecture as the original notebook. `notebooks/` and `reports/` preserve experiment history and limitations. `reports/detailed_metrics/` contains the source test protocol, 502-row per-class validation/test results and class-ID confusion matrices. The code-only variant removes source label text from those CSVs; the private variant retains it. `tests/` includes smoke and input/metric checks. The private variant also contains two checkpoints, calibration and source labels. Neither variant contains raw Pose files or full feature shards.

The new GRU script has now been run on all frozen PASS train/validation shards in Colab. Its 23 history rows and all 20 model-state tensors exactly match the original saved GRU; the new checkpoint has a different file SHA-256 because it includes an extra experiment configuration field. A clean replay of the CNN training/comparison script yielded a byte-identical selected CNN checkpoint and matching 18-row history. Replaying the original validation calibration and terminal test protocols produced a byte-identical calibration configuration, test protocol, primary per-class CSV and confusion matrix, with identical primary and secondary metrics. See the three replay verification reports in `reports/`. Neither training script loads the test split. GRU output directories must be new to avoid overwriting the historical model. A replay of an already observed test is a reproducibility check, not a new independent test.

## Installation and smoke tests

Use a separate Python environment for ML. The recorded Colab inference used Python 3.13; this package's loader and tests were also checked with Python 3.12, PyTorch 2.11.0, NumPy 2.2.6, and `pose-format==0.15.0`. The dependency file contains compatible ranges rather than a full lock. The experimental CV pilot uses a different NumPy 1.26.4/MediaPipe 0.10.21 environment and is not part of this package.

```bash
python -m pip install -r karsl/ml/requirements_502.txt
python -m unittest discover -s karsl/ml/tests -v
```

Run those commands from the repository root. The code-only package skips checkpoint-dependent assertions. A real source-compatible `.pose` path is optional for the inherited test; a skipped test must not be counted as a pass. Tests do not certify accuracy, sign semantics, OOV detection, or full retraining.

## Controlled data and training

Obtain the pinned Pose source and frozen metadata, signer folds, download inventory, earlier 30-class reference features and full 502-class shards through the team's controlled storage. The ZIP files do not contain these large/private inputs. Do not substitute another feature representation or split. The frozen scripts have recorded SHA-256 identities; edits to them require an explicitly new experiment version.

The feature-preparation script takes explicit paths:

```bash
python karsl/ml/ml_expansion/scripts/signova_prepare_features.py \
  --repo-root "$PWD" \
  --metadata "$PRIVATE_SOURCE/metadata.json" \
  --folds "$PRIVATE_SOURCE/karsl_pose_signer_folds.csv" \
  --download-report "$PRIVATE_SOURCE/pose_download_report.json" \
  --output "$PRIVATE_SOURCE/features_502_v1"
```

`$PRIVATE_SOURCE` is a directory supplied by the team, not a value bundled here. The download report's `pose_root` must resolve on the machine running preprocessing. The script also needs the earlier private 30-class feature and test caches at `karsl/ml/artifacts/private/` for its historical-test exposure check.

The independent GRU entry point takes the prepared shards and a new output directory:

```bash
python karsl/ml/training/train_gru_502.py \
  --features-root "$PRIVATE_SOURCE/features_502_v1" \
  --cache "$PRIVATE_SOURCE/train_cache" \
  --output "$PRIVATE_SOURCE/new_gru_experiment" \
  --device cuda
```

The frozen CNN training/comparison script requires the original reference GRU checkpoint hash and a CUDA machine:

```bash
python karsl/ml/ml_expansion/scripts/signova_compare_cnn_502.py \
  --base "$PRIVATE_SOURCE/ml_expansion" \
  --cache "$PRIVATE_SOURCE/train_cache"
```

Calibration and test evaluation accept the same `--base` and `--cache` pattern. Keep the frozen validation-only selection and one-time test protocol. Do not tune any threshold or model using the reported test results. Reproduction of the full original run needs the private manifests, feature shards and original experiment files; the present compact ZIP is not a complete retraining dataset.

## Prediction interface

Use the private variant or authorized controlled artifacts with these paths:

```python
from pathlib import Path
import sys

project = Path.cwd()
sys.path.insert(0, str(project / 'karsl/ml/ml_expansion/scripts'))
from signova_predict_502 import KArSL502Predictor

predictor = KArSL502Predictor(
    base=project / 'karsl/ml/ml_expansion',
    repo_root=project,
)
result = predictor.predict_pose_file('/trusted/path/to/sample.pose', top_k=3)
```

`predict_pose_bytes(bytes)` decodes a source-compatible `.pose` internally. `predict_features(features, preprocessing_version)` accepts exactly one finite `(32, 150)` sample with version `pose_v1_32x50_xy_visibility_shoulder_conf05`; it is not a webcam, MP4 or `(T,126)` input. The result includes status, uncertain flag, nullable top-level class/labels, confidence, advisory `top_k`, and model/calibration versions. The frozen acceptance rule leaves the top-level label null. `reports/backend_contract_502.md` contains the full schema and the separate downstream interface constraints.

## Recorded results and release boundary

The historical CNN was selected on validation macro F1 0.351794 (GRU 0.342525). Its primary test summary reports 52.5187% accuracy, 0.485208 macro F1 and 74.2908% top-5 over 24,536 PASS examples with 500 represented true classes and 502 output classes. Classes 27 and 28 were excluded; historical test exposure exists. These numbers are not a fresh independent-video benchmark. See `reports/KArSL_502_Full_Delivery_Audit.md` for the exact remaining plan and team-message gates.

Keep checkpoints, Pose, derived arrays and mapping candidates in controlled storage until provenance and release rights are reviewed. The private checkpoint package is for authorized team use only. The existing original-video CV pilot is diagnostic and not a required new-video collection step for this ML research branch.

## Handoff status

See `reports/handoff_status_2026-10-26.md` for the team-confirmed usage and label review status. This branch carries code and aggregate research reports. Private model weights, feature shards and source Pose files are distributed separately. The frozen acceptance policy still marks every prediction uncertain; video-to-Pose equivalence and OOV recognition have not been established.

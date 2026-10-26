# KArSL-502 ML research handoff

This directory contains the isolated-sign KArSL-502 research baseline. It is a separate Arabic Sign Language reference, not an Egyptian Sign Language recognizer or continuous translator. The current frozen CNN returns `uncertain` for every prediction. `integration/research_preview.py` exposes the top-ranked class as a provisional `research_candidate` with `confirmed=false` for Backend/NLP development. `integration/karsl_cv_adapter.py` provides a separate experimental RGB-frame path. See `reports/cv_nlp_development_contract.md`.

## Layout

`src/preprocessing.py` is the exact source used to create the 32 × 150 features. `scripts/` contains the frozen feature preparation, CNN training/comparison, calibration, evaluation and prediction scripts. `training/train_gru_502.py` and `training/train_baselines_502.py` are standalone training/replay entry points; the latter verified the historical centroid and linear checkpoint metrics on all frozen train/validation PASS samples. `artifacts/` contains the selected CNN, GRU comparison checkpoint, calibration config, and 502-class label mapping needed to load the final predictor directly from this checkout. `notebooks/07_KArSL_Colab_Verification.ipynb` preserves the source audit and `notebooks/08_KArSL_502_GRU_Training.ipynb` the model experiments. The 07 notebook retains earlier failed cells followed by successful retries, not a clean top-to-bottom run. `reports/detailed_metrics/` contains the source test protocol, 502-row per-class validation/test results and class-ID confusion matrices. Source video/Pose and full feature shards are not needed for inference and are not bundled in this checkout. Artifact hashes are in `reports/checkpoints_manifest.md`.

The new GRU script has now been run on all frozen PASS train/validation shards in Colab. Its 23 history rows and all 20 model-state tensors exactly match the original saved GRU; the new checkpoint has a different file SHA-256 because it includes an extra experiment configuration field. A clean replay of the CNN training/comparison script yielded a byte-identical selected CNN checkpoint and matching 18-row history. Replaying the original validation calibration and terminal test protocols produced a byte-identical calibration configuration, test protocol, primary per-class CSV and confusion matrix, with identical primary and secondary metrics. See the three replay verification reports in `reports/`. Neither training script loads the test split. GRU output directories must be new to avoid overwriting the historical model. A replay of an already observed test is a reproducibility check, not a new independent test.

## Installation and smoke tests

Use a separate Python environment for ML. `requirements_502_cpu_py312.lock.txt` fully pins and hashes the CPU/Linux Python 3.12 dependencies and was installed in a clean environment with the smoke suite passing. The earlier Tesla T4 CUDA run has recorded direct versions but not a complete transitive GPU lock. The experimental CV adapter uses separate NumPy 1.26.4/MediaPipe 0.10.21 dependencies. See `reports/environment_502.md`.

```bash
python -m pip install uv==0.13.4
uv venv .venv-karsl --python 3.12
uv pip sync karsl/ml/requirements_502_cpu_py312.lock.txt \
  --python .venv-karsl/bin/python --torch-backend cpu --require-hashes
.venv-karsl/bin/python -m unittest discover -s karsl/ml/tests -v
```

Run those commands from the repository root. The release test loads the committed CNN checkpoint and checks the prediction contract without retraining. Tests do not certify accuracy, sign semantics, OOV detection, or full retraining.

## Controlled data and training

For retraining, obtain the pinned Pose source and frozen metadata, signer folds, download inventory, earlier 30-class reference features and full 502-class shards through the team's dataset storage. A checkout already contains everything required for prediction; the raw training data is not needed to load the model. Do not substitute another feature representation or split. The frozen training and preprocessing scripts have recorded SHA-256 identities; edits to them require an explicitly new experiment version.

The feature-preparation script takes explicit paths:

```bash
python karsl/ml/scripts/signova_prepare_features.py \
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

The two simple references have a standalone full-shard command; the historical linear checkpoint is available only through controlled storage:

```bash
python karsl/ml/training/train_baselines_502.py \
  --features-root "$PRIVATE_SOURCE/features_502_v1" \
  --cache "$PRIVATE_SOURCE/train_cache" \
  --output "$PRIVATE_SOURCE/new_baseline_experiment" \
  --device cpu --epochs 30
```

To verify the *older* baseline metrics using its original checkpoint, use the separate `--verify-historical-checkpoint "$PRIVATE_BASELINE"` option with a new output path. `reports/baseline_replay_502.md` gives both commands, full validation results and the distinction between the historical checkpoint replay and the new linear training run.

The frozen CNN training/comparison script requires the original reference GRU checkpoint hash and a CUDA machine:

```bash
python karsl/ml/scripts/signova_compare_cnn_502.py \
  --base "$PRIVATE_SOURCE/ml_expansion" \
  --cache "$PRIVATE_SOURCE/train_cache"
```

Calibration and test evaluation accept the same `--base` and `--cache` pattern. Keep the frozen validation-only selection and one-time test protocol. Do not tune any threshold or model using the reported test results. Reproduction of the full original run needs the private manifests, feature shards and original experiment files; the present compact ZIP is not a complete retraining dataset.

## Prediction interface

From the repository root, the published artifact paths are resolved automatically:

```python
from pathlib import Path
import sys

project = Path.cwd()
sys.path.insert(0, str(project / 'karsl/ml/scripts'))
from signova_predict_502 import KArSL502Predictor

predictor = KArSL502Predictor()
result = predictor.predict_pose_file('/trusted/path/to/sample.pose', top_k=3)
```

`predict_pose_bytes(bytes)` decodes a source-compatible `.pose` internally. `predict_features(features, preprocessing_version)` accepts exactly one finite `(32, 150)` sample with version `pose_v1_32x50_xy_visibility_shoulder_conf05`; it is not a webcam, MP4 or `(T,126)` input. The result includes status, uncertain flag, nullable top-level class/labels, confidence, advisory `top_k`, and model/calibration versions. The frozen acceptance rule leaves the top-level label null. `reports/backend_contract_502.md` contains the full schema and the separate downstream interface constraints.

For a clearly marked development preview, use `predict_for_preview(predictor, pose_bytes=trusted_pose_bytes, top_k=3)` from `integration/research_preview.py`. This returns a nested `candidate` class/gloss for Backend and NLP with `confirmed=false`; it does not change the frozen predictor. The experimental video adapter reads isolated RGB frames and applies the **same** frozen preprocessor, but is not a production camera claim. See `integration/README.md`, `reports/cv_nlp_development_contract.md` and `reports/KArSL_CV_Pilot_Report.md`.

## Recorded results and release boundary

The historical CNN was selected on validation macro F1 0.351794 (GRU 0.342525). Its primary test summary reports 52.5187% accuracy, 0.485208 macro F1 and 74.2908% top-5 over 24,536 PASS examples with 500 represented true classes and 502 output classes. Classes 27 and 28 were excluded; historical test exposure exists. These numbers are not a fresh independent-video benchmark. See `reports/KArSL_502_Full_Delivery_Audit.md` for the exact remaining plan and team-message gates.

The project owner authorized publication of the trained CNN/GRU checkpoints, calibration and source label mapping for this team handoff. Raw Pose, source videos and full derived feature shards remain available through the dataset source/storage, not in Git. The existing original-video CV pilot is diagnostic and not a required new-video collection step for this ML research branch.

## Handoff status

See `reports/handoff_status_2026-10-26.md` for the handoff and label review status. This branch carries code, model weights, calibration, labels, tests and research reports. Training feature shards and source Pose files remain with the dataset source/storage. The frozen acceptance policy still marks every prediction uncertain; video-to-Pose equivalence and OOV recognition have not been established.

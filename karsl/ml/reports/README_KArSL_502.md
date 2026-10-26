# SIGNOVA KArSL 502-class research baseline

This package implements a closed-set, isolated-sign classifier for a third-party KArSL Pose collection. It is a research reference for the SIGNOVA team. It does not establish Egyptian Sign Language recognition, continuous translation, unknown-sign detection, or production reliability.

## Status and directory layout

The selected model is the 502-output `SignCNN502` at epoch 12. Its original checkpoint, frozen calibration, source metadata, and feature shards are private artifacts in `/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion/`. The older `ml_delivery_english/` package remains the separate 30-class prototype; its `karsl/ml/src/preprocessing.py` is the exact version used to build the 502-class features. Do not substitute the earlier 30-class checkpoint or class mapping.

Relevant files:

| Path under `ml_expansion/` | Purpose |
| --- | --- |
| `scripts/signova_prepare_features.py` | Resume-safe Pose decoding, preprocessing, QC, and frozen shards. |
| `scripts/signova_compare_cnn_502.py` | 502-class CNN training, GRU validation replay, and model comparison. |
| `scripts/signova_calibrate_cnn_502.py` | Validation-only temperature and acceptance-policy analysis. |
| `scripts/signova_evaluate_test_502.py` | One-time frozen test evaluation and inference benchmark. |
| `scripts/signova_predict_502.py` | Checkpoint loading and prediction from features or Pose data. |
| `artifacts/private/features_502_v1/` | Feature shards and full sample-level QC; do not publish. |
| `experiments/gru_502_pass_v1/` | Earlier 502-class GRU checkpoint and history. |
| `experiments/cnn_502_pass_v1/` | Selected CNN, history, configuration, calibration, and evaluation. |
| `source_metadata/labels_mapping.json` | Original 502-class labels; semantic approval is pending. |

The handoff archive is rooted at `SIGNOVA_KArSL_502/`. It contains `ml_expansion/` for this experiment, the exact shared preprocessing source under `ml_delivery_english/karsl/ml/src/`, `notebooks/`, `reports/`, and `tests/`. The code-only archive excludes model weights, feature arrays, source Pose files, and source label metadata. The private inference archive includes the selected CNN checkpoint, the frozen calibration and source labels, plus the comparison GRU checkpoint; it does not include the 75,515 source Pose files or full feature shards. Training replay requires the separately controlled source dataset or private Drive shards.

`reports/backend_contract_502.md` freezes the tested KArSL ML-side input/output shape and spells out the incompatible existing continuous-sign interface. `reports/next_handoff_502.md` identifies precisely what Backend, CV, and NLP can consume next. `reports/handoff_gap_review_502.md` maps this KArSL delivery to the team request and SIGNOVA plan; remaining implementation is not counted as completed work.

`reports/karsl_pre_release_audit.md` records the source-label and video/Pose checks. All 502 source label IDs and Arabic/English text match the original KArSL-502 spreadsheet; this does not validate that the external Pose samples depict the assigned sign or establish media and weight release rights.

`notebooks/07_KArSL_Colab_Verification.ipynb` contains the subsequent whole-source exact-duplicate audit and checkpoint/calibration hashes. `reports/KArSL_502_Continuation_Audit.md` reviews that evidence and the later CV pilot. `reports/checkpoints_manifest.md` and `reports/experiment_log_502.csv` preserve the artifact hashes and model comparison records supplied with that continuation. The 50 conflicting class-27/28 pairs are within the test split and were excluded under the previously frozen evaluation policy. None of these additions changes the selected checkpoint or the reported test metrics.

Use Python 3.13 with compatible NumPy, PyTorch, SciPy, scikit-learn, pandas, and `pose-format==0.15.0` when decoding `.pose` files. The Colab experiments used CUDA-enabled PyTorch; the inference loader also supports CPU. Install a CUDA-compatible PyTorch build for your runtime rather than copying a CPU-only wheel. Keep the exact `experiment.json`, frozen source fingerprint, and `runtime_sessions.json` with the research record.

In the private inference archive, install dependencies with `python -m pip install -r requirements_502.txt` in a compatible Python environment, then run `python -m unittest discover -s tests -v`. In the code-only archive, the frozen preprocessing check runs, while tests requiring private weights or a real source `.pose` file are skipped. To exercise the real file test, set `SIGNOVA_REAL_POSE` to a source-compatible `.pose` path. Passing this smoke test does not establish prediction accuracy.

Colab runtimes can reset independently of the Drive files. Before a real `.pose` inference check, install the decoder in that runtime and confirm its import:

```python
import subprocess
import sys

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pose-format==0.15.0'], check=True)
subprocess.run([sys.executable, '-c', 'from pose_format import Pose; print("Pose decoder: PASS")'], check=True)
```

An `ImportError: Install pose-format to decode .pose files` means this decoder is missing from the current Python environment; it does not invalidate checkpoint loading or the synthetic preprocessing check. Run the real-file check again after installation and record its actual outcome separately.

## Reproducing the frozen steps

Mount the private Drive folder. Run feature preparation once with the documented command and the frozen source files, or reuse its completed, hash-verified shards. The 502-class GRU notebook is `ml_expansion/08_KArSL_502_GRU_Training.ipynb`. From `ml_expansion/scripts/`, the subsequent standalone steps are:

```bash
python -u signova_compare_cnn_502.py
python -u signova_calibrate_cnn_502.py
python -u signova_evaluate_test_502.py
python -u signova_predict_502.py
```

The training, calibration, and evaluation scripts have frozen checksums and output identity checks. A completed test evaluation is reused on rerun; do not revise the threshold or select a new model from test results. Store trained weights, features, logits, and any source Pose files privately. The code-only handoff should reference controlled artifact locations rather than include source data or unpublished model weights.

## Prediction interface

`KArSL502Predictor(base, repo_root, device='cpu')` loads the selected checkpoint, original 502-class label source, frozen calibration, and the **same preprocessing implementation** used in training. It verifies their SHA-256 hashes and class order before prediction.

`predict_features(features, preprocessing_version, top_k=5)` accepts one numeric `32 × 150` feature array or nested list. The version must equal `pose_v1_32x50_xy_visibility_shoulder_conf05`. The 150 channels encode 50 selected landmarks as `(x, y, visible)` per frame. Raw landmarks, webcam frames, video, and a 30-class feature vector are not valid feature inputs. The method rejects nonfinite values, invalid visibility bits, and invalid coordinate ranges. This feature interface still requires the caller to supply trustworthy features produced by the frozen preprocessing.

`predict_pose_bytes(pose_bytes, top_k=5)` accepts the binary content of one source-compatible `.pose` file and applies `preprocess_pose` internally. `predict_pose_file(path, top_k=5)` reads a **trusted server-side** file path. Neither method extracts landmarks from video. `predict_pose_object(pose_object, top_k=5)` accepts a decoded Pose object with the source-compatible 543-landmark format.

```python
from pathlib import Path
from signova_predict_502 import KArSL502Predictor

predictor = KArSL502Predictor(
    base=Path('/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion'),
    repo_root=Path('/content/drive/MyDrive/SIGNOVA/KArSL/ml_delivery_english'),
)
result = predictor.predict_pose_file('/trusted/source/sample.pose', top_k=3)
```

The output always has `status`, `uncertain`, `class_id`, `label`, `label_arabic`, `confidence`, `top_k`, `model_version`, and `calibration_version`. Each top-k candidate has `class_id`, `label`, `label_arabic`, `score`, `label_review_status`, and `label_conflict`. Class IDs are **1–502**, while tensor indexes are **0–501**. Source class 1 maps to the sign gloss `0`; it is not an off-by-one error. The labels come from the dataset and have not received expert semantic approval.

The frozen policy did not meet its validation acceptance target. Consequently `status='uncertain'`, `uncertain=true`, the top-level class and labels are `null`, and the top-k list is advisory only. A caller must not display a top-k candidate as a confirmed translated sign. Unknown/OOV detection is unvalidated. The Python return schema is fixed for this KArSL research package; an HTTP route and CV adapter have not yet been implemented.

An independent Colab smoke check decoded the pinned source file `sign_0001_s2_sess2_014.pose` with `pose-format==0.15.0`, loaded the frozen model and preprocessing, and returned a schema-compliant uncertain result. Its highest candidate was class 181 (`think`) with score 0.611551; this is not a correct-label claim. The check establishes that this source-compatible binary path runs end to end, not recognition quality or out-of-vocabulary behavior.

## Results and restrictions

The selected CNN validation accuracy is **38.71%** over 25,254 PASS samples from signer 2. The primary test evaluation is **52.52% accuracy**, **0.485208 macro F1**, and **74.29% top-5** on **24,536 PASS samples from 500 evaluated true classes** from signer 1. The model continues to output 502 classes. The test exclusions, historical exposure, calibration limitations, per-class scores, confusion matrix, and hardware timing are detailed in `evaluation_report_502.md` and the frozen machine-readable experiment reports. Accepted-prediction coverage is **0%**, so accepted accuracy is undefined.

KArSL is a separate Arabic sign-language research source and must not be relabeled as Egyptian Sign Language. Its publisher's license listing and the original Pose/video provenance are not final rights approval. Do not upload raw Pose files, feature arrays, full predictions, or model weights to a public repository before source and release review.

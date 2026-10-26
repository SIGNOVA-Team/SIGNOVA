# SIGNOVA KArSL 502-output CNN: experiment and evaluation report

## Frozen design

The dataset is the pinned third-party `tfmohamedyahia/kars502-pose-75/versions/1` Pose collection, represented by 75,515 sample records and 502 source class IDs. Frozen fingerprint: `5d0d43950dec9c69bef2766e48550463a027b49a44748631b976f211c77e7c4a`. The original processing implementation has SHA-256 `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c` and produces `(32, 150)` float32 features. It passed parity checks against eight earlier cached samples.

Signer 3 supplies train samples, signer 2 validation, and signer 1 test. Source QC produced 75,392 technical PASS, 119 REVIEW, and four FAILED records. The model comparison used 25,032 PASS train and 25,254 PASS validation samples; no REVIEW or FAILED record was silently converted into a valid zero tensor. Exact raw duplicates do not cross the frozen splits. Source labels and rights remain subject to review.

| Candidate | Best validation epoch | Validation accuracy | Validation macro F1 | Validation top-5 |
| --- | ---: | ---: | ---: | ---: |
| Bidirectional GRU | 17 | 37.7564% | 0.342525 | 59.9192% |
| 1D-CNN | 12 | 38.7067% | 0.351794 | 63.2415% |

Both models were compared on the same 502 classes and PASS validation split. The CNN was selected by the prespecified validation macro-F1 criterion. The 30-class prototype is a separate experiment and is not a comparable 502-class baseline. The selected CNN has 576,886 parameters and a 2.215 MiB inference checkpoint. Its training history records 18 epochs, with best epoch 12 and early stopping.

## Calibration and acceptance

Temperature scaling was fitted without test predictions: **T = 2.4141954373370376**. Per-class deterministic validation partitions contained 12,557 temperature-fit, 6,053 threshold-selection, and 6,644 policy-audit samples. The 15-bin policy-audit expected calibration error decreased from **0.347401** to **0.046079**. Temperature scaling preserves the top-class ranking; it does not improve classification accuracy.

No threshold passed the recorded exploratory selection criterion of at least 100 accepted samples and a descriptive 95% Wilson lower bound of at least 80% accepted accuracy. Acceptance is therefore **disabled**. The system returns `uncertain` for every input; top-k entries remain unverified candidates. The validation partitions were already used for model/epoch selection, so this is post-selection analysis, not an independent estimate or a production guarantee.

## Frozen test evaluation

The test protocol was recorded before generating the new 502-output CNN test predictions. The primary diagnostic subset contains **24,536 PASS samples from 500 true classes**. It excludes six test REVIEW samples, all 150 samples from source classes 27 and 28 due to unresolved conflicting duplicate labels, and 420 further paths used in historical evaluations. Thirty of the 450 historically evaluated paths already occur in the excluded two classes. No source labels or files were edited. The selected model still has **502 output classes**; a prediction into 27 or 28 for another evaluated true class counts as an error. Macro F1 averages the fixed 500 evaluated true classes.

| Metric | Primary test result |
| --- | ---: |
| Forced-choice accuracy | 52.5187% |
| Macro F1 across 500 evaluated true classes | 0.485208 |
| Top-5 accuracy across 502 output scores | 74.2908% |
| ECE before temperature scaling | 0.265556 |
| ECE after frozen temperature scaling | 0.111822 |
| Accepted-prediction coverage | 0.00% |
| Accepted-prediction accuracy | Undefined: no accepted predictions |

A secondary 24,956-sample diagnostic includes historical test paths and is recorded separately, not used for model selection or tuning. The primary subset still comes from a source pool with historical exposure. Some evaluated classes have only two primary test samples. These figures are **not** a clean full-502-class benchmark or an end-to-end translation score. The branch includes class-ID per-class results, the 502-column confusion matrix and frozen protocol in `detailed_metrics/`; class IDs 1–502 can be joined to `../artifacts/labels_mapping.json`. The original reliability bins and high-confidence mistakes remain in the separate experiment directory `ml_expansion/experiments/cnn_502_pass_v1/test_evaluation_v1/`.

## Real Pose smoke verification

In Colab, the selected checkpoint and frozen preprocessing hash reloaded successfully. Synthetic Pose-to-feature and direct feature predictions matched. After installing `pose-format==0.15.0`, decoding and prediction also completed for the real pinned source file `sign_0001_s2_sess2_014.pose`. The response had the required schema, `status=uncertain`, and confidence 0.611551. Its highest candidate was class 181 (`think`), which must not be treated as a verified translation of this file. This one-file smoke check establishes executable file-path compatibility only; it does not change the test metrics or validate unknown-sign rejection.

## Performance

The timed operation used synthetic, preconstructed normalized feature tensors. It included device transfer, CNN execution, temperature softmax, top-5 selection, and transfer of scores to CPU arrays, after 20 warm-up and 100 measurement iterations per batch size. It excluded `.pose` decoding, landmark extraction from video, model loading, input validation, JSON encoding, backend transport, and UI time.

| Device | Batch | Mean batch latency | P95 batch latency |
| --- | ---: | ---: | ---: |
| CUDA runtime | 1 | 1.025 ms | 1.108 ms |
| CUDA runtime | 32 | 1.261 ms | 1.353 ms |
| CPU runtime | 1 | 0.814 ms | 0.954 ms |
| CPU runtime | 32 | 9.266 ms | 14.964 ms |

## Limits and follow-up

This is an isolated-sign Pose research baseline. It does not implement webcam/video landmark extraction, continuous translation, tested unknown-sign detection, or production integration. Dataset label meanings, Pose/video provenance, and rights for redistribution or model release have not been independently approved. KArSL is not an Egyptian Sign Language dataset. The backend contract needs team review. Improve later model versions using fresh training/validation evidence, without adjusting this frozen model or its policy to the reported test scores.

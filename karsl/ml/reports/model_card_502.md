# Model card: SIGNOVA KArSL SignCNN502 v1

**Research handoff update (2026-10-26):** The owner authorized inclusion of the trained CNN/GRU checkpoints, calibration and source label mapping in the existing public `karsl` branch for team integration. The experimental CV adapter and provisional Backend/NLP candidate contract are also included. This publication does not change the frozen checkpoint, 0% accepted coverage or production limits below. The exact research contracts and diagnostics are in `cv_nlp_development_contract.md` and `KArSL_CV_Pilot_Report.md`.

**Version:** `karsl_cnn_502_pass_v1`  
**Type:** Closed-set 1D temporal CNN over 32 frames × 50 landmarks × `(x, y, valid)`  
**Parameters:** 576,886  
**Selected checkpoint:** `best_cnn_502_pass_v1.pt`, epoch 12; SHA-256 `f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254`  
**Frozen preprocessing:** `pose_v1_32x50_xy_visibility_shoulder_conf05`; source implementation SHA-256 `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c`  
**Calibration:** `cnn502_validation_temperature_v1`, temperature 2.4141954373370376, acceptance disabled.

The intended use is reproducible research on the specific third-party KArSL Pose representation and class IDs 1–502. It is not an Egyptian Sign Language recognizer, a general Arabic/ASL model, an interpreter, an OOV detector, or a production translation component. The developer-facing interface is in `signova_predict_502.py`; no top-k label is a confirmed translated sign under the frozen policy.

The pinned source has 75,515 entries. Frozen signer-disjoint groups are 25,032 PASS train from signer 3, 25,254 PASS validation from signer 2, and signer-1 test with documented exclusions. The selected model's validation accuracy is 38.7067%, macro F1 is 0.351794, and top-5 accuracy is 63.2415%. The test report's **52.5187% accuracy, 0.485208 macro F1, and 74.2908% top-5** apply to a restricted **500-true-class, 24,536-sample** subset while the classifier still produces 502 outputs. Classes 27 and 28 and previously evaluated samples were excluded by a frozen metadata policy; do not present those metrics as a full-502-class test.

Post-selection validation calibration reduced policy-audit ECE from 0.347401 to 0.046079, but no acceptance threshold satisfied the recorded criterion. The deployed research policy returns uncertain for all inputs: **coverage 0% and accepted accuracy undefined**. On the primary test subset, ECE fell from 0.265556 to 0.111822. Calibration does not improve top-1 accuracy. Unknown/OOV behavior has not been validated.

Key limitations include uncertain label semantics, conflicting raw duplicates for source classes 27 and 28, historical exposure of 450 test paths in older experiments, one test signer, very low test support for some classes after exclusion, and reuse of validation for epoch/model selection before calibration analysis. The upstream dataset listing's declared CC BY-SA 4.0 license is not an independent provenance audit. Source Pose/video correspondence remains unverified. The model artifacts were published on the project owner's authorization; raw Pose, original videos and full feature shards are not included.

An October 2026 source audit compared the private 502-class mapping with the official KArSL-502 label spreadsheet: all 502 sign IDs and original Arabic/English text fields matched. The repository's separate 500-class reviewed mapping contains 48 proposed English changes, including six explicitly uncertain glosses, without expert validation. Neither check verifies the sign movements in the third-party Pose dataset.

The benchmark measures ready-made feature inference and top-5 on one recorded runtime, not webcam-to-response latency. The backend team has not approved the provisional JSON contract. Further model changes require a new documented experiment that does not tune on the frozen test results.

A Colab smoke check decoded one real source-compatible `.pose` file and produced a schema-compliant uncertain response after installing `pose-format==0.15.0`. The first candidate differed from the source file's apparent class ID; this is a compatibility check, not a model-correctness or OOV validation.

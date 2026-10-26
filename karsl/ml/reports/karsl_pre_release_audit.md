# KArSL 502 pre-release audit

**Date:** 2026-10-25. **Scope:** this KArSL research branch only. A passing inference smoke test is not a completed video-to-model integration or a release authorization.

## Source and labels verified here

| Check | Result |
| --- | --- |
| Original source | The [KArSL project page](https://hamzah-luqman.github.io/KArSL/) identifies 502 isolated Arabic Sign Language words, three professional signers, and RGB/depth/skeleton modalities. Its nominal count is 75,300 samples. It requests citation but does not state an explicit dataset license on that page. |
| Source label spreadsheet | The original KArSL-502 Drive folder contains `KARSL-502_Labels.xlsx`, SHA-256 `c13717c549b8cb8cfa465237a3f7dfed73f84149ca5c448b2b253fd96321b14e`. It has 502 ID, Arabic, and English rows. |
| Model label mapping | The 502 entries in the private `labels_mapping.json`, SHA-256 `3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a`, match the original source spreadsheet's IDs and both label text columns exactly after trimming whitespace: **502/502 matched, 0 text mismatches**. This proves transcription from the spreadsheet, not the physical meaning of every signed movement. |
| Existing repository review mapping | `karsl/data/metadata/karsl_class_mapping.csv` contains 500 rows and omits classes 27 and 28. Its 500 original label pairs match the private source mapping. It lists 48 English revisions: 42 marked `english_gloss_corrected_no_expert_review` and six `gloss_uncertain_needs_expert_check`. The model currently returns the original publisher glosses, not those tentative revisions. |

## Video-to-Pose evidence and remaining gap

The team's existing KArSL dataset card inventories 75,514 original video records versus 75,515 external Kaggle Pose records. A candidate mapping connects 75,514 rows by signer, folder class, and within-group order, with 75,513 matching frame counts. One additional five-frame Pose and a frame-count exception remain. The mapping is not an independent identity proof. In the earlier 12-pair diagnostic, 11 identities remained unverified and one coordinate overlay did not align. Those reports also flag 33 disputed class-5/6 assignments and original-video conflicts for classes 27/28.

Consequently, the verified model input remains the external source-compatible `.pose` representation. Decoding one real Pose file and predicting successfully does not show that MediaPipe output from an original KArSL video or a webcam matches the trained feature distribution. The existing repository CV stream `(T,126)` is a different representation. A real video adapter must be developed and evaluated against source-compatible Pose data; if equivalence cannot be established, the honest path is to extract a new training set from original videos and retrain/evaluate a new model version.

The current [MediaPipe Holistic Landmarker documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/holistic_landmarker) describes 553 landmarks (33 body, 478 face, 42 hands), whereas the trained source Pose object has 543 (33 body, 468 face, 42 hands). The frozen model selects hand indices 501–542. A current Holistic result cannot be passed into the old 543-index preprocessor directly; even a deliberate body/hand remapping must be validated for coordinate and confidence semantics against the training source.

## Unknown signs and acceptance

All 502 source classes were included in training. The excluded test classes 27 and 28 are still model output classes and were not held out during training; they cannot serve as true out-of-vocabulary examples. The frozen validation rule selected no safe acceptance threshold. Current coverage is zero and all responses are uncertain. No independently labeled unknown-sign set or OOV detection result exists. Blank frames, random arrays, and shuffled familiar signs would be stress tests, not evidence of recognition of unseen signs.

## Publication and acceptance gates

1. Obtain written clarification from the original KArSL rights holder covering academic training, third-party derived Pose files, sharing code/derived metadata, publishing checkpoints, and any public demo. The external Kaggle listing's claimed license does not by itself establish that the original rights holder authorized the derivative Pose release. Do not infer release permission from a download link.
2. Review the 48 proposed English gloss edits, 33 disputed assignments, classes 27/28, and representative source videos with a qualified sign-language reviewer. Record signed-off decisions and keep original IDs/labels intact.
3. Implement a video extraction path and demonstrate frame/landmark semantics, coordinate system, and prediction stability on verified video/Pose pairs. The previous non-aligned overlay prevents a compatibility claim now.
4. Acquire a consented, labeled negative set of genuine signs outside the 502-class vocabulary if OOV behavior is an acceptance criterion. Freeze a fresh evaluation protocol; do not tune the existing checkpoint to its reported test set.
5. Run the packaged code and private model from a clean checkout with controlled artifact pointers, complete a branch-ready manifest, and keep private data/weights outside the public repository until release terms are clarified.

The first two gates require source-owner and qualified reviewer input. They cannot be marked complete by code execution or by the label-spreadsheet equality check.

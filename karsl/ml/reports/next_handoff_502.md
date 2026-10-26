# KArSL 502: immediate CV, NLP, and Backend handoff

## Current state

The selected isolated-sign KArSL CNN loads from a private checkpoint. The real source `.pose` inference smoke test passed. It applies the same verified `(32,150)` preprocessing used in training. The active policy returns `uncertain` for every sample because no validation-selected acceptance threshold met the recorded target. The 52.5187% primary test accuracy covers 24,536 samples from 500 evaluated true classes, while the model outputs 502 scores. This is not an Egyptian Sign Language system or a deployable interpreter.

## What Backend can implement now

Load one `KArSL502Predictor` instance from the private model bundle and call `predict_pose_bytes(pose_bytes, top_k=3)` for one trusted source-compatible KArSL `.pose` binary. Preserve the returned dictionary: `status`, `uncertain`, nullable `class_id`, `label`, `label_arabic`, calibrated `confidence`, `top_k`, `model_version`, and `calibration_version`. Keep this KArSL isolated-sign path separate from the repository's continuous ArabSign/How2Sign `(T,126)` route. Catch validation/decode errors at the service boundary; an HTTP request/error schema has not been implemented by the ML package. Do not forward an uncertain top-k candidate as translated text.

## What CV needs to provide later

For a camera-to-KArSL path, produce a source-compatible Pose sequence with body data `(F,1,543,3)` and confidence `(F,1,543)`, or a compatible `.pose` binary. Preserve the landmark order, body+hand components, confidence semantics, and coordinate conventions. The ML preprocessor then selects 32 frames and produces `(32,150)`. Test a real CV-produced sample against the frozen preprocessor and document where the source video/Pose correspondence comes from. The existing `(T,126)` hand-keypoint stream targets different models; it cannot feed this CNN by reshaping.

Current MediaPipe Holistic documents a 553-landmark output rather than this model's 543-landmark Pose layout. A mapping or a fresh training representation is required; the prior video/Pose spatial overlay did not establish equivalence.

## What NLP receives now

For current KArSL predictions: `recognized_gloss = null`, `recognized_text = null`, `status = uncertain`. The candidate list can be shown in a research/debug view with its `unverified` label status, but is not an NLP translation input. NLP may develop and test its own gloss-to-text interface on explicit labeled fixtures independently. It should not treat KArSL class names as Egyptian Sign Language, or use its top-1 candidate as a sentence token. A future accepted-gloss handoff requires a validated policy and semantic label review.

## Remaining items before declaring the overall project done

1. Complete source rights, Pose/video provenance, and semantic label review for this KArSL reference; keep weights and source arrays in controlled storage until release is approved.
2. Build and validate CV extraction for an actual target language track. If it is KArSL, demonstrate source-compatible Pose output; if it is ESL or ASL, use that track's own data, model, preprocessing, and labels.
3. Evaluate unknown/OOV signs and derive a safe acceptance policy on fresh validation evidence for a later model version; do not tune the frozen CNN from its test score.
4. Define and test a Backend route for each linguistic/model track. The research KArSL route can show an uncertain status now; real-time translated text requires stronger model and integration evidence.
5. Land the code-only package in a focused branch after checking the repo layout and artifact pointers. Keep the private model package outside the public repository; hand over its checksum and controlled location.

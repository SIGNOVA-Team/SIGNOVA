# KArSL 502 research inference contract v1

**Development option:** `../integration/research_preview.py` exposes the frozen predictor's top-ranked class under `candidate` with `confirmed=false`; the separate RGB adapter and research CV/NLP contract are in `../integration/karsl_cv_adapter.py` and `cv_nlp_development_contract.md`. The frozen response documented below is unchanged.

**Status:** Frozen and tested as the ML-side Python contract for the KArSL research track. This document does not claim that a CV adapter or shared HTTP route has been implemented. This is an isolated-sign KArSL Pose research model. It does not accept a webcam frame, an MP4, a `(T, 126)` dual-hand tensor, or a continuous signing sequence.

## Supported Python inputs

Instantiate `KArSL502Predictor(base, repo_root, device='cpu')` once. The loader checks the checkpoint, calibration, label source, model implementation, and preprocessing SHA-256 fingerprints and class ordering.

| Method | Input | Preprocessing ownership |
| --- | --- | --- |
| `predict_pose_bytes(pose_bytes: bytes, top_k=5)` | Nonempty binary source-compatible `.pose` file | The predictor decodes and runs the frozen training preprocessor. Recommended boundary for a trusted service that already has a compatible Pose file. |
| `predict_pose_object(pose_object, top_k=5)` | Decoded Pose body data `(F, 1, 543, 3)` and confidence `(F, 1, 543)` | The predictor runs the same frozen preprocessor. |
| `predict_features(features, preprocessing_version, top_k=5)` | One finite `float32`-compatible array/list `(32, 150)` and version `pose_v1_32x50_xy_visibility_shoulder_conf05` | The caller must produce features using the exact frozen preprocessor; the predictor validates their shape, finite values, visibility bits, and coordinate range. |
| `predict_pose_file(path, top_k=5)` | Trusted server-side source-compatible `.pose` path | Like `predict_pose_bytes`; never treat a client-supplied path as a safe file location. |

The frozen feature shape is 32 frames × 50 landmarks × `(normalized_x, normalized_y, valid)`, flattened frame-major. Landmark order is body indices `11,12,13,14,15,16,23,24`, then hand indices `501..542` from the source-compatible 543-landmark Pose layout. A landmark is valid when its source confidence is greater than 0.5 and x/y are finite. Coordinates are centered and scaled using the two shoulders, clipped to `[-5, 5]`, invalid x/y are zero, and 32 frames are uniformly selected. The source implementation and its fingerprint are supplied in the package. These 150 values are **not** the existing repo's 126 xyz hand-keypoint representation.

## Exact returned dictionary

```json
{
  "status": "uncertain",
  "uncertain": true,
  "class_id": null,
  "label": null,
  "label_arabic": null,
  "confidence": 0.611551,
  "top_k": [
    {
      "class_id": 181,
      "label": "think",
      "label_arabic": "يفكر",
      "score": 0.611551,
      "label_review_status": "unverified",
      "label_conflict": false
    }
  ],
  "model_version": "karsl_cnn_502_pass_v1",
  "calibration_version": "cnn502_validation_temperature_v1"
}
```

The example is shortened to one candidate for readability; the verified real-file call used `top_k=3`. Source class IDs are 1–502; tensor indices are 0–501. `confidence` is the temperature-scaled maximum closed-set softmax probability, not a correctness guarantee. The frozen acceptance policy is disabled, so every current result has `status='uncertain'`, null top-level class/labels, and candidate `top_k` entries for investigation only. Unknown/OOV detection has not been validated. Invalid inputs or missing artifacts raise Python exceptions; an HTTP error schema has not been agreed.

## Downstream handoff

The existing `docs/CV_INTEGRATION_GUIDE.md` and `ml/sign_language/src/inference.py` in the repository target continuous ArabSign/How2Sign models. They accept `(T, 126)` keypoints or video and return `success`, `language`, `text`, `tokens`, and related fields. Their preprocessing, model, linguistic track, and response semantics differ from this KArSL branch. CV must produce a source-compatible 543-landmark Pose object/binary and demonstrate parity on real samples before this model can consume camera output; reshaping 126 values to 150 is invalid. Backend can use the tested Python dictionary above as the KArSL-specific internal contract now, while keeping it separate from the existing continuous-sign route. NLP must receive no confirmed KArSL gloss when `status='uncertain'`; `top_k` is available for research inspection only. A public HTTP shape, real-time CV path, and production translation behavior remain future implementation work.

The current MediaPipe Holistic task describes 553 landmarks, including 478 face landmarks; this model's source Pose layout has 543, including 468 face landmarks. The selected body and hand points can be mapped deliberately, but the layout, coordinates, and confidence meanings must be validated. Directly passing a modern 553-landmark array into the frozen 543-landmark preprocessor is invalid.

# KArSL-502 CV, Backend and NLP development contract

The frozen selected CNN ranks 502 classes but currently accepts none as a confirmed sign. For development, `../integration/research_preview.py` wraps the unchanged predictor and provides the top-1 `candidate` with `status="research_candidate"`, `confirmed=false`, and `model_status="uncertain"`. This supplies a concrete class/gloss for wiring and research demos without misreporting accuracy or editing the frozen calibration.

## CV / Backend inputs

| Input | Exact contract |
| --- | --- |
| `pose_bytes` | One nonempty source-compatible binary `.pose` for one isolated sign. `KArSL502Predictor` performs the frozen decoding/preprocessing. |
| `pose_object` | Body data `(F,1,543,3)` and confidence `(F,1,543)` with matching coordinate and landmark meanings. |
| `features` | One finite `(32,150)` array plus `preprocessing_version="pose_v1_32x50_xy_visibility_shoulder_conf05"`, produced by the exact frozen source. |
| Original RGB frames, experimental | Separate `karsl_cv_adapter.py`: RGB `uint8` frames of one isolated sign, maximum 300; MediaPipe Holistic 0.10.21 with heavy pose model/tracking; the adapter then runs the frozen preprocessor and calls the same predictor. This route requires a separate CV environment and is not validated for arbitrary webcams. |

An MP4, 553-point modern landmark stream, or the existing repository's `(T,126)` hand stream cannot be passed directly to the trained model. Backend should instantiate the private-artifact-backed predictor once. Never allow an untrusted client to choose arbitrary server file paths.

```python
from karsl.ml.integration.research_preview import predict_for_preview

preview = predict_for_preview(predictor, pose_bytes=trusted_pose_bytes, top_k=3)
```

```json
{
  "status": "research_candidate",
  "confirmed": false,
  "candidate": {"class_id": 181, "label": "think", "label_arabic": "يفكر", "score": 0.611551},
  "confidence": 0.611551,
  "top_k": [{"class_id": 181, "label": "think", "label_arabic": "يفكر", "score": 0.611551}],
  "model_status": "uncertain",
  "model_version": "karsl_cnn_502_pass_v1",
  "calibration_version": "cnn502_validation_temperature_v1"
}
```

This example abbreviates the diagnostic list; actual `top_k` also has review/conflict fields. Source class IDs are 1–502, tensor indices 0–501. NLP may use `candidate.label` in an explicitly provisional development flow and retain `confirmed=false` on any generated text. It must not silently render a candidate as a verified translation. The threshold required to make a confirmed result has **not** been validated.

The earlier CV pilot checked 12 candidate original-video/Pose pairs and a separately predeclared 20-pair S03 train/S02 validation development follow-up. With the heavy/tracking setting, the first group had 11/12 matching top-1 decisions; the second 19/20. A direct recheck on 15 original JPEG frames produced finite `(32,150)` features with the packaged adapter. These are *pair consistency diagnostics*, not fresh video recognition accuracy; one pair in each set differed, source-video/Pose identity is not established for the whole collection, and all compared model outputs remained uncertain. See `KArSL_CV_Pilot_Report.md` and the included aggregate JSON. No new filming is needed merely to integrate the research path. A fresh, independently labeled video/OOV evaluation is required before promising certain or production translations.

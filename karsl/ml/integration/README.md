# KArSL-502 CV, Backend and NLP handoff

Clone the existing `karsl` branch and install the ML environment from `../README.md`. The selected CNN weights, calibration, labels and exact training preprocessor are already in the checkout. The predictor is an isolated KArSL research model, separate from the repository's continuous `(T,126)` models.

## Backend: source-compatible Pose input

From the repository root, instantiate the model once and pass one trusted `.pose` binary per isolated sign:

```python
from karsl.ml.scripts.signova_predict_502 import KArSL502Predictor

predictor = KArSL502Predictor()
result = predictor.predict_pose_bytes(pose_bytes, top_k=3)
```

The input must be nonempty `bytes` encoded with `pose-format` and compatible with the frozen 543-landmark layout. For already-preprocessed data, call `predict_features(features, preprocessing_version="pose_v1_32x50_xy_visibility_shoulder_conf05", top_k=3)` with a finite `float32` array of shape `(32,150)`. The same `preprocess_pose` source creates features for training and prediction. Invalid shapes, versions or missing artifacts fail explicitly.

The returned dictionary has `status`, `uncertain`, nullable `class_id`, `label`, `label_arabic`, calibrated `confidence`, advisory `top_k`, `model_version` and `calibration_version`. The frozen policy returns `status="uncertain"` and null top-level labels; this must not be presented as a verified translation. For UI and NLP development, call `predict_for_preview(predictor, pose_bytes=pose_bytes, top_k=3)` from `research_preview.py`. It supplies `candidate` with `confirmed=false` and `model_status="uncertain"` without changing the model's result.

## CV: experimental RGB path

In a CV environment with `mediapipe==0.10.21` and compatible OpenCV/NumPy versions, pass RGB `uint8` arrays of shape `(height,width,3)` for one isolated sign (1–300 frames) to `features_from_rgb_frames(frames)` from `karsl_cv_adapter.py`. For a video file, `read_rgb_video(path)` yields the frames first. Send the resulting `(32,150)` feature array and the exact preprocessing version to the ML process; there call `predictor.predict_features(features, preprocessing_version=PREPROCESSING_VERSION)`. The two environments can communicate through the Backend contract rather than forcing incompatible CV and ML dependency locks into one process. `predict_rgb_frames(predictor, frames)` is a convenience wrapper only when a combined compatible environment has been verified. The adapter extracts Holistic landmarks and passes the result through the frozen preprocessor; it does not accept the other SIGNOVA tracks' `(T,126)` keypoints directly.

The CV video/Pose pilot is a research consistency check, not measured real-camera accuracy. NLP can consume the explicitly marked candidate during integration testing; it must preserve the `confirmed=false` marker. See `../reports/backend_contract_502.md` and `../reports/cv_nlp_development_contract.md` for exact fields and limitations.

# Isharah-500 backend integration contract

This guide defines the Python integration boundary for the trained SI
checkpoint. It does not define an HTTP API.

## Initialization

Create one [`IsharahPredictor`](../src/isharah_inference.py) during backend
application startup. Keep that object alive and reuse it for requests. Model
weights, the SI training vocabulary, and the serialized preprocessing settings
are loaded once from the checkpoint.

```python
from pathlib import Path

from isharah_inference import IsharahPredictor, load_predictor

CHECKPOINT = Path("path/to/isharah500_si_best.pt")

predictor: IsharahPredictor = load_predictor(
    CHECKPOINT,
    device_name="cuda",  # or "cpu" / "auto"
)
```

The checkpoint path is supplied by application configuration. The inference
module contains no project-specific absolute path.

When the backend process has already activated the Isharah-500 virtual
environment, use `python` in local commands instead of embedding a machine
specific interpreter path. The verified Windows interpreter is
`E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe`.

The CV team owns the upstream conversion from camera/video frames to tracked
dual-hand keypoints. It must provide a real `numpy.float32` array with shape
`(T, 2, 63)`, using the same two-hand slot convention used during training.
`IsharahPredictor` accepts keypoints only; it is not a video-to-keypoints
adapter and does not accept video files or image frames.

## Input contract

`predictor.predict(keypoints)` accepts one real keypoint sequence:

| Field | Contract |
|---|---|
| Type | `numpy.ndarray` |
| Dtype | `numpy.float32` |
| Shape | `(T, 2, 63)` |
| `T` | Positive number of frames |
| Values | Finite values, with NaNs allowed only for missing detections handled by the saved preprocessing configuration |

The two hand entries contain 63 features each. The predictor rejects wrong
types, shapes, dtypes, empty sequences, infinities, unsupported flattened-hand
configuration, and NaNs when the checkpoint uses `nan_strategy="none"`.

NaNs are not replaced by backend code with a new policy. They follow the
checkpoint's saved preprocessing contract, such as temporal interpolation or
zero filling. The original hand-validity mask is retained and passed to the
model. If preprocessing still produces a nonfinite value, prediction fails
with `ValueError`.

## Output contract

`predict` returns an immutable [`Prediction`](../src/isharah_inference.py)
object with these fields:

| Field | Type | Meaning |
|---|---|---|
| `gloss_tokens` | `tuple[str, ...]` | Greedy CTC decoded gloss sequence |
| `gloss_text` | `str` | Tokens joined with one space; empty when no token is decoded |
| `input_length` | `int` | Number of input frames after validation |
| `valid_hand_coverage` | `float` | Fraction of frame/hand entries marked valid before preprocessing |
| `device` | `str` | Device used for inference |
| `preprocessing_config` | `dict[str, Any]` | Exact serialized preprocessing settings restored from the checkpoint |

For JSON or backend serialization, call `prediction.to_dict()`. The resulting
mapping has the same fields, with `gloss_tokens` represented as a JSON list.
No confidence score is returned because the current contract does not define a
calibrated confidence measure.

## Minimal request flow

```python
import numpy as np


def recognize(predictor: IsharahPredictor, keypoints: np.ndarray) -> dict:
    prediction = predictor.predict(keypoints)
    return prediction.to_dict()


keypoints = np.load("incoming_sequence.npy", allow_pickle=False)
response = recognize(predictor, keypoints)
```

The same predictor instance can serve repeated calls. The model is switched to
evaluation mode at load time and each call runs under `torch.inference_mode()`.

## Import isolation

Import the Isharah-500 source directory explicitly in the backend environment,
or package this module under its own application namespace. The current
implementation imports `isharah_dataset`, `isharah_ctc`, and
`isharah_training` from `isharah500/src`; do not place another directory with
same-named modules ahead of it on `sys.path`. The separate
`ml/sign_language` module currently uses different module names, but explicit
source-path setup avoids future collisions.

## Error handling

- `TypeError`: input is not a NumPy array.
- `ValueError`: input shape, dtype, length, values, or saved contract is invalid.
- `RuntimeError`: an explicitly requested CUDA device is unavailable.
- Checkpoint loading errors: propagated from the checkpoint loader, including
  missing or incompatible trusted checkpoint files.

Callers should treat these as failed prediction requests and log the exception
with request context. They should not replace invalid input with synthetic
keypoints or retry with a different preprocessing configuration.

## Local smoke check

From the project root, the existing Windows virtual environment can run one
real SI dev keypoint file through the CLI:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" .\isharah500\src\isharah_inference.py --checkpoint ".\isharah500\checkpoints\si_hand_centric_resume_15ep\isharah500_si_best.pt" --keypoints "E:\DEPI Project\SIGNOVA_ESL\data\processed\isharah500\hand_keypoints_tracked\dev\02_0001.npy" --device cuda
```

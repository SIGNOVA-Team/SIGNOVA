# SIGNOVA — Computer Vision (CV) Integration Guide

This guide defines the exact technical interface and integration contracts between the **Computer Vision Team** and the **Machine Learning & Deep Learning Core** for the SIGNOVA project.

---

## 1. Landmark & Feature Contract

The continuous sign language models (ArabSign ArSL and How2Sign ASL) operate on temporal sequences of hand keypoints extracted from video frames.

### Tensor Shape & Specification
- **Per-Frame Keypoint Dimension:** `126` float32 values.
  - Left Hand: 21 landmarks $\times$ 3 coordinates $(x, y, z) = 63$ floats.
  - Right Hand: 21 landmarks $\times$ 3 coordinates $(x, y, z) = 63$ floats.
  - Concatenated as: `[Left_Hand (63), Right_Hand (63)]`.
- **Temporal Sequence Shape:** `(T, 126)`, where $T$ is the number of frames (recommended $30 \le T \le 120$).
- **Coordinate Space:** Normalized $[0.0, 1.0]$ with respect to frame width and height. Origin $(0, 0)$ is top-left.
- **Handling Missing Hands:** When a hand is not detected or occluded, fill its 63 coordinates with `NaN`. The ML preprocessing layer automatically applies linear temporal interpolation to reconstruct trajectory data.

---

## 2. Integration Modes for the CV Team

### Mode A: End-to-End Video File Processing
If the CV team has processed or saved video files (`.mp4`, `.avi`, `.mov`, `.webm`), they can trigger prediction with a single function call:

```python
from ml.sign_language.src.inference import predict_video

# Run full pipeline: Video -> MediaPipe Landmark Extraction -> Model -> NLP Translation
result = predict_video("path/to/user_sign.mp4", language="ar")

print("Recognized Sign Glosses:", result["text"])
print("Spoken Standard Arabic:", result["spoken_arabic"])
print("English Translation:", result["english_translation"])
print("Confidence:", result["confidence"])
```

---

### Mode B: Real-Time Live Webcam / Streaming Feed
For live webcam translation in a desktop app, web browser, or OpenCV loop, use `RealtimeCVStreamProcessor` or `StreamingKeypointBuffer`:

```python
import cv2
from ml.sign_language.src.cv_interface import RealtimeCVStreamProcessor

# Initialize real-time processor for Arabic Sign Language
processor = RealtimeCVStreamProcessor(language="ar", window_size=60, stride=15)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process each BGR frame directly
    prediction = processor.process_frame(frame)
    if prediction is not None:
        # A full gesture window has completed and fired!
        print(f"Recognized: {prediction['spoken_arabic']} ({prediction['confidence']:.2f})")
        cv2.putText(
            frame,
            prediction["spoken_arabic"],
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    cv2.imshow("SIGNOVA Live Sign Translation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

---

### Mode C: Passing Pre-Extracted Keypoints Directly
If the CV team uses custom hand tracking or alternative pose models (e.g. YOLOv8-Pose, MMPose, OpenPose), simply pass the extracted `(T, 126)` numpy array:

```python
import numpy as np
from ml.sign_language.src.inference import predict

# keypoints_array shape: (T, 126)
keypoints_array = np.load("tracked_gestures.npy")

result = predict(keypoints_array, language="ar")
print("Prediction Result:", result)
```

---

## 3. Performance & Latency Targets
- **Inference Latency:** $\sim 3.8 \text{ ms}$ per sequence on GPU (RTX 5060) / $\sim 15 \text{ ms}$ on CPU.
- **Throughput:** $> 250$ sequences/second.
- **Recommended Camera FPS:** 25–30 FPS.

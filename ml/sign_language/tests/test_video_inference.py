"""
Integration test for end-to-end Video Inference (Video -> MediaPipe -> Model).
"""

from pathlib import Path

import cv2
import numpy as np

from ml.sign_language.src.inference import load_model, predict_video


def test_video_inference_pipeline(tmp_path):
    root_dir = Path(__file__).resolve().parents[3]
    checkpoint_path = root_dir / "ml/sign_language/artifacts/arabsign/best_model.pth"
    vocab_path = root_dir / "ml/sign_language/artifacts/arabsign/vocab.json"

    if not checkpoint_path.exists() or not vocab_path.exists():
        return

    # 1. Create a synthetic test video (30 frames, 640x480)
    video_file = tmp_path / "test_sign.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, 30.0, (640, 480))

    for i in range(30):
        # Blank canvas
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a simulated moving skin-toned hand blob
        cv2.circle(frame, (200 + i * 5, 240), 40, (180, 200, 240), -1)
        out.write(frame)
    out.release()

    assert video_file.exists()

    # 2. Test predict_video
    predictor = load_model(checkpoint_path, vocab_path, device="cpu")
    result = predict_video(video_file, predictor=predictor)

    assert "language" in result
    assert result["language"] == "ArSL"
    # Even if no real hand landmarks are resolved in the synthetic blob, the pipeline must handle it gracefully
    assert "success" in result

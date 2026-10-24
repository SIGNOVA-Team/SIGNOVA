"""
Unit tests for the backend-ready inference API in SIGNOVA.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ml.sign_language.src.inference import SignLanguagePredictor, load_model, predict  # noqa: E402
from ml.sign_language.src.model import ContinuousSignBiLSTM  # noqa: E402
from ml.sign_language.src.preprocessing import SignVocabulary  # noqa: E402


def test_predictor_and_api(tmp_path):
    # Setup temporary mock artifacts
    vocab = SignVocabulary(language="ArSL")
    vocab.build_from_texts(["مرحبا بكم", "السلام عليكم"])
    vocab_path = tmp_path / "vocab.json"
    vocab.save(vocab_path)

    model = ContinuousSignBiLSTM(
        input_dim=126,
        hidden_dim=256,
        num_layers=2,
        num_classes=len(vocab),
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": len(vocab),
            "language": "ArSL",
        },
        checkpoint_path,
    )

    # Initialize predictor
    predictor = SignLanguagePredictor(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        device="cpu",
    )

    # Test predict with synthetic keypoints array (T, 126)
    dummy_keypoints = np.random.randn(45, 126).astype(np.float32)
    output = predictor.predict(dummy_keypoints)

    assert output["success"] is True
    assert "text" in output
    assert "tokens" in output
    assert "confidence" in output
    assert "latency_ms" in output
    assert isinstance(output["text"], str)
    assert isinstance(output["tokens"], list)
    assert 0.0 <= output["confidence"] <= 1.0

    # Test module-level convenience functions
    loaded = load_model(checkpoint_path, vocab_path, device="cpu")
    res = predict(dummy_keypoints, predictor=loaded)
    assert res["success"] is True
    assert "text" in res

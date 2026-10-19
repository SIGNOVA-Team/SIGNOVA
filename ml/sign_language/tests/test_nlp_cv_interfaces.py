"""
Tests for NLP and Computer Vision Bridge Interfaces.
"""

from __future__ import annotations

import numpy as np

from ml.sign_language.src.cv_interface import StreamingKeypointBuffer
from ml.sign_language.src.nlp_interface import GlossToTextTranslator, get_translator


def test_nlp_gloss_translation_exact_match():
    translator = get_translator()

    # Test exact match
    res = translator.translate_arabic_glosses(["اسم", "الله"])
    assert res["match_type"] == "exact"
    assert res["spoken_arabic"] == "اسم الله"
    assert res["english_translation"] == "In the name of Allah"
    assert res["confidence_score"] == 1.0


def test_nlp_gloss_translation_fuzzy_match():
    translator = get_translator()

    # Slight variation / extra token
    res = translator.translate_arabic_glosses(["السلام", "عليكم", "رحمة", "الله"])
    assert res["match_type"] in {"exact", "fuzzy"}
    assert "السلام عليكم" in res["spoken_arabic"]
    assert res["confidence_score"] > 0.5


def test_nlp_prompt_generation():
    translator = GlossToTextTranslator()
    prompt_ar = translator.generate_nlp_prompt(["الله", "اكبر"], target_lang="ar")
    assert "الله اكبر" in prompt_ar
    assert "ArSL" in prompt_ar

    prompt_en = translator.generate_nlp_prompt(["THANK", "YOU"], target_lang="en")
    assert "THANK YOU" in prompt_en


def test_cv_streaming_keypoint_buffer():
    buffer = StreamingKeypointBuffer(window_size=10, stride=5, activity_threshold=0.0001)

    # Push stationary/silent frames (zeros)
    for _ in range(9):
        out = buffer.add_keypoint_frame(np.zeros(126, dtype=np.float32))
        assert out is None

    # Push active motion frame
    motion_frame = np.random.randn(126).astype(np.float32)
    out = buffer.add_keypoint_frame(motion_frame)

    assert out is not None
    assert out.shape == (10, 126)

    # Next immediate frame should be None until stride (5) is reached
    out2 = buffer.add_keypoint_frame(motion_frame)
    assert out2 is None

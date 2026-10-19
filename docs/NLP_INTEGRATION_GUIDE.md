# SIGNOVA — Natural Language Processing (NLP) Integration Guide

This guide defines the integration contracts and translation workflows between the **Machine Learning Core** and the **NLP / Text Translation Team** for the SIGNOVA project.

---

## 1. The Linguistic Challenge: Sign Glosses vs Spoken Language

Sign languages are natural languages with their own independent grammar, syntax, and morphology:
- **ArabSign (ArSL):** Signs in Arabic Sign Language follow a topic-prominent, non-inflected structure without grammatical particles (conjunctions, prepositions, case endings).
  - *ArSL Gloss Output:* `السلام عليكم رحمة الله بركة`
  - *Standard Spoken Arabic (MSA):* `السلام عليكم ورحمة الله وبركاته`
  - *ArSL Gloss Output:* `شكرا انتم`
  - *Standard Spoken Arabic (MSA):* `شكراً لكم / شكراً جزيلاً لكم`
- **How2Sign (ASL):** Continuous American Sign Language glosses/transcriptions aligned with English instructional videos.

The ML core outputs **continuous gloss sequences**, and the NLP pipeline's role is **Gloss-to-Text Translation (G2T)**, grammatical normalization, and Modern Standard Arabic generation.

---

## 2. Using the ML-NLP Bridge Interface

The ML core provides [`GlossToTextTranslator`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/ml/sign_language/src/nlp_interface.py) for direct usage by the NLP team:

```python
from ml.sign_language.src.nlp_interface import get_translator

translator = get_translator()

# 1. Exact or Fuzzy Translation Lookup (ArSL -> MSA)
recognized_glosses = ["السلام", "عليكم", "رحمة", "الله", "بركة"]
result = translator.translate_arabic_glosses(recognized_glosses)

print("Spoken Arabic (MSA):", result["spoken_arabic"])
# -> "السلام عليكم ورحمة الله وبركاته"
print("English Translation:", result["english_translation"])
# -> "Peace be upon you, and Allah's mercy and blessings"
print("Match Confidence:", result["confidence_score"])
# -> 1.0
print("Domain Category:", result["domain_category"])
# -> "Greeting"
```

---

## 3. Generative NLP / LLM Integration (Open-Vocabulary Translation)

For sentences beyond the 50 canonical ArabSign mapped sentences, the NLP team can feed the ML-predicted glosses into neural translation or generative language models (e.g., **mT5**, **CAMeL-Tools**, **MarianMT**, or **LLaMA/GPT**):

```python
from ml.sign_language.src.nlp_interface import get_translator

translator = get_translator()

# Generate structured prompt for Seq2Seq / LLM models
prompt = translator.generate_nlp_prompt(["الله", "اكبر", "جميع", "الناس"], target_lang="ar")
print(prompt)
# Output:
# حول متتالية إشارات لغة الإشارة العربية (ArSL Glosses) التالية إلى جملة عربية فصيحة ومترابطة نحوياً ودلالياً:
# الإشارات: الله اكبر جميع الناس
# الجملة الفصيحة:
```

---

## 4. Metadata and Dataset Files for the NLP Team

The NLP team has access to the following curated datasets in the repository:
1. **ArabSign Dual-Tier Mapping:**
   - Path: [`data/arabsign/metadata/arabsign_sentences_mapping.csv`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/data/arabsign/metadata/arabsign_sentences_mapping.csv)
   - Columns: `sentence_id`, `arsl_gloss_structure`, `spoken_arabic_standard`, `english_translation`, `domain_category`.
   - Purpose: Ideal for training/fine-tuning a small gloss-to-text Seq2Seq model or dictionary-based translation.
2. **How2Sign Continuous ASL Transcriptions:**
   - Path: [`data/how2sign/metadata/how2sign_canonical_metadata.csv`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/data/how2sign/metadata/how2sign_canonical_metadata.csv)
   - 35,263 continuous sentences with English translations, start/end timestamps, and video references.
3. **ArabSign Sign Vocabulary:**
   - Path: [`ml/sign_language/artifacts/arabsign/vocab.json`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/ml/sign_language/artifacts/arabsign/vocab.json)
   - 98 unique sign glosses mapped to integer IDs for tokenization.

# SIGNOVA — Sign Language & Multimodal Assistive Communication

[![CI/CD Pipeline](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml/badge.svg)](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-126%20passed-brightgreen.svg)](tests/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](LICENSE)

> **SIGNOVA** bridges Arabic & American Sign Language and spoken language through Computer Vision, Deep Learning, and NLP — making assistive communication accessible for Arabic-speaking deaf and hard-of-hearing communities.

---

## What is SIGNOVA?

SIGNOVA is an end-to-end research platform for sign language recognition and translation. It takes a live video stream or a recorded video clip, extracts hand and body landmarks, runs a trained deep learning model to decode sign language, and outputs natural spoken Arabic (or English) text.

```
📷 Camera / Video
       │
       ▼
🖐️  Computer Vision   →  MediaPipe hand & body landmark extraction
       │                  (126-D keypoints per frame)
       ▼
🧠  ML Recognition    →  BiLSTM / BiGRU + CTC decoder
       │                  (ArabSign · Isharah-500 · How2Sign · KArSL-502)
       ▼
📝  NLP Translation   →  Gloss → Spoken Modern Standard Arabic + English
       │                  (fuzzy match · Seq2Seq prompting)
       ▼
🔊  Output            →  Text display · Text-to-Speech · REST API
```

---

## The 4 ML Tracks at a Glance

| Track | Dataset | Sign Language | Task | Model | Size |
|:------|:--------|:-------------|:-----|:------|:-----|
| **1** | [ArabSign](https://hamzah-luqman.github.io/ArabSign/) | Arabic (ArSL) | Continuous recognition + NLP | BiLSTM + CTC | 9,335 samples · 50 sentences |
| **2** | [Isharah-500](isharah500/) | Arabic (ArSL) | Continuous, signer-independent | BiGRU + CTC | 7,500 samples · 15 signers |
| **3** | [How2Sign](https://how2sign.github.io/) | American (ASL) | Continuous multimodal benchmark | BiLSTM + CTC | 35,263 sentences · 56 hours |
| **4** | [KArSL-502](karsl/) | Arabic (ArSL) | Isolated word recognition | 1D CNN + BiGRU | 75,095 samples · 502 classes |

> [!WARNING]
> **ADR-001 — Linguistic Separation Rule:** ArabSign, Isharah-500, and KArSL-502 represent Arabic Sign Language and must **never** be relabeled or confused with Egyptian Sign Language (ESL). Each track has its own vocabulary, tokenizer, loss function, and evaluation. See [`docs/decisions/ADR_001_p2_datasets_evaluation.md`](docs/decisions/ADR_001_p2_datasets_evaluation.md).

---

## Repository Structure

```
SIGNOVA/
│
├── 📄 README.md                        ← You are here
├── 📄 pyproject.toml                   ← Ruff · Pytest · Bandit config
├── 📄 requirements.txt                 ← Core runtime deps (pandas, numpy)
├── 📄 requirements-dev.txt             ← Dev/CI deps (pytest, ruff, bandit)
│
├── 📁 ml/sign_language/                ← Track 1 & 3: ArabSign + How2Sign
│   ├── src/
│   │   ├── model.py                    ← ContinuousSignBiLSTM architecture
│   │   ├── train.py                    ← CTC training loop
│   │   ├── inference.py                ← predict() + predict_video() API
│   │   ├── cv_interface.py             ← Live webcam streaming buffer
│   │   ├── nlp_interface.py            ← Gloss → Arabic/English translator
│   │   ├── dataset.py                  ← PyTorch dataset loader
│   │   ├── preprocessing.py            ← Keypoint normalization & vocab
│   │   ├── evaluate.py                 ← WER · CER · Token Accuracy
│   │   └── video_extractor.py          ← MediaPipe landmark extractor
│   ├── artifacts/
│   │   ├── arabsign/best_model.pth     ← Trained ArabSign weights
│   │   └── how2sign/best_model.pth     ← Trained How2Sign weights
│   ├── configs/config.yaml             ← Hyperparameters
│   ├── notebooks/                      ← Training & inspection notebooks
│   ├── tests/                          ← 18 unit & integration tests
│   └── README.md
│
├── 📁 isharah500/                      ← Track 2: Signer-Independent ArSL
│   ├── src/                            ← BiGRU+CTC · hand-centric features
│   ├── notebooks/                      ← Training · eval · error analysis
│   ├── reports/                        ← SI benchmark · blank calibration
│   ├── tests/                          ← 70+ unit tests
│   └── README.md
│
├── 📁 karsl/                           ← Track 4: 502-class isolated ArSL
│   ├── ml/
│   │   ├── src/preprocessing.py        ← 32×150 pose+hand features
│   │   ├── artifacts/
│   │   │   ├── cnn/best_cnn_502_pass_v1.pt   ← Trained CNN weights
│   │   │   └── gru/best_gru_502_pass_v1.pt   ← Trained GRU weights
│   │   ├── integration/
│   │   │   ├── karsl_cv_adapter.py     ← CV → feature extraction bridge
│   │   │   └── research_preview.py     ← Backend integration stub
│   │   ├── training/                   ← Reproducible training scripts
│   │   ├── tests/                      ← 10 unit & release tests
│   │   └── README.md
│   ├── data/                           ← KArSL metadata, splits, QC
│   ├── scripts/                        ← Dataset validation pipeline
│   ├── tests/                          ← Dataset integrity tests
│   └── README.md
│
├── 📁 wlasl/                           ← WLASL-2000 ASL baseline (reference)
│   ├── data/                           ← Splits & metadata
│   ├── docs/
│   └── README.md
│
├── 📁 data/                            ← Canonical dataset metadata & CSVs
│   ├── arabsign/
│   │   ├── metadata/
│   │   │   ├── arabsign_canonical_metadata.csv   ← 9,335 records
│   │   │   └── arabsign_sentences_mapping.csv    ← 50 gloss→Arabic mappings
│   │   └── splits/                     ← Signer-independent train/val/test
│   ├── how2sign/
│   │   ├── metadata/                   ← 35,263 sentence records
│   │   └── splits/
│   └── reports/                        ← Download manifests · QC reports
│
├── 📁 docs/                            ← Integration guides & decisions
│   ├── CV_INTEGRATION_GUIDE.md         ← For the Computer Vision sub-team
│   ├── NLP_INTEGRATION_GUIDE.md        ← For the NLP sub-team
│   ├── dataset_card_arabsign.md        ← ArabSign dataset card
│   ├── dataset_card_how2sign.md        ← How2Sign dataset card
│   ├── ML_CLOUD_LINKS_AND_STREAMING.md ← Cloud training guide (Colab/Kaggle)
│   └── decisions/
│       └── ADR_001_p2_datasets_evaluation.md   ← Linguistic guardrails
│
├── 📁 notebooks/                       ← Root-level cloud training notebooks
│   ├── cloud_ml_signova_colab.ipynb    ← ArabSign+How2Sign Colab notebook
│   ├── cloud_ml_starter.ipynb          ← Quick-start cloud notebook
│   └── train_isharah500_colab.ipynb    ← Isharah-500 Colab notebook
│
├── 📁 scripts/                         ← Automation & CI scripts
│   ├── ci/
│   │   └── check_project_quality.py   ← 5-gate quality checker
│   ├── data_pipeline/                  ← Metadata generation scripts
│   └── export_p2_summary.py           ← Dataset stats exporter
│
├── 📁 tests/                           ← Root-level cross-dataset tests
│   └── test_p2_datasets.py            ← Schema & integrity (7 tests)
│
└── 📁 .github/workflows/
    ├── ci.yml                          ← Main CI: lint · security · tests
    └── karsl-ml.yml                    ← KArSL smoke tests (isolated env)
```

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/SIGNOVA-Team/SIGNOVA.git
cd SIGNOVA

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only
```

### 2. Run All 126 Tests

```bash
pytest
```

Runs all unit and integration tests across all 4 tracks in one command.

### 3. Run Quality Gate (5/5 Checks)

```bash
python scripts/ci/check_project_quality.py
```

Validates: secrets scan · git hygiene · notebook health · ADR-001 linguistic rules · core deliverables.

### 4. Sign Language Video Inference (ArabSign → Arabic Text)

```python
from ml.sign_language.src.inference import predict_video

result = predict_video("path/to/sign_video.mp4", language="ar")

print("Glosses :", result["text"])
print("Arabic  :", result["spoken_arabic"])
print("English :", result["english_translation"])
print("Confidence:", result["confidence"])
```

### 5. Live Webcam Streaming

```python
import cv2
from ml.sign_language.src.cv_interface import RealtimeCVStreamProcessor

processor = RealtimeCVStreamProcessor(language="ar", window_size=60, stride=15)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    prediction = processor.process_frame(frame)
    if prediction:
        print(f"[SIGNOVA] {prediction['spoken_arabic']}  ({prediction['confidence']:.2f})")
    cv2.imshow("SIGNOVA Live", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### 6. KArSL-502 Isolated Sign Prediction

```python
from karsl.ml.scripts.signova_predict_502 import KArSL502Predictor
import numpy as np

predictor = KArSL502Predictor()
# features: numpy array of shape (32, 150) — 32 frames × 150 pose+hand features
features = np.zeros((32, 150), dtype=np.float32)
result = predictor.predict(features)
print(result["top_k"])   # top-3 Arabic sign labels with scores
```

---

## Integration Guides for Sub-Teams

| Team | Guide | What it covers |
|:-----|:------|:---------------|
| **Computer Vision** | [`docs/CV_INTEGRATION_GUIDE.md`](docs/CV_INTEGRATION_GUIDE.md) | `RealtimeCVStreamProcessor` · `StreamingKeypointBuffer` · `extract_keypoints_from_video()` |
| **NLP** | [`docs/NLP_INTEGRATION_GUIDE.md`](docs/NLP_INTEGRATION_GUIDE.md) | `GlossToTextTranslator` · fuzzy matching · LLM prompt generation |
| **Cloud Training** | [`docs/ML_CLOUD_LINKS_AND_STREAMING.md`](docs/ML_CLOUD_LINKS_AND_STREAMING.md) | Google Colab / Kaggle setup · remote streaming strategy |
| **Backend** | [`karsl/ml/integration/research_preview.py`](karsl/ml/integration/research_preview.py) | KArSL research candidate contract for backend integration |

---

## CI/CD Pipeline

Every push to any branch runs the full automated pipeline:

| Job | Tool | What it checks |
|:----|:-----|:---------------|
| 🎨 Lint & Formatting | Ruff | PEP8 · imports · style |
| 🛡️ Security Audit | Bandit | Vulnerability scan |
| 🔍 Domain Guardrails | Custom | ADR-001 · secrets · notebook JSON |
| 🧪 Tests & Datasets | Pytest | 126 tests across all 4 tracks |
| 🏁 CI Pipeline Gate | Aggregate | All 4 must pass |

---

## References

- **ArabSign** — Hamzah Luqman, *"ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition"*, IEEE FG 2023. [[Website]](https://hamzah-luqman.github.io/ArabSign/)
- **How2Sign** — Amanda Duarte et al., *"How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language"*, CVPR 2021. [[Website]](https://how2sign.github.io/)
- **Isharah-500** — SIGNOVA Continuous Sign Language Benchmark, 2026.
- **KArSL** — Sidig et al., *"KArSL: Arabic Sign Language Dataset"*, ACM TALLIP 2021.
- **WLASL** — Li et al., *"Word-level Deep Sign Language Recognition from Video"*, WACV 2020.

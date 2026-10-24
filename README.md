# SIGNOVA — Sign Language & Multimodal Assistive Communication

[![CI/CD Pipeline](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml/badge.svg)](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml)
[![CI - Test Suite](https://img.shields.io/badge/tests-116%20passed-brightgreen.svg)](tests/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](LICENSE)

> **SIGNOVA** is an end-to-end assistive communication platform connecting sign language, speech, and text for Arabic and English users.
> **Mission Principle:** *Egypt-First*. Egyptian Sign Language (ESL) and Arabic are the primary focus, with American Sign Language (ASL) and English providing secondary multimodal benchmarks.

---

## 1. Project Overview & Multi-Track Architecture

SIGNOVA combines state-of-the-art Computer Vision (CV), Deep Learning (BiLSTM/BiGRU + CTC), and Natural Language Processing (NLP) into a decoupled, reproducible architecture:

```
[ Video / Camera Stream ]
           │
           ▼
[ Computer Vision Layer ] ────────► MediaPipe / OpenPose Hand Landmark Extraction
           │                        (126-D normalized keypoints: T × 126)
           ▼
[ Machine Learning Core ] ────────► Continuous Sign Recognition (BiLSTM/BiGRU + CTC)
           │                        • ArabSign (Continuous ArSL)
           │                        • Isharah-500 (Signer-Independent ArSL)
           │                        • How2Sign (Continuous ASL)
           ▼
[ NLP Bridge Interface ]  ────────► Gloss-to-Text Translation (ArSL Gloss ➔ Spoken MSA)
           │                        • Exact & Levenshtein Fuzzy Lookup
           │                        • Generative Seq2Seq / LLM Conditioning
           ▼
[ Web & Assistive Output ] ───────► Text Display, Text-to-Speech (TTS), Real-time Feedback
```

---

## 2. The 4 Machine Learning & Dataset Tracks

| Track | Dataset | Language | Task Type | Architecture | Scope & Benchmark |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track 1** | **ArabSign** | Arabic Sign Language (ArSL) | Continuous Sign Recognition + Dual-Tier NLP | BiLSTM + CTC (2-layer, 256 hidden) | **9,335 samples**, 50 sentences, 98 unique glosses. Zero-leakage signer-independent split (Train: S01–04, Val: S05, Test: S06). Dual-tier mapping to Modern Standard Arabic. |
| **Track 2** | **Isharah-500** | Arabic Sign Language (ArSL) | Continuous Signer-Independent (SI) | BiGRU + CTC (Hand-centric normalization) | **7,500 samples**, 15 signers. Hand-tracked keypoints `(T, 2, 63)`. SI Dev & Test benchmark with blank calibration and error analysis. |
| **Track 3** | **How2Sign** | American Sign Language (ASL) | Continuous Multimodal Benchmark | BiLSTM + CTC (2-layer, 256 hidden) | **35,263 continuous sentences (56 hours)**, 11 signers. Official Train/Val/Test partitions. Multimodal reference (RGB, Depth, Skeletons). |
| **Track 4** | **KArSL-502** | Arabic Sign Language (ArSL) | Isolated Sign Recognition | Deep Feature Classifier / 3D CNN | **75,095 verified samples** across 500 isolated classes. 3-fold signer-independent evaluation protocol. |

> [!WARNING]
> **LINGUISTIC SEPARATION RULE (ADR-001):**
> ArabSign and Isharah-500 represent Pan-Arab/Saudi Sign Language and must never be relabeled as Egyptian Sign Language (ESL). Each dataset maintains independent vocabulary, tokenizer, loss computation, and evaluation metrics. Refer to [ADR-001](docs/decisions/ADR_001_p2_datasets_evaluation.md).

---

## 3. Computer Vision & NLP Integration Contracts

To facilitate seamless collaboration across sub-teams, SIGNOVA provides production-ready integration interfaces:

### Computer Vision (CV) Integration
- **Module:** [`ml/sign_language/src/cv_interface.py`](ml/sign_language/src/cv_interface.py)
- **Features:**
  - `StreamingKeypointBuffer`: Sliding temporal window buffer (`window_size=60`, `stride=15`) with motion energy thresholding (skips idle/resting hands).
  - `RealtimeCVStreamProcessor`: Directly ingests OpenCV camera frames (`cv2.VideoCapture(0)`) $\rightarrow$ extracts landmarks $\rightarrow$ triggers ML inference.
  - `extract_keypoints_from_video()`: Batch extraction from `.mp4`, `.avi`, `.mov`.
- **Full Guide:** [`docs/CV_INTEGRATION_GUIDE.md`](docs/CV_INTEGRATION_GUIDE.md)

### Natural Language Processing (NLP) Integration
- **Module:** [`ml/sign_language/src/nlp_interface.py`](ml/sign_language/src/nlp_interface.py)
- **Features:**
  - `GlossToTextTranslator`: Translates ArSL continuous gloss sequences (e.g. `['السلام', 'عليكم', 'رحمة', 'الله', 'بركة']`) into grammatical Modern Standard Arabic (`السلام عليكم ورحمة الله وبركاته`) and English.
  - Fuzzy Levenshtein matching for OCR/CTC error resilience.
  - Generative prompt generation for Seq2Seq / LLM models (mT5, CAMeL-Tools, LLaMA).
  - Metadata: [`data/arabsign/metadata/arabsign_sentences_mapping.csv`](data/arabsign/metadata/arabsign_sentences_mapping.csv).
- **Full Guide:** [`docs/NLP_INTEGRATION_GUIDE.md`](docs/NLP_INTEGRATION_GUIDE.md)

---

## 4. Repository Structure

```
SIGNOVA/
├── README.md                                # Root documentation & architecture
├── pyproject.toml                           # Modern project config (Ruff, Pytest, Bandit)
├── requirements.txt                         # Runtime dependencies
├── requirements-dev.txt                     # CI, linting & test dependencies
├── data/
│   ├── arabsign/                            # ArabSign metadata, canonical CSVs & splits
│   │   ├── metadata/
│   │   │   ├── arabsign_canonical_metadata.csv  # 9,335 canonical ArabSign records
│   │   │   └── arabsign_sentences_mapping.csv   # 50 ArSL gloss & standard Arabic mappings
│   │   └── splits/                              # Signer-independent & signer-dependent manifests
│   ├── how2sign/                            # How2Sign metadata, canonical CSVs & splits
│   └── reports/                             # Dataset sources, download manifests & QC reports
├── docs/
│   ├── CV_INTEGRATION_GUIDE.md              # Complete integration guide for Computer Vision team
│   ├── NLP_INTEGRATION_GUIDE.md             # Complete integration guide for NLP team
│   ├── dataset_card_arabsign.md             # Dataset card for ArabSign
│   ├── dataset_card_how2sign.md             # Dataset card for How2Sign
│   └── decisions/
│       └── ADR_001_p2_datasets_evaluation.md# Linguistic guardrails & architecture decisions
├── ml/sign_language/                        # Continuous ArabSign & How2Sign ML Core
│   ├── README.md                            # ML module guide & architecture
│   ├── artifacts/                           # Saved checkpoints & vocabularies
│   │   ├── arabsign/best_model.pth          # Trained ArabSign PyTorch weights
│   │   └── how2sign/best_model.pth          # Trained How2Sign PyTorch weights
│   ├── configs/config.yaml                  # Hyperparameters & model configurations
│   ├── manifests/                           # Zero-download remote manifests
│   ├── notebooks/                           # Jupyter notebooks (data inspection, training, eval)
│   ├── src/
│   │   ├── cv_interface.py                  # Real-time sliding window buffer & camera streamer
│   │   ├── nlp_interface.py                 # Gloss-to-Text translation & prompt generator
│   │   ├── inference.py                     # Unified predictor (predict, predict_video)
│   │   ├── model.py                         # ContinuousSignBiLSTM architecture
│   │   ├── train.py                         # CTC training and validation loops
│   │   ├── evaluate.py                      # WER, CER, Token Accuracy & greedy decoders
│   │   ├── dataset.py                       # SignLanguageDataset PyTorch loader
│   │   └── video_extractor.py               # MediaPipe landmark extractor
│   └── tests/                               # ML unit & integration tests (18 tests)
├── isharah500/                              # Isharah-500 Signer-Independent ML Module
│   ├── README.md                            # Isharah-500 baseline documentation
│   ├── docs/                                # Backend integration contract & artifact manifest
│   ├── metadata/                            # SI and US CSV manifests
│   ├── notebooks/                           # Inspection, training, experiments & eval notebooks
│   ├── reports/                             # Error analysis, blank calibration & test evaluations
│   ├── src/                                 # BiGRU+CTC, feature normalization, training & inference
│   └── tests/                               # Isharah-500 test suite
├── karsl/                                   # KArSL-502 500-class isolated sign benchmark
├── wlasl/                                   # WLASL-2000 ASL isolated sign baseline
├── scripts/
│   └── ci/
│       └── check_project_quality.py         # Automated project quality gate (5/5 checks)
└── tests/
    └── test_p2_datasets.py                  # Cross-dataset integrity & schema tests
```

---

## 5. Quickstart & Command Center

### Installation
```bash
git clone https://github.com/SIGNOVA-Team/SIGNOVA.git
cd SIGNOVA

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 1. Run All Tests Across All Modules (116+ Tests)
```bash
pytest
```
*Executes all unit tests simultaneously across root tests, ArabSign/How2Sign, Isharah-500, and KARSL.*

### 2. Run Project Quality Gate
```bash
python scripts/ci/check_project_quality.py
```
*Validates secrets scan, git hygiene, binary avoidance, notebook health, and ADR-001 linguistic separation.*

### 3. Run Live Video Prediction (ArabSign ArSL ➔ Spoken Arabic)
```python
from ml.sign_language.src.inference import predict_video

# Run end-to-end video inference
result = predict_video("path/to/sign_video.mp4", language="ar")

print("Recognized Sign Glosses:", result["text"])
print("Spoken Standard Arabic :", result["spoken_arabic"])
print("English Translation    :", result["english_translation"])
print("Confidence Score       :", result["confidence"])
print("Inference Latency      :", result["latency_ms"], "ms")
```

### 4. Run Real-Time Webcam Streaming (Computer Vision Loop)
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
        print(f"Recognized: {prediction['spoken_arabic']} ({prediction['confidence']:.2f})")

    cv2.imshow("SIGNOVA Live Sign Translation", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

### 5. Run Code Quality & Formatting Checks
```bash
python -m ruff check .
python -m ruff format --check .
```

---

## 6. References & Citations
- **How2Sign:** Amanda Duarte et al., *"How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language"*, CVPR 2021.
- **ArabSign:** Hamzah Luqman, *"ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition"*, IEEE FG 2023.
- **Isharah-500:** SIGNOVA Continuous Sign Language Benchmark, 2026.
- **KArSL:** Sidig et al., *"KArSL: Arabic Sign Language Dataset"*, 2021.

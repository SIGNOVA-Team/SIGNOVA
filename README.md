# SIGNOVA — Sign Language & Multimodal Assistive Communication

[![CI/CD Pipeline](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml/badge.svg)](https://github.com/SIGNOVA-Team/SIGNOVA/actions/workflows/ci.yml)
[![CI - Test Suite](https://img.shields.io/badge/tests-8%20passed-brightgreen.svg)](tests/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Security: Bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License: CC-BY-NC-4.0](https://img.shields.io/badge/License-CC--BY--NC--4.0-lightgrey.svg)](LICENSE)

> **SIGNOVA** is an assistive communication system bridging sign language, speech, and text for Egyptian Arabic and English users.
> **Mission Principle:** *Egypt-First*. Egyptian Sign Language (ESL) and Egyptian Arabic are the primary communication pair. ASL and English are secondary tracks.

---

## 1. Project Overview & Boundaries

SIGNOVA is designed as an assistive communication prototype connecting sign language, speech, and text.
- **In Scope:** Reproducible isolated-sign baselines, validated data pipelines, calibrated uncertainty handling, and continuous sign language research benchmarks.
- **Out of Scope (v0.1):** Claiming interpreter-level reliability, full continuous translation without validated continuous models, and relabeling Gulf/Saudi sign language as Egyptian Sign Language.

### Dataset Priority Tiers:
- **P0 — Egyptian Sign Language (ESL):** Primary isolated-sign source (Mendeley Data v3).
- **P1 — WLASL:** Secondary ASL isolated-sign baseline (Official GitHub).
- **P2 — How2Sign:** Future continuous ASL research reference (~80h multimodal).
- **P2 — ArabSign:** Continuous Arabic Sign Language (ArSL) research reference (9,335 samples, 50 sentences).
- **Benchmark — KArSL-502:** Arabic Sign Language (ArSL) 500-class isolated-sign benchmark & keypoints dataset ([KArSL documentation](karsl/README.md)).

---

## 2. P2 Datasets Workstream (How2Sign & ArabSign)

This repository includes the complete quality control, canonical schemas, split manifests, and automated test suite for both **Priority 2 (P2)** continuous datasets:

### Summary Table

| Metric | P2 — How2Sign | P2 — ArabSign |
| :--- | :--- | :--- |
| **Language** | American Sign Language (ASL) | Arabic Sign Language (ArSL - Saudi / Pan-Arab) |
| **Type** | Continuous Instructional Sentences | Continuous Sentences (Repetitions) |
| **Total Samples** | 35,263 continuous sentences | 9,335 samples |
| **Signers** | 11 fluent deaf signers | 6 male deaf signers (21–30 years) |
| **Duration** | $\approx 80$ hours | 10 hours, 13 minutes |
| **Modalities** | RGB Front/Side, Depth, OpenPose 2D/3D, Text | RGB Color, Depth, 25 Kinect V2 3D Skeletons |
| **Official Splits** | Train: 31,165 \| Val: 1,741 \| Test: 2,357 | 6-fold signer-dependent splits |
| **Zero-Leakage Split** | Documented signer overlap (topic-split) | **Train: 6,323 (S01–04) \| Val: 1,500 (S05) \| Test: 1,512 (S06)** |
| **Documentation** | [Dataset Card](docs/dataset_card_how2sign.md) | [Dataset Card](docs/dataset_card_arabsign.md) |

> [!WARNING]
> **CRITICAL LINGUISTIC RULE:**
> ArabSign was recorded at KFUPM (Saudi Arabia) and represents Saudi/Pan-Arab Sign Language (ArSL). **It must never be relabeled as Egyptian Sign Language (ESL)** or mixed into ESL isolated training data. Refer to [ADR-001](docs/decisions/ADR_001_p2_datasets_evaluation.md) for full policy details.

---

## 3. Repository Structure

```
SIGNOVA/
├── README.md                                # Project documentation & guide
├── data/
│   ├── README.md                            # Complete data guide and governance rules
│   ├── arabsign/                            # Continuous Arabic Sign Language (ArSL)
│   │   ├── README.md                        # ArabSign guide, metrics, and splits
│   │   ├── metadata/
│   │   │   ├── arabsign_canonical_metadata.csv  # 9,335 canonical ArabSign records
│   │   │   └── arabsign_sentences_mapping.csv   # 50 ArSL gloss & standard Arabic sentences
│   │   ├── splits/
│   │   │   ├── arabsign_split_manifest_signer_independent.csv  # Zero-leakage benchmark
│   │   │   └── arabsign_split_manifest_signer_dependent.csv    # 6-fold signer splits
│   │   └── raw/                             # Ground truth text & original author splits
│   │       ├── groundTruth.txt
│   │       └── author_splits/
│   ├── how2sign/                            # Continuous American Sign Language (ASL)
│   │   ├── README.md                        # How2Sign guide, metrics, and splits
│   │   ├── metadata/
│   │   │   └── how2sign_canonical_metadata.csv  # 35,263 canonical How2Sign records
│   │   ├── splits/
│   │   │   └── how2sign_split_manifest.csv      # Train/Val/Test official partition
│   │   └── raw/                             # Official text & re-aligned annotations
│   │       ├── how2sign_train.csv
│   │       ├── how2sign_val.csv
│   │       ├── how2sign_test.csv
│   │       └── how2sign_realigned_*.csv
│   └── reports/                             # Cross-dataset registers & QC
│       ├── README.md                        # Data reports & sources guide
│       ├── dataset_sources.csv              # Source register (URLs, licenses, approvals)
│       ├── download_manifest.csv            # Modality sizes, direct links & access gates
│       ├── qc_flags_p2.csv                  # Machine-readable QC flags table
│       └── qc_report_p2.md                  # Comprehensive P2 QC & linguistic audit
├── docs/
│   ├── dataset_card_how2sign.md             # Standard dataset card for How2Sign
│   ├── dataset_card_arabsign.md             # Standard dataset card for ArabSign
│   └── decisions/
│       └── ADR_001_p2_datasets_evaluation.md# Architecture Decision Record for P2
├── karsl/                                   # KArSL-502 dataset engineering & benchmarks
│   ├── README.md                            # KArSL overview, QC, and keypoints benchmark
│   ├── data/                                # Metadata, splits, and QC reports
│   ├── docs/                                # KArSL dataset card and handoff
│   ├── notebooks/                           # Verification notebooks (inventory, QC, keypoints)
│   ├── scripts/                             # Validation scripts
│   └── tests/                               # Automated integrity tests
├── scripts/
│   ├── ci/
│   │   └── check_project_quality.py         # Automated project quality, hygiene & ADR guardrails
│   ├── data_pipeline/
│   │   ├── download_p2_metadata.py          # Metadata & code downloader
│   │   └── generate_canonical_p2_metadata.py# Canonical parser & split builder
│   └── export_p2_summary.py                 # Summary statistics printer
├── tests/
│   └── test_p2_datasets.py                  # Automated test suite (schemas, zero leakage)
├── .github/workflows/ci.yml                 # Production-grade GitHub Actions CI/CD Pipeline
├── pyproject.toml                           # Modern project config (Ruff, Pytest, Bandit)
├── requirements.txt                         # Runtime dependencies
└── requirements-dev.txt                     # Development, QA & CI dependencies
```

---

## 4. Quickstart, Quality Checks & Verification

### Prerequisites
- Python 3.12+
```bash
pip install -r requirements-dev.txt
```

### 1. Automated Test Suite
Run the unified test suite across all verified datasets (P2 and KArSL):
```bash
python -m pytest -v
```

### 2. Project Quality & Domain Guardrails Check
Run the comprehensive quality gate (checks secrets, forbidden binary files, notebook JSON validity, ADR-001 linguistic rules):
```bash
python scripts/ci/check_project_quality.py
```

### 3. Code Quality, Formatting & Security Audit
Run Ruff linting, formatting check, and Bandit security scans:
```bash
# Code style and import sorting
python -m ruff check .

# Code formatting
python -m ruff format --check .

# Security scan
python -m bandit -r scripts karsl/scripts -c pyproject.toml
```

### 4. Displaying Dataset Summaries
Print the verified dataset distributions and split metrics:
```bash
python scripts/export_p2_summary.py
```

### 5. Re-generating Canonical Metadata
To re-process raw annotations into canonical manifests:
```bash
python scripts/data_pipeline/generate_canonical_p2_metadata.py
```

---

## 5. References & Citations
- **How2Sign:** Amanda Duarte et al., *"How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language"*, CVPR 2021.
- **ArabSign:** Hamzah Luqman, *"ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition"*, IEEE FG 2023.
- **SIGNOVA Execution Plan:** v1.1 Dataset & ML Execution Plan.

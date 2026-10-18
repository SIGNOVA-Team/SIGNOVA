# SIGNOVA — Sign Language Recognition & Translation Machine Learning Module

This module provides deep learning pipelines, reproducible preprocessing, training loops, evaluation suites, and an inference API for Continuous Sign Language Recognition and Translation.

It strictly adheres to **ADR-001**, preserving independent linguistic pipelines and separate vocabularies for:
- **ArabSign:** Arabic Sign Language (ArSL)
- **How2Sign:** American Sign Language (ASL)

---

## 📁 Module Directory Structure

```
ml/sign_language/
├── configs/
│   └── config.yaml             # Hyperparameters, data paths, model configuration
├── src/
│   ├── dataset.py              # Modular dataset adapter with 0-leakage split loading
│   ├── data_loader.py          # Dynamic padding, masking, and batch collation
│   ├── preprocessing.py        # NaN interpolation, coordinate normalization, vocabulary
│   ├── model.py                # ContinuousSignBiLSTM & CTC sequence architecture
│   ├── evaluate.py             # CTC greedy decoder, WER, CER, Token Accuracy
│   ├── train.py                # Robust GPU-accelerated training pipeline
│   └── inference.py            # Production-ready backend inference interface
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing & tokenization
│   ├── test_dataset.py         # Unit tests for dataset & collation
│   ├── test_model.py           # Unit tests for forward pass & loss
│   └── test_inference.py       # Unit tests for prediction API contract
├── artifacts/
│   ├── arabsign/               # Checkpoints (best_model.pth) & vocab.json for ArSL
│   └── how2sign/               # Checkpoints (best_model.pth) & vocab.json for ASL
└── reports/
    └── evaluation_report.md    # Detailed empirical evaluation & technical analysis
```

---

## ⚡ Quickstart & Testing

### 1. Run All Unit & Integration Tests
Execute the automated pytest suite (100% test pass rate):
```bash
pytest ml/sign_language/tests -v
```

### 2. Train Models

#### ArabSign (ArSL)
```bash
python ml/sign_language/src/train.py --dataset arabsign --epochs 5 --batch-size 16
```

#### How2Sign (ASL)
```bash
python ml/sign_language/src/train.py --dataset how2sign --epochs 5 --batch-size 16 --max-samples 2000
```

---

## 🔌 Backend / CV Team Integration Guide

Teammates working on the Computer Vision pipeline or Backend API can consume the model with two lines of code:

```python
import numpy as np
from ml.sign_language.src.inference import load_model, predict

# 1. Initialize predictor (automatically selects CUDA if available)
predictor = load_model(
    checkpoint_path="ml/sign_language/artifacts/arabsign/best_model.pth",
    vocab_path="ml/sign_language/artifacts/arabsign/vocab.json"
)

# 2. Predict on hand keypoints (T, 126) or a saved .npy file
features = np.random.randn(60, 126).astype(np.float32)
response = predict(features, predictor=predictor)

print(response)
```

### Response JSON Contract
```json
{
  "success": true,
  "language": "ArSL",
  "text": "مرحبا بكم",
  "tokens": ["مرحبا", "بكم"],
  "token_ids": [4, 18],
  "confidence": 0.8924,
  "latency_ms": 11.2,
  "sequence_length": 60
}
```

---

## 🛡️ Linguistic Guardrails & Compliance

- **No Vocabulary Contamination:** ArSL glosses and ASL words are stored in isolated vocabularies with unique indices.
- **Signer-Independent Protocol:** ArabSign evaluation splits guarantee signers in `train` are disjoint from `val` and `test`.
- **Reproducible Features:** If raw high-volume video archives are not present locally, synthetic feature simulation enables complete CI verification and offline development without breaking builds.

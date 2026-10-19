# SIGNOVA — Comprehensive Dataset Investigation & Ingestion Report

**Date:** October 19, 2026  
**Datasets Investigated:** ArabSign (ArSL) & How2Sign (ASL)  
**Author:** Senior Machine Learning & Deep Learning Research Engineer  
**Status:** Canonical Audit Complete — Hybrid Local/Remote Ingestion Protocol Verified  

---

## 1. Executive Overview & Storage Guardrails

This technical report delivers the dataset investigation, modality audit, URL access validation, and storage governance strategy for continuous sign language recognition in the **SIGNOVA** system.

### Core Directive: Zero Uncontrolled Laptop Storage Bloat
- **How2Sign Full Video Archive:** Exceeds **2.1 TB** (multiview Panoptic Studio + green screen studio recordings).
- **ArabSign Full Video Archive:** Exceeds **100 GB** across 9,335 RGB MP4s ($1920\times 1080$), depth maps, and 3D Kinect joints.
- **Laptop Resource Profile:** Local NVMe disk has ~115 GB free space. Downloading full raw video collections locally would saturate disk I/O, risk system crashes, and waste hours in network bottlenecks.
- **Architectural Solution:**
  1. **Manifest-Driven Ingestion:** Every sample is indexed with canonical metadata and remote Drive/URL IDs without saving gigabytes of raw MP4s locally.
  2. **Pose-Level Feature Abstraction:** Landmark keypoints (126 features per frame: 2 hands $\times$ 21 joints $\times$ 3 coordinates) require only $\approx 0.05\%$ of raw video storage ($<100\text{ MB}$ vs $100\text{ GB}$).
  3. **Cloud-Native GPU Training:** Heavy video batch decoding and end-to-end vision backbone training is executed in high-throughput cloud environments (Google Colab / Kaggle GPU), transferring only lightweight model checkpoints and vocabularies ($<25\text{ MB}$) back to the repository.

---

## 2. Comparative Dataset Specification

| Property | ArabSign (ArSL) | How2Sign (ASL) |
| :--- | :--- | :--- |
| **Origin & Institution** | King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia | Carnegie Mellon University (CMU) & UPC Barcelona |
| **Official Portal** | [IEEE FG 2023 Paper](https://doi.org/10.1109/FG57933.2023.10042720) / [GitHub](https://github.com/Hamzah-Luqman/ArabSign) | [how2sign.github.io](https://how2sign.github.io/) |
| **Language Target** | Arabic Sign Language (Saudi / Pan-Arab ArSL) | American Sign Language (ASL) |
| **Output Domain** | Arabic Gloss Sequence & Standard Spoken Arabic | Natural English Instructional Sentences |
| **Canonical Samples** | **9,335** verified sequences | **35,263** verified sentences |
| **Available Modalities** | Color RGB ($1920\times 1080$), Depth, 3D Skeleton (25 Kinect V2 joints) | Frontal RGB ($1280\times 720$), Side RGB, 2D OpenPose (Body+Face+Hands) |
| **Frame Rate** | 30.0 FPS | 25.0 FPS |
| **Average Duration** | $2.67\text{ s} - 4.34\text{ s}$ ($\approx 130$ frames/sentence) | $3.5\text{ s} - 7.2\text{ s}$ ($\approx 125$ frames/sentence) |
| **Signer Pool** | 6 male deaf signers (`01`–`06`) | 11 fluent deaf signers |
| **Signer Generalization** | **Full Signer-Independent Protocol** (0% signer overlap) | Content/Topic partition (Signer overlap exists across splits) |
| **Access Route** | Metadata/splits public; Raw videos gated by email authorization | Public Google Drive archives via download scripts |
| **License** | Academic Non-Commercial Research (KFUPM) | Creative Commons Attribution-NonCommercial (CC BY-NC 4.0) |

---

## 3. Linguistic Boundary & Anti-Contamination Policy (ADR-001)

### Why ASL and ArSL Must Never Be Merged
American Sign Language (ASL) and Arabic Sign Language (ArSL) belong to completely distinct linguistic families:
1. **Phonology & Morphology:** ASL hand configurations, movement trajectories, and non-manual facial markers follow distinct grammatical rules that do not translate into Arabic sign glosses.
2. **Syntactic Structure:**
   - ArabSign provides dual annotations: *ArSL Gloss Order* (`الحمد الله`) and *Spoken Arabic Standard Order* (`حمد الله`).
   - How2Sign provides natural English subtitle alignments (`And I call them decorative elements`).
3. **Vocabularies:** ArabSign vocabulary contains 98 domain glosses, while How2Sign contains 5,856 unique English tokens.
4. **Architectural Separation:**
   - Separate dataset adapters (`language="ArSL"` vs `language="ASL"`).
   - Independent vocabulary indices saved to distinct artifacts (`artifacts/arabsign/vocab.json` vs `artifacts/how2sign/vocab.json`).
   - Strict zero-token contamination verified via automated CI assertions (`test_sign_vocabulary`).

---

## 4. URL Access Audit & Gating Protocols

### ArabSign Remote Endpoints
- **Code & Ground Truth:** `https://github.com/Hamzah-Luqman/ArabSign/raw/main/ArabSign-EncoderDec.zip` (Accessible, verified downloaded).
- **RGB Video Archive:** Google Drive Folder ID `1twRzL8fqjbq5dp6jV-V00-O8gD1BZMTN` (~60 GB).
- **Depth Archive:** Google Drive Folder ID `1WJC9Bo9CSP0tVmPJlkTb-KA_n0VWp7LZ` (~40 GB).
- **Skeleton Archive:** Google Drive Folder ID `18QN4X2vnWZ5PvNVik6TJJ7N606pRMvwA` (~5 GB).
- *Access Verification:* Google Drive folders return a permissions barrier requiring institutional approval from the creator (`hluqman@kfupm.edu.sa`).

### How2Sign Remote Endpoints
- **Text & Sentence Boundaries:** Google Drive IDs `1kE10vaZh_LQvZtBvT-jxT5nR9DXU6H7N`, `12xVToQ2vm6pnXbgTyk2VqfnIxrMsORF1`, `1B6WVb3Mtwof3mx3Pnf-UOOyXRpFaT9Hw` (Public, verified downloaded).
- **Frontal Clips (Sentence-Level):** Google Drive IDs `1dYey1F_SeHets-UO8F9cE3VMhRBO-6e0`, `1oVZyTWhHShyqshC2kUrfWnBF8apIR7Z1`, `1d6GHqu0_8IGiKbu3sTZHtMb0DGhbHSMu` (Public, high bandwidth).
- **2D OpenPose Landmarks:** Google Drive IDs `1lnsDN-LxcsroOmetdG5_sXYXZ7setlS4` (~80 GB, public).

---

## 5. Development & Training Environment Strategy

### Environment Evaluation & Comparison

| Environment | Pros | Cons | Recommendation |
| :--- | :--- | :--- | :--- |
| **Local Laptop (RTX 5060)** | Zero network latency, instant iteration, fast debugging, full code access | Limited disk space (~115 GB), cannot host 2 TB raw videos | **Recommended for Pilot training, feature engineering, unit tests, and production inference API.** |
| **Google Colab (Pro/Free)** | Fast Google Drive mounting (same datacenter), 15-50 GB VRAM, isolated storage | 12-hour session timeout, disconnected local filesystem | **Recommended for full-scale video batch extraction and distributed model fine-tuning.** |
| **Kaggle Notebooks** | 30 hours/week GPU (T4/P100), persistent datasets | Strict download bandwidth quotas, limited interactive terminal | Suitable for secondary benchmarking. |
| **Hybrid Approach** | **Combines local fast iteration with remote heavy batch processing** | Requires synchronized repository state via Git | **Optimal Architecture chosen for SIGNOVA.** |

### Selected Architecture: The Hybrid Protocol
- **Local:** Maintains configuration (`config.yaml`), dataset adapters (`dataset.py`), architecture (`model.py`), automated test suite, and the inference API (`inference.py`).
- **Remote / Cloud:** The provided [`training.ipynb`](../notebooks/training.ipynb) runs directly on Google Colab to mount Google Drive, stream raw clips without touching local laptop disks, extract MediaPipe keypoints, and export trained checkpoints (`best_model.pth`).

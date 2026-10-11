# SIGNOVA — Quality Control & Research Audit Report: P2 Datasets
**Track:** Machine Learning & Dataset Engineering Workstreams  
**Datasets:** `P2 — How2Sign` & `P2 — ArabSign`  
**Version:** 1.0 (Test-Driven Revision)  
**Status:** Approved for Exploratory Research & Continuous Reference / Gated for Phase 1 Isolated Baselines  

---

## 1. Executive Summary

This report establishes the dataset engineering verification, quality control (QC), and architectural reference audit for the two **Priority 2 (P2)** continuous sign-language datasets designated in the SIGNOVA Execution Plan:
1. **How2Sign** (Continuous American Sign Language — ASL)
2. **ArabSign** (Continuous Arabic Sign Language — ArSL)

### Key Verdicts & Core Working Rules
- **Non-Blocker for Isolated Baseline:** Per SIGNOVA Execution Plan Section 2 & 7, P2 datasets are **exploratory research benchmarks**. They do **not** block the core Phase 1 isolated-sign baselines (`P0 — ESL` on Mendeley Data and `P1 — WLASL`).
- **Strict Linguistic Boundary (ArabSign $\neq$ ESL):** ArabSign was collected at King Fahd University of Petroleum & Minerals (KFUPM) in Saudi Arabia, reflecting Modern Standard Arabic / Gulf-regional Sign Language conventions. **Under no circumstances may ArabSign be relabeled or combined as Egyptian Sign Language (ESL)** without explicit linguistic verification and Deaf community validation.
- **Split & Leakage Audit:**
  - **How2Sign:** Official splits inherit the topic-based partition of the How2 speech dataset. Signers 1, 2, 3, 5, and 8 appear across both train, val, and test splits. Therefore, models evaluated on the official test set evaluate *topic generalization*, not *unseen-signer generalization*.
  - **ArabSign:** Contains 9,335 samples recorded by 6 male signers. While the original author provided 6 signer-dependent folds, SIGNOVA has established a verified **Signer-Independent Split** (Signers 01–04 Train: 6,323 samples; Signer 05 Val: 1,500 samples; Signer 06 Test: 1,512 samples) guaranteeing **0% signer leakage**.

---

## 2. Dataset Verification & Inventory

### 2.1 P2 — How2Sign Audit

| Attribute | Specification | Verified Status |
| :--- | :--- | :--- |
| **Official Source** | [How2Sign Project](https://how2sign.github.io/) / CVPR 2021 | Confirmed |
| **Language Track** | American Sign Language (ASL) — Continuous | Confirmed |
| **License** | Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0) | Research use permitted |
| **Total Duration** | $\approx 80$ hours continuous signing | Confirmed |
| **Total Sentences** | 35,263 continuous sentence clips | **Verified: 35,263** |
| **Split Breakdown** | Train: 31,165 \| Val: 1,741 \| Test: 2,357 | **Verified via official CSVs** |
| **Signer Metadata** | 11 deaf native/fluent signers encoded in filenames (`-1-`, `-2-`, etc.) | Verified |
| **Modalities Available** | Frontal RGB, Side RGB, Depth, 2D/3D OpenPose keypoints, Speech audio, English text | Cataloged in `download_manifest.csv` |

#### Signer Distribution Across Official Splits:
- **Train Signers (8):** `1`, `2`, `3`, `4`, `5`, `8`, `9`, `11`
- **Val Signers (5):** `1`, `2`, `3`, `5`, `8`
- **Test Signers (6):** `1`, `2`, `3`, `5`, `8`, `10`
- **Leakage Finding:** 5 signers (`1`, `2`, `3`, `5`, `8`) appear in all three partitions. This is flagged as `QC-P2-001` (`SIGNER_OVERLAP_ACROSS_SPLITS`). Any future generalization claims to novel signers will require a signer-stratified re-partition.

---

### 2.2 P2 — ArabSign Audit

| Attribute | Specification | Verified Status |
| :--- | :--- | :--- |
| **Official Source** | [Hamzah Luqman / KFUPM](https://hamzah-luqman.github.io/ArabSign/) / IEEE FG 2023 | Confirmed |
| **Language Track** | Arabic Sign Language (ArSL) — Continuous | Confirmed (Non-ESL) |
| **License** | Non-Commercial Academic Research (KFUPM terms) | Permitted for research |
| **Total Duration** | 10 hours, 13 minutes | Confirmed |
| **Total Samples** | Reported: 9,335 samples | **Verified: Exactly 9,335 samples** |
| **Vocabulary & Sentences** | 50 unique sentences, 155 signs in sentences, 95 unique vocabulary signs | **Verified: 50 sentences** |
| **Signers** | 6 male signers (aged 21–30, right-handed, 1 with eyeglasses) | Verified (`01` through `06`) |
| **Modalities** | Color RGB (1920x1080 @ 30fps), Depth, 25 Kinect V2 3D skeleton joints | Cataloged in `download_manifest.csv` |

#### Verified Sample Counts Per Signer:
- **Signer 01:** 1,535 samples
- **Signer 02:** 1,598 samples
- **Signer 03:** 1,640 samples
- **Signer 04:** 1,550 samples
- **Signer 05:** 1,500 samples
- **Signer 06:** 1,512 samples
- **Total Verified:** **9,335 samples** (100% matched)

---

## 3. Linguistic Audit: ArabSign vs Egyptian Sign Language (ESL)

A critical failure mode in multi-lingual sign-language engineering is assuming that regional sign languages in the Arab world are mutually intelligible or identical because the written spoken language is Arabic.

```
                      ┌────────────────────────────────────────┐
                      │          Spoken / Written:             │
                      │     Modern Standard Arabic (MSA)       │
                      └──────────────────┬─────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   ┌───────────────────────────┐                   ┌───────────────────────────┐
   │    ArabSign (ArSL)        │                   │     Egyptian Sign (ESL)   │
   │  Region: Saudi / Gulf     │   NOT IDENTICAL   │     Region: Egypt         │
   │  Grammar: ArSL Gloss      │ ◄───────────────► │     Grammar: ESL Syntax   │
   │  Vocabulary: 95 ArSL Signs│                   │     Vocabulary: Mendeley  │
   └───────────────────────────┘                   └───────────────────────────┘
```

### Key Linguistic Distinctions:
1. **Lexical Divergence:** Over 60–75% of basic vocabulary differs between Egyptian Sign Language (ESL) and Gulf/Saudi Sign Language (ArSL). Signs for common concepts (e.g., family members, government, numbers, food) utilize different handshapes, orientations, and locations.
2. **Grammar & Word Order:**
   - ArabSign annotations reveal two distinct sentence structures: **ArSL Gloss order** vs **Spoken Arabic Standard order** (e.g., in sample 1, `الحمد الله` vs `حمد الله`; in sample 31, topicalized gloss `الصم تمني اشارة-هم جميع السامع`).
   - ESL uses distinctive spatial topicalization, non-manual markers (facial expressions, mouthings in Egyptian dialect), and unique classifier predicates that do not map directly to Saudi Kinect skeletons.
3. **Mandatory Policy Rule:** ArabSign serves as an architectural template for **Continuous Arabic Sign Language Translation models (Seq2Seq / CTC)**, but **must never be labeled as ESL** or mixed into the P0 Egyptian training set.

---

## 4. Continuous vs Isolated Sign Language Modeling

The transition from Phase 1/2 (Isolated Signs) to Phase 3+ (Continuous Sentences) involves substantial architectural differences:

| Dimension | Phase 1 & 2: Isolated Signs (P0 ESL, P1 WLASL) | Phase 3+: Continuous Signs (P2 How2Sign, P2 ArabSign) |
| :--- | :--- | :--- |
| **Input Unit** | Single segmented clip containing 1 target sign | Multi-sign video containing continuous sentence stream |
| **Boundary Ambiguity** | Fixed start and end; clear pre-segmented sign | Unsegmented; coarticulation between adjacent signs |
| **Model Architecture** | Spatial Temporal GCN, 1D-CNN, BiLSTM / GRU classifier | CTC-based CNN-BiLSTM, Transformer Encoder-Decoder |
| **Output Target** | Single class label (e.g., class index $0 \dots C-1$) | Sequence of gloss tokens or translated spoken words |
| **Evaluation Metrics** | Top-1 Accuracy, Top-5 Accuracy, Macro-F1, Confusion Matrix | Word Error Rate (WER), BLEU-1 to 4, ROUGE-L |
| **Representation** | 75–125 MediaPipe landmarks per frame | MediaPipe / OpenPose / 3D Kinect Skeletons / CNN features |

---

## 5. Storage, Infrastructure & Compute Assessment

1. **How2Sign:**
   - Full raw RGB videos $\approx 2\text{ TB}$ $\rightarrow$ Exceeds typical local developer environments.
   - **Recommended Approach for SIGNOVA:** Utilize the pre-extracted 2D OpenPose keypoint tarball ($\approx 80\text{ GB}$) or pre-extracted sentence-level clips for feature engineering.
2. **ArabSign:**
   - Raw color videos $\approx 60\text{ GB}$; depth $\approx 40\text{ GB}$; skeleton joints $\approx 5\text{ GB}$.
   - Full annotation and split manifest files ($\approx 1.5\text{ MB}$) have been verified and integrated into `data/metadata/` and `data/splits/`.
   - The 25 3D skeleton joint stream provides an efficient, lightweight representation suitable for fast prototype training on modest GPU hardware.

---

## 6. Quality Gate Status

- [x] **P2 Source Register Delivered:** `data/reports/dataset_sources.csv` populated with full provenance, URLs, terms, and approvals.
- [x] **Download Manifest Cataloged:** `data/reports/download_manifest.csv` created with modality sizes and direct IDs.
- [x] **Canonical Metadata Processed:**
  - `data/metadata/how2sign_canonical_metadata.csv` (35,263 samples)
  - `data/metadata/arabsign_canonical_metadata.csv` (9,335 samples)
  - `data/metadata/arabsign_sentences_mapping.csv` (50 sentences mapped)
- [x] **Split Manifests Generated:**
  - `data/splits/how2sign_split_manifest.csv` (Train/Val/Test with signer audit)
  - `data/splits/arabsign_split_manifest_signer_dependent.csv` (Folds 1–6)
  - `data/splits/arabsign_split_manifest_signer_independent.csv` (Zero-leakage partition)
- [x] **Automated Pipeline Scripts:** `scripts/data_pipeline/download_p2_metadata.py` & `generate_canonical_p2_metadata.py` verified.
- [x] **Quality Flags Documented:** `data/reports/qc_flags_p2.csv` updated.

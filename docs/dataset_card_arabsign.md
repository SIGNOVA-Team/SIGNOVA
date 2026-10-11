---
dataset_info:
  dataset_name: ArabSign
  language: Arabic Sign Language (ArSL - Saudi / Pan-Arab)
  multilingual: false
  modalities:
    - Color RGB Video (1920x1080 @ 30fps)
    - Depth Maps
    - 3D Skeleton Joints (25 Kinect V2 joints)
    - ArSL Gloss Annotations
    - Modern Standard Arabic Text Sentences
  size_categories:
    - 1K<n<10K
  splits:
    - name: all_samples
      num_samples: 9335
    - name: signer_independent_train
      num_samples: 6323
    - name: signer_independent_val
      num_samples: 1500
    - name: signer_independent_test
      num_samples: 1512
  license: academic-research-noncommercial-kfupm
  task_categories:
    - continuous-sign-language-recognition
    - multimodal-gesture-recognition
    - skeleton-based-action-recognition
---

# Dataset Card: ArabSign (Continuous ArSL)

## 1. Dataset Summary
ArabSign is a multimodal continuous Arabic Sign Language (ArSL) dataset created by Dr. Hamzah Luqman at King Fahd University of Petroleum & Minerals (KFUPM), Saudi Arabia (published in IEEE FG 2023). It comprises 9,335 video and skeleton samples representing 50 continuous sentences performed by 6 male deaf signers.

In the **SIGNOVA** roadmap, ArabSign is designated as a **Priority 2 (P2)** continuous sign language research reference. It is used to study temporal sequence modeling, ArSL syntactic structure, and multimodal feature fusion (RGB + Skeleton), preparing the foundation for future continuous Arabic sign systems.

> [!WARNING]
> **CRITICAL LINGUISTIC DIRECTIVE (DO NOT RELABEL AS ESL):**
> ArabSign represents Saudi / Pan-Arab Sign Language (ArSL). It is linguistically distinct from Egyptian Sign Language (ESL). Under no circumstances may ArabSign be relabeled as ESL or merged into the primary Egyptian Sign Language training pipeline.

## 2. Dataset Structure

### 2.1 Volume & Balance Statistics
- **Total Samples:** Exactly 9,335 samples
- **Total Duration:** 10 hours, 13 minutes
- **Sentences:** 50 unique continuous sentences
- **Vocabulary:** 95 unique sign vocabulary items; 155 total signs across sentences
- **Average Sentence Length:** 3.1 signs (range: 2 to 5 signs)
- **Frames per Sentence:** Average 130.3 frames ($\approx 4.3$ seconds at 30 fps)
- **Signer Distribution:**
  - Signer 01: 1,535 samples
  - Signer 02: 1,598 samples
  - Signer 03: 1,640 samples
  - Signer 04: 1,550 samples
  - Signer 05: 1,500 samples
  - Signer 06: 1,512 samples
  - *Each sentence was repeated at least 30 times by each signer across multiple sessions to ensure diversity.*

### 2.2 Dual-Tier Arabic Annotations
ArabSign uniquely provides annotations for both **ArSL sign gloss order** and **spoken Arabic standard order**, enabling linguistic modeling of sign syntax vs spoken text:
- **Example 1:** ArSL Gloss: `الحمد الله` | Arabic Text: `حمد الله`
- **Example 2:** ArSL Gloss: `الصم تمني اشارة-هم جميع السامع` (Deaf wish their signs are for all hearing) | Arabic Text: `تمني الصم اشارتهم لجميع السامعين`
- **Example 3:** ArSL Gloss: `اين السعودية` | Arabic Text: `اين السعودية`

All 50 sentences are fully mapped in `data/metadata/arabsign_sentences_mapping.csv`.

### 2.3 Split Protocols in SIGNOVA
1. **Signer-Dependent Protocol (Author's 6-fold):**
   - Stored in `data/splits/arabsign_split_manifest_signer_dependent.csv`
   - Replicates the author's cross-validation where each signer has their own train/test split.
2. **Signer-Independent Protocol (SIGNOVA Benchmark):**
   - Stored in `data/splits/arabsign_split_manifest_signer_independent.csv`
   - **Train:** Signers 01, 02, 03, 04 (6,323 samples, 67.7%)
   - **Validation:** Signer 05 (1,500 samples, 16.1%)
   - **Test:** Signer 06 (1,512 samples, 16.2%)
   - Guarantees **0% signer overlap**, providing an uncompromising test of model generalization to unseen signers.

## 3. Data Collection & Recording
- **Hardware:** Microsoft Kinect V2 sensor (synchronous RGB color, depth stream, and 25 3D skeleton joints).
- **Environment:** Controlled lab lighting at KFUPM.
- **Signers:** 6 male deaf signers aged 21–30 with varying skin tones. All signers are right-handed; one wore eyeglasses.
- **Signing Speed:** Natural conversational signing speed without artificial pauses between words.

## 4. Licensing & Data Governance
- **License:** Non-Commercial Academic Research Use (KFUPM).
- **Access Route:** Annotation files and code are public via GitHub. Raw RGB and Depth video Google Drive folders require email authorization from the creator (`hluqman@kfupm.edu.sa`).
- **Privacy:** Signer footage is authorized for research benchmark purposes. No personal identifiers beyond anonymized signer IDs (`01`–`06`) are tracked.

## 5. Official Citation
```bibtex
@inproceedings{luqman2023arabsign,
  author={Luqman, Hamzah},
  title={ArabSign: A Multi-modality Dataset and Benchmark for Continuous Arabic Sign Language Recognition},
  booktitle={2023 IEEE 17th International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2023},
  pages={1--8},
  doi={10.1109/FG57933.2023.10042720}
}
```

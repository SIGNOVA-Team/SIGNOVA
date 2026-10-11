---
dataset_info:
  dataset_name: How2Sign
  language: American Sign Language (ASL)
  multilingual: false
  modalities:
    - RGB Video (Frontal)
    - RGB Video (Side View)
    - Depth Video
    - 2D Keypoints (OpenPose Body, Face, Hands)
    - 3D Keypoints (Panoptic Studio subset)
    - Speech Audio
    - English Text Transcriptions
  size_categories:
    - 10K<n<100K
  splits:
    - name: train
      num_samples: 31165
    - name: val
      num_samples: 1741
    - name: test
      num_samples: 2357
  license: cc-by-nc-4.0
  task_categories:
    - continuous-sign-language-recognition
    - sign-language-translation
    - sign-language-production
---

# Dataset Card: How2Sign (Continuous ASL)

## 1. Dataset Summary
How2Sign is a public, large-scale, multimodal dataset for continuous American Sign Language (ASL) developed by Amanda Duarte et al. (Universitat Politècnica de Catalunya, Carnegie Mellon University, Facebook AI). It consists of over 80 hours of sign language videos corresponding to the How2 instructional video corpus. 

Within the **SIGNOVA** project, How2Sign is classified as a **Priority 2 (P2)** exploratory dataset. It serves as an architectural and algorithmic reference for continuous sign language translation (SLT) and continuous sign language recognition (CSLR), complementing the isolated-sign baselines (`P0 — ESL` and `P1 — WLASL`).

## 2. Dataset Structure

### 2.1 Verified Volume & Splits
The dataset is split according to the original How2 dataset splits:
- **Train:** 31,165 sentences
- **Validation:** 1,741 sentences
- **Test:** 2,357 sentences
- **Total Verified Sentences:** 35,263 clips

### 2.2 Canonical Sample Schema in SIGNOVA
Every How2Sign entry is mapped to the canonical SIGNOVA schema:
- `sample_id`: E.g., `H2S_-d5dN54tH2E_0_1`
- `source_dataset`: `How2Sign`
- `source_sample_id`: Original clip name, e.g., `-d5dN54tH2E_0-1-rgb_front`
- `sign_language`: `ASL`
- `original_label`: English subtitle text
- `gloss_label`: Aligned English transcript
- `signer_id`: Signer ID extracted from filename (`1` to `11`)
- `session_id`: Source video ID
- `relative_video_path`: `videos/how2sign/{split}/rgb_front/{sentence_name}.mp4`
- `fps`: 25.0
- `duration_sec`: Duration in seconds calculated from START and END timestamps
- `split`: `train`, `val`, or `test`
- `license_id`: `CC-BY-NC-4.0`
- `qc_status`: `PASS`
- `qc_flags`: `SIGNER_OVERLAP_ACROSS_SPLITS`

## 3. Data Collection & Provenance
- **Environment:** Recorded in two studio settings: a green screen studio for frontal and side views, and Carnegie Mellon University's Panoptic Studio for multiview 3D capture.
- **Signers:** 11 fluent deaf signers who translated instructional videos in topics such as cooking, sports, crafts, and home improvement.
- **Alignment:** Videos and English subtitles were re-aligned and verified by native ASL speakers and sign language linguists.

## 4. Quality Control & Limitations in SIGNOVA

### 4.1 Signer Leakage in Official Splits
A critical finding of the SIGNOVA audit (`QC-P2-001`) is that the official How2Sign splits were partitioned based on video/topic content rather than signer identity. As a consequence, 5 signers appear across all three partitions (train, val, test).
- **Impact:** Models trained and tested on the official benchmark evaluate translation accuracy on familiar signers. They do not demonstrate generalization to unseen signers.
- **SIGNOVA Protocol:** When evaluating zero-shot signer generalization, a signer-stratified partition must be employed.

### 4.2 Storage & Bandwidth
The full raw multiview video dataset exceeds 2 TB. For ML experiments in SIGNOVA, teams are directed to use pre-extracted 2D OpenPose keypoint archives (80 GB) or frontal clips.

## 5. Licensing & Ethics
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Permitted Use:** Non-commercial academic research. Commercial use is strictly prohibited.
- **Privacy:** Signers consented to video distribution for scientific research under IRB protocol. No restricted personal data or secrets are stored in Git.

## 6. Official Citation
```bibtex
@inproceedings{duarte2021how2sign,
  title={How2Sign: A Large-Scale Multimodal Dataset for Continuous American Sign Language},
  author={Duarte, Amanda and Palaskar, Shruti and Ventura, Lucas and Ghadiyaram, Deepti and DeHaan, Kenneth and Metze, Florian and Torres, Jordi and Giro-i-Nieto, Xavier},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={2735--2744},
  year={2021}
}
```

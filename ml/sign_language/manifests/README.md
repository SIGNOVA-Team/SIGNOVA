# SIGNOVA — ML Data Manifests

This directory maintains the operational sample manifests for the Machine Learning training and evaluation pipeline across **ArabSign** and **How2Sign**.

---

## 📋 Manifest Structure & Field Specification

Each manifest CSV maps every dataset sample ID to its linguistic supervision, remote storage location, split assignment, and local caching status:

| Column | Description |
| :--- | :--- |
| `sample_id` | Canonical, collision-resistant identifier (e.g. `ARAB_01_01_01_0001_...` or `HOW2_...`) |
| `dataset` | Source benchmark name (`ArabSign` or `How2Sign`) |
| `language` | Standard linguistic code (`ArSL` for Arabic Sign Language, `ASL` for American Sign Language) |
| `gloss_label` / `target_text` | Ground-truth gloss sequence (ArabSign) or English sentence transcript (How2Sign) |
| `signer_id` | Anonymized signer identity for zero-leakage cross-signer validation |
| `duration_sec` | Sample video duration in seconds |
| `fps` | Video capture frame rate (30.0 for ArabSign, 25.0 for How2Sign) |
| `split` | Assigned evaluation split (`train`, `val`, `test`) adhering to ADR-001 |
| `remote_video_url_or_folder` | Remote cloud storage archive ID or folder link |
| `access_status` | Gating status (`GOOGLE_DRIVE_PUBLIC`, `GATED_EMAIL_REQUEST`, `PUBLIC_GITHUB`) |
| `local_cache_status` | Storage state (`SYNTHETIC_SIMULATED_LOCAL` or `LOCAL_NPY_PRESENT`) |

---

## 🗃️ Files in this Directory

1. **[`arabsign_manifest.csv`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/ml/sign_language/manifests/arabsign_manifest.csv):**
   - **Sample Count:** 9,335 samples
   - **Signers:** 6 male deaf signers (`01`–`06`)
   - **Protocol:** Signer-Independent split (`train`: signers 1–4, `val`: signer 5, `test`: signer 6)
   - **Remote Storage:** KFUPM Google Drive folder (gated by email to `hluqman@kfupm.edu.sa`)

2. **[`how2sign_manifest.csv`](file:///c:/Users/moham/Desktop/DEPI/Microsoft%20Machine%20Learning/Technical/Final%20Project/SIGNOVA/ml/sign_language/manifests/how2sign_manifest.csv):**
   - **Sample Count:** 35,263 continuous sentence clips
   - **Signers:** 11 fluent deaf signers
   - **Protocol:** Official content/topic partition (`train`: 31,165, `val`: 1,741, `test`: 2,357)
   - **Remote Storage:** CMU / UPC Google Drive public archives (~150 GB frontal clips / ~80 GB OpenPose keypoints)

---

## 🛡️ Resource Protection (Laptop Storage Protection)
Neither full video dataset (exceeding 2 TB combined) is stored locally on developer laptops. The manifests allow remote runtimes (Google Colab, Kaggle) to fetch and stream only the specific batches or keypoint slices needed for training.

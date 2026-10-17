# SIGNOVA — KArSL-502 Data Engineering

This folder contains the KArSL metadata, QC and evaluation manifests, third-party Pose manifest, validation scripts, recorded audits, and a visual review register. Contact sheet images are stored separately on the project owner's Google Drive. It does **not** contain the original `.7z` videos, the bulk Kaggle `.pose` files, or the NPY keypoint arrays.

## At a glance

| Data | Coverage | Qualification |
| --- | ---: | --- |
| Original video inventory | 75,514 videos; 502 classes; 3 signers; 30 archives | Manifest only; source videos are separate |
| QC | 75,296 PASS, 150 EXCLUDE, 68 REVIEW | REVIEW cases await expert confirmation |
| PASS-only benchmark | 75,095 videos; 500 classes; 3 signer-independent folds | Excludes classes 27 and 28 |
| External Kaggle Pose | 75,515 entries; 502 classes; 3 signers | 502 *sampled* `.pose` files structurally checked |
| Video-to-Pose mapping | 75,514 candidate pairs | **Not identity verified** |
| Contact sheets | 502 generated externally | Images are not published in this repository; semantic labels remain unreviewed |
| CSV streaming stress test | 5/5 PASS, 678,373 rows | Does not benchmark archive decompression or MediaPipe |
| Saved-artifact reproducibility audit | 14/14 PASS | Does not reproduce extraction from raw videos |

## Entry points

- [Handoff](docs/karsl_handoff.md): what the ML team can use and what is pending.
- [Dataset card](docs/karsl_dataset_card.md): provenance, limits, and coverage.
- [Video metadata](data/metadata/karsl_metadata.csv) and [signer-independent splits](data/splits/karsl_split_manifest.csv).
- [Pose manifest](data/pose/karsl_pose_manifest.csv) and [candidate mapping](data/pose/karsl_video_pose_candidate_mapping.csv).
- [Visual review register](data/reports/karsl_502_class_visual_review_register.csv).
- `notebooks/01_KArSL_Inventory.ipynb`, `02_KArSL_QC_and_Splits.ipynb`, `03_KArSL_Keypoints_Audit.ipynb`: **read-only packaged artifact checks**.
- `notebooks/original_pipeline/`: preserved historical Colab code, **not** guaranteed end-to-end reproducible without external files/dependencies.

## Validate

From `karsl/` after installing Python, pandas and numpy:

```bash
python -m scripts.validate_dataset
python -m unittest discover -s tests -v
```

In the verification notebooks, set `SIGNOVA_KARSL_ROOT` to this folder when running outside it.

## Dataset sources

- Original KArSL: https://hamzah-luqman.github.io/KArSL/
- External Pose files: https://www.kaggle.com/datasets/tfmohamedyahia/kars502-pose-75
- Separate NPY keypoint subset: see `data/metadata/dataset_sources.csv` for its recorded provenance; no verified public download URL is asserted here.

## Release limitations

KArSL is Arabic Sign Language; **it is not established to be Egyptian Sign Language**. The Kaggle `.pose` dataset is an independent third-party representation. The video-to-Pose links are candidates inferred from naming and sequence ordering, not confirmed matching identities. Of 12 diagnostic pairs, 11 are unverified and one has an unresolved coordinate alignment mismatch. No accuracy or deployment readiness is claimed. Expert label review, source licensing, and image/derivative redistribution permission remain pending. The public GitHub release of contact sheets should wait for permission confirmation.

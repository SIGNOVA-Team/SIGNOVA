# KArSL-502 — SIGNOVA handoff to the ML team

## Delivery status

The packaged CSV artifacts, contact sheets, code history, and recorded audits are ready for technical review. **Data-source redistribution approval, expert semantic label sign-off, and video-to-Pose identity matching are NOT complete.** This folder does not contain the underlying video archives, raw NPY arrays or bulk `.pose` files.

## Which data to use

| Data | File | Safe interpretation |
| --- | --- | --- |
| Source video inventory and QC | `data/metadata/karsl_metadata.csv` | 75,514 videos: 75,296 PASS, 150 EXCLUDE, 68 REVIEW |
| Original PASS-only benchmark | `data/splits/karsl_split_manifest.csv` | 75,095 distinct videos, 500 classes, three signer-independent train/val/test folds |
| Video access references | `data/metadata/karsl_video_access_manifest.csv` | Archive-relative paths, **not** direct downloadable video URLs |
| Third-party Pose inventory | `data/pose/karsl_pose_manifest.csv` | 75,515 rows, 502 classes, three signers |
| Pose evaluation folds | `data/pose/karsl_pose_signer_folds.csv` | Three signer-independent folds; independent from original video folds |
| Pose candidate links | `data/pose/karsl_video_pose_candidate_mapping.csv` | 75,514 **CANDIDATE_UNVERIFIED** pairs; do not treat as ground truth |
| Visual review | `data/reports/karsl_502_class_visual_review_register.csv` | 502 sheets generated and retained on the project owner's Google Drive; JPGs not included in GitHub; zero semantic labels reviewed |

**Benchmark count clarification:** The historical 75,362 `usable` figure is *not* the current PASS-only benchmark count. Current original benchmark is 75,095 across 500 classes, and excludes REVIEW / EXCLUDE records and classes 27–28. The full original inventory still has 502 classes. The external Pose dataset independently covers 502 classes.

## Recorded quality and test evidence

- Original QC and counts: `data/reports/karsl_qc_report.md`, `data/reports/karsl_qc_flags.csv`.
- Duplicate SHA-256 audit: `data/reports/karsl_duplicate_checksum_audit.csv` (50 duplicate pairs / 100 excluded videos). Exact hashes only, not perceptual duplicate detection.
- Original fold leakage: `data/reports/karsl_split_leakage_audit.csv` (three folds, no detected non-PASS inclusion or cross-split hash leakage).
- Structural regression: `data/reports/karsl_regression_scale_report.csv` (16/16 recorded PASS).
- Pose structural content samples: `data/reports/karsl_pose_502_class_quality_audit.csv` (502 actual files sampled, one per class, recorded PASS). **Not** 75,515-file binary audit.
- Pose discrepancy report: `data/pose/karsl_pose_label_discrepancies.csv` (33 discrepancies, unresolved).
- Pose candidate mapping exceptions: `data/reports/karsl_video_pose_mapping_exceptions.csv` (75,513 frame-count agreements and one mismatch; frame count does **not** establish identity).
- Video/Pose diagnostic: `data/reports/karsl_video_pose_identity_sample.csv` (12 selected; 11 NOT_VERIFIED and one COORDINATE_MISMATCH_UNRESOLVED, KARSL_000001).
- Open video QC cases: `data/reports/karsl_review_decision_register.csv` (68 pending expert confirmation).
- Contact sheet readiness: 502/502 JPGs generated and checked in the owner's Google Drive staging area. The JPGs are intentionally excluded from this public GitHub release. No label meaning independently validated.
- CSV streaming stress test: `data/reports/karsl_pipeline_stress_test.csv` and `_summary.json` (5/5 PASS; 678,373 CSV rows, ~11.01 sec, reported peak process RSS ~939.57 MB including prior Colab allocations). **Not** a MediaPipe or decompression benchmark.
- Saved-artifact reproducibility check: `data/reports/karsl_reproducibility_audit.json` (14/14 PASS). **Does not** regenerate from original archives or independently rerun splits.
- Automated tests: `tests/` and `scripts/validate_dataset.py`; rerun locally with dependencies installed.

## Source and processing provenance

- Original source: https://hamzah-luqman.github.io/KArSL/
- External Pose: https://www.kaggle.com/datasets/tfmohamedyahia/kars502-pose-75
- Source register: `data/metadata/dataset_sources.csv`.
- `notebooks/original_pipeline/`: historical processing / label QC / five-video MediaPipe pilot with code preserved; outputs cleared. Execution requires original archives and additional model assets. **Do not execute all historical cells unattended**.
- `notebooks/01_KArSL_Inventory.ipynb`, `02_KArSL_QC_and_Splits.ipynb`, `03_KArSL_Keypoints_Audit.ipynb`: checks on artifacts distributed in this package, not full pipeline replay.
- Scripts `scripts/data_pipeline/generate_qc_reports.py` and `generate_karsl_splits.py` regenerate derived reports/folds from canonical metadata, **not from raw video**.

## Pending decisions — do not silently approve

1. Expert review of 68 original QC REVIEW cases, plus disputed Pose labels; do not auto-correct labels.
2. License, privacy/consent, and redistribution permission for original footage, derived contact sheets, and third-party Pose assets. The project GitHub repository is public.
3. Video-to-Pose identity: folder/signer/rank correspondence is a **candidate** only. Coordinate-alignment test on KARSL_000001 was inconclusive; do not pair files as verified supervision.
4. Project lead sign-off on chosen dataset use and remaining limitations.

**No original video or label is modified by the packaged visual review and identity tests.**

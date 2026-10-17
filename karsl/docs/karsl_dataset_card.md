# Dataset card — KArSL-502 preparation for SIGNOVA

## Intended use and language

Research/prototyping of **isolated** Arabic Sign Language signs in the SIGNOVA training project. Not independently established to be Egyptian Sign Language, continuous signing, or general-purpose translation. No accuracy or deployment claims.

## Original video source

- Official page: https://hamzah-luqman.github.io/KArSL/
- Inventory: **75,514 videos**, 502 sign classes, three signers, across 30 `.7z` archives.
- Recorded video metadata includes signer, labels, frame counts, FPS, dimensions, author split, QC flags and SHA-256 hashes: `data/metadata/karsl_metadata.csv`.
- QC totals: 75,296 PASS; 150 EXCLUDE; 68 REVIEW.
- Original source videos, archive files, and raw skeleton/depth recordings are not distributed in this package.
- Label/filename disagreements and QC decisions are recorded; **68 REVIEW rows still await qualified human assessment**.
- Known limited signer diversity (three signers) and isolated-sign-only coverage constrain conclusions.

## Benchmarks and splitting

- `data/splits/karsl_split_manifest.csv`: **75,095 PASS-only** samples, **500 shared classes**, three signer-independent folds with train/validation/test designations. Original classes 27/28 are outside the shared benchmark.
- `data/reports/karsl_split_leakage_audit.csv`: recorded no cross-split exact-checksum leakage, duplicate hash inclusions or non-PASS rows in the benchmark folds.
- Historical intermediate sample counts are not the current benchmark denominator.

## Other keypoints subset

A separate third-party NPY keypoints subset is described by `data/metadata/keypoints_manifest.csv` and associated reports. Coverage is narrower than the original video inventory; underlying NPY arrays are not included. Do not conflate with the following `.pose` dataset.

## External Kaggle Pose dataset

- Source: https://www.kaggle.com/datasets/tfmohamedyahia/kars502-pose-75
- `data/pose/karsl_pose_manifest.csv`: **75,515** entries, 502 classes, three signers; `.pose` landmark sequences with body (33), face (468), and each hand (21): 543 total points/frame.
- `data/pose/karsl_pose_signer_folds.csv`: three signer-independent folds; publisher's own split is retained separately in `data/pose/karsl_pose_source_split.csv`.
- `data/reports/karsl_pose_502_class_quality_audit.csv`: **502 actual Pose files** inspected, one per class, with structural checks recorded as PASS. This is sampled verification, not inspection of all 75,515 binary files.
- `data/pose/karsl_video_pose_candidate_mapping.csv`: 75,514 candidate video/Pose pairs, inferred from class-folder and signer order. **Individual matching identities have not been established**.
- `data/reports/karsl_video_pose_identity_sample.csv`: 12 diagnostic candidates, 11 NOT_VERIFIED, one with unresolved coordinate mismatch. Do not assume pixel coordinate equivalence or use the candidate mapping as verified supervision.
- 33 source-label disagreements are documented in `data/pose/karsl_pose_label_discrepancies.csv`; one extra Pose sample candidate `pose_dir/sign_0001_s1_sess1_009.pose` is not assigned in the candidate mapping.

## Visual review materials

- Contact sheets: 502 JPG sheets generated from representative PASS videos, stored separately on the project owner's Google Drive and not distributed in this public repository.
- `data/reports/karsl_502_class_visual_review_register.csv`: 502 rows marked **READY** for visual review, but **NOT_REVIEWED** for semantic label correctness.

## Validation and reproducibility scope

- CSV chunk-reading benchmark: **5/5 PASS**, 678,373 rows streamed; not a raw-video/MediaPipe throughput benchmark.
- Saved-artifact count/status audit: **14/14 PASS**; not complete end-to-end recomputation from raw archives.
- Historical Colab notebooks are preserved under `notebooks/original_pipeline/` as source code history, without execution outputs.
- Read-only verification notebooks and automated tests check current packaged artifacts. Underlying videos/pose binaries are not included.

## Licensing and distribution

Original KArSL redistribution and participant image permissions, as well as upstream reuse conditions for third-party Pose and NPY representations, **have not been established** from source ownership evidence. Any claimed Kaggle listing license should not be treated as blanket permission to redistribute original participant images. Expert label validation and project-lead approval are pending. **Avoid public distribution of original participant images/contact sheets before permission is confirmed.**

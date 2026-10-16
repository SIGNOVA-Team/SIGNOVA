# Handoff: KArSL-502 (dataset team to ML team)

Dataset version: KArSL-502 v1, karsl_metadata.csv, karsl_split_manifest.csv. Prepared 2026-10-15. Approval by the project lead: pending.

## Files
- karsl_metadata.csv: one row per video with labels, QC status and usable flag.
- karsl_split_manifest.csv: author_split and leave-one-signer-out columns across 3 folds for usable benchmark samples.
- keypoints_manifest.csv, keypoints_quality.csv: keypoint benchmark manifests and quality reports.
- karsl_class_mapping.csv, karsl_qc_flags.csv, karsl_qc_report.md, karsl_open_review_cases.csv, karsl_dataset_card.md, dataset_sources.csv, download_manifest.csv.

## How to load
1. Read karsl_metadata.csv and keep rows with usable == True (75362 rows).
2. Use class_label as the class key (502 classes). original_label keeps the source value.
3. Use the karsl_split_manifest.csv columns. fold and split columns define the 3-fold cross-validation scheme.
4. Videos are inside the .7z archives: archive_relpath plus video_path_in_archive. Raw videos are not in Git.
5. REVIEW samples (66 usable) can be included or excluded with qc_status.

## Constraints
- Do not use author_split alone for evaluation claims.
- With S01 as test, classes 27 and 28 have no test samples.
- Label decisions were not reviewed by a sign-language expert.
- License is not stated. Derived data may not be redistributed until it is clarified.
- Representation extraction (for example landmarks) is decided and documented by the ML team. This team made no such choice.

## Version 3 Benchmark Update

See karsl_benchmark_summary.md for the final PASS-only benchmark.
The approved-by-tests subset contains 75,095 samples and 500 classes.
Three signer-independent folds contain train, validation and test partitions.
Classes 27 and 28 are outside this benchmark.
The 502-class original dataset remains preserved.
The benchmark is technically validated, not legally approved for use.
License, expert review and project-lead approval remain pending.

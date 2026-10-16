# Handoff: KArSL-502 (dataset team to ML team)

Dataset version: KArSL-502 v1, metadata_arsl.csv, split_manifest_v2.csv. Prepared 2026-10-15. Approval by the project lead: pending.

## Files
- metadata_arsl.csv: one row per video with labels, QC status and usable flag.
- split_manifest_v2.csv: author_split and leave-one-signer-out columns for usable samples.
- class_mapping_v2.csv, qc_flags.csv, qc_report.md, count_anomalies.csv, duplicate_report.csv, label_issues.md, dataset_card.md, dataset_sources.csv, download_manifest.csv.

## How to load
1. Read metadata_arsl.csv and keep rows with usable == True (75362 rows).
2. Use class_label as the class key (502 classes). original_label keeps the source value.
3. Use the split_manifest_v2.csv columns. loso_test_S01, loso_test_S02 and loso_test_S03 mark the test signer.
4. Videos are inside the .7z archives: archive_relpath plus video_path_in_archive. Raw videos are not in Git.
5. REVIEW samples (66 usable) can be included or excluded with qc_status.

## Constraints
- Do not use author_split alone for evaluation claims.
- With S01 as test, classes 27 and 28 have no test samples.
- Label decisions were not reviewed by a sign-language expert.
- License is not stated. Derived data may not be redistributed until it is clarified.
- Representation extraction (for example landmarks) is decided and documented by the ML team. This team made no such choice.

## Version 3 Benchmark Update

See benchmark_v3_summary.md for the final PASS-only benchmark.
The approved-by-tests subset contains 75,095 samples and 500 classes.
Three signer-independent folds contain train, validation and test partitions.
Classes 27 and 28 are outside this benchmark.
The 502-class original dataset remains preserved.
The benchmark is technically validated, not legally approved for use.
License, expert review and project-lead approval remain pending.

# KArSL-502 Quality Control Report

Generated: 2026-10-15. Source: KArSL-502 (Arabic Sign Language), 3 signers, 502 classes.

## Summary
- Videos in archives: 75514
- Usable videos: 75362
- QC status: PASS 75296, REVIEW 68, EXCLUDE 150
- Usable per signer: S01 24961, S02 25316, S03 25085
- Usable videos per class: min 100, median 150, max 168

## Methods
- Archive integrity: 7z test on all 30 archives (30 passed).
- Video checks on every video: opens, first and last frame decode, frame count, fps, resolution, duration, SHA-256.
- Exact duplicates: SHA-256 comparison.
- Label checks: filename sign ID against folder, video counts per signer and sign against the expected 50, recording-time clusters, and visual checks by a team member. No sign-language expert was available.

## Results
- Decode failures: 0. Last-frame failures: 0.
- Resolution 1920x1080 and 30 fps for all videos.
- Frames per video: min 5, median 38, max 269.
- Videos in exact-duplicate groups: 100.

## QC reasons (counts)
- SHORT_SIGNER_STYLE: 1624
- QUARANTINE_S01_SIGN27_28_UNRESOLVED: 150
- LABEL_CORRECTED_5_TO_6: 33
- FILENAME_WRONG_LABEL_7_KEPT_BY_GESTURE: 33
- UNDER_10_FRAMES_MANUAL_CHECK: 2

## Exclusions and corrections
- S01 signs 27 and 28 (150 videos) are excluded and quarantined. Folder 0028 holds 50 byte-identical copies of 50 videos in folder 0027, and the other 50 videos in folder 0027 have unique hashes. Unresolved.
- S01 sign 5 to sign 6: 33 videos relabeled in sign_id_corrected (original kept). Evidence: recording-time gap, count arithmetic 17 + 33 = 50, visual check by a team member. No expert review.
- S01 sign 7: 33 files named 0001 are kept under folder ID 7 (count 33 + 17 = 50).
- 2 videos with fewer than 10 frames are excluded from use pending manual check.
- Short videos (fewer than 15 frames): 1624 kept as PASS. By signer: S01 51, S02 1569, S03 4. Treated as signer style, mostly S02 on letter signs.

## Count differences (raw counts before correction, difference of 10 or more)
- S01 sign 5: 83 videos (expected 50, +33)
- S01 sign 6: 17 videos (expected 50, -33)
- S01 sign 27: 100 videos (expected 50, +50)
- S01 sign 348: 25 videos (expected 50, -25)
- S01 sign 363: 25 videos (expected 50, -25)
- S01 sign 364: 25 videos (expected 50, -25)
- S02 sign 499: 68 videos (expected 50, +18)
- S03 sign 59: 60 videos (expected 50, +10)
- S03 sign 163: 25 videos (expected 50, -25)
- S03 sign 175: 25 videos (expected 50, -25)
- S03 sign 184: 25 videos (expected 50, -25)
- S03 sign 387: 25 videos (expected 50, -25)
- S03 sign 388: 25 videos (expected 50, -25)
- All differences are in count_anomalies.csv. Missing recordings are treated as missing data, not removed.

## Splits
- author_split (from the source): same signers and sessions appear in train and test. 76 of 77 signer-batch-date groups appear in both. Do not use it alone to claim generalization.
- split_manifest_v2.csv: leave-one-signer-out columns for S01, S02 and S03. With S01 as test, classes 27 and 28 have no test samples (500 classes).

## Open issues
- License and terms not stated by the source.
- Quarantine of S01 signs 27 and 28 unresolved.
- No sign-language expert review of any label decision.
- Only 3 signers; no demographic metadata.
- Duplicate English labels: signs 213 and 214, signs 104 and 355. Use the sign ID as the class key.

## Version 3 Benchmark Update

See benchmark_v3_summary.md for the final PASS-only benchmark.
The approved-by-tests subset contains 75,095 samples and 500 classes.
Three signer-independent folds contain train, validation and test partitions.
Classes 27 and 28 are outside this benchmark.
The 502-class original dataset remains preserved.
The benchmark is technically validated, not legally approved for use.
License, expert review and project-lead approval remain pending.

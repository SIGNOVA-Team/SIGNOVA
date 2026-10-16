# Dataset Card: KArSL-502 (SIGNOVA preparation)

## Intended use
Isolated Arabic Sign Language sign classification research in a graduation project. Not for commercial use. Not validated for continuous translation or interpreter-level reliability.

## Source and language
- Source: KArSL database, official page https://hamzah-luqman.github.io/KArSL/ (accessed 2026-10-15).
- Language: Arabic Sign Language (ArSL). It is not labeled or claimed as Egyptian Sign Language.
- Modality used: RGB video (Kinect V2). Depth and skeleton were not obtained.
- License: not stated on the source page. Citation of the source paper is requested.

## Composition
- 75514 videos, 502 classes, 3 signers (S01, S02, S03), 30 archives.
- Expected 50 repetitions per signer per sign (75300 total). Actual count differs, see karsl_qc_report.md.
- 1920x1080 at 30 fps. Median 38 frames per video.
- File name format: batch_signer_signID_(day_month_year_hour_minute_second)_c.mp4. The signer is the second field and matches the folder. The meaning of the first field (batch) is inferred, not documented.

## Processing
- Metadata built from archive listings and file names. Original sample IDs and labels preserved.
- Full video QC (decode, duration, fps, resolution, SHA-256).
- Labels: sign ID is the class key. Original and reviewed labels are in karsl_class_mapping.csv.
- sign_id_original is never overwritten. sign_id_corrected holds corrections.

## Splits
- author_split: train and test from the source. Same signers and sessions on both sides.
- karsl_split_manifest.csv: leave-one-signer-out folds for S01, S02 and S03 across 500 benchmark classes.

## Known issues
- English glosses of 48 classes were corrected in karsl_class_mapping.csv without expert review. The sign ID stays the class key.
- S01 signs 27 and 28 quarantined (150 videos).
- S01 sign 5 and 6: 33 videos relabeled to 6 (visual check, no expert).
- S01 sign 7: 33 files with wrong filename sign ID, folder ID used.
- Some signs have 25 instead of 50 videos for one signer.
- Short videos (under 15 frames) are concentrated in S02 letter signs.

## Limitations
- Three signers only: no claim about unseen-signer generalization beyond leave-one-signer-out results.
- Labels were not reviewed by a sign-language expert.
- Only isolated signs. No continuous signing, no out-of-vocabulary signs.
- Archive-level checksums were not computed. Video-level SHA-256 is in karsl_metadata.csv.

## Version 3 Benchmark Update

See karsl_benchmark_summary.md for the final PASS-only benchmark.
The approved-by-tests subset contains 75,095 samples and 500 classes.
Three signer-independent folds contain train, validation and test partitions.
Classes 27 and 28 are outside this benchmark.
The 502-class original dataset remains preserved.
The benchmark is technically validated, not legally approved for use.
License, expert review and project-lead approval remain pending.

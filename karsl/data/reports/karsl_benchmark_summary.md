# KArSL-502 Benchmark Version 3

Date: 2026-10-15
Dataset: KArSL-502
Language: Arabic Sign Language (ArSL), not Egyptian Sign Language
Original samples: 75514
Original classes: 502
PASS samples before class filtering: 75296
Benchmark samples: 75095
Benchmark classes: 500
Signers: 3
QC REVIEW samples: 68
QC EXCLUDE samples: 150
PASS samples outside benchmark: 201

## Split Protocol
Three signer-independent train/validation/test folds.
One signer is assigned to each partition per fold.
All 500 benchmark classes occur in each partition.
There is no sample or signer overlap within a fold.
The same samples occur in different roles across folds.
Model selection must use validation data only.
Final test results must not be used for tuning.

## Limitations
Only three signers are available.
Each fold trains on one signer.
Classes 27 and 28 are excluded from this benchmark.
Original metadata and the 502-class mapping remain preserved.
68 QC REVIEW cases remain unresolved.
Some label corrections lack sign-language expert validation.
The source license is unresolved.
Project-lead approval and usage-rights verification are pending.

## Benchmark Files
- karsl/data/splits/karsl_split_manifest.csv
- karsl/data/splits/karsl_split_distribution.csv
- karsl/data/splits/karsl_class_gaps.csv
- karsl/data/splits/karsl_excluded_classes.csv
- karsl/data/metadata/karsl_class_mapping.csv
- karsl/data/reports/karsl_dataset_tests.csv

## Status
Technical data and split checks: PASS
License approval: PENDING
Expert label review: PENDING
Final project-lead acceptance: PENDING

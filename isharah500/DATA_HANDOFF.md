
# SIGNOVA — Dataset Handoff

## 1. Dataset Overview

Dataset: Isharah-500
Sign Language: Saudi Sign Language
Task: Continuous Sign Language Recognition / Translation

Total samples: 7,500
Total signers: 15

Each sample contains an ordered sequence of video frames.

## 2. Preprocessing

Hand landmarks were extracted using MediaPipe Tasks Vision.

- Maximum detected hands: 2
- Landmarks per hand: 21
- Coordinates per landmark: x, y, z
- Output shape: (T, 2, 63)
- Data type: float32
- Missing landmarks: NaN

T represents the number of frames in a sample.

Temporal hand tracking was applied to reduce hand-slot switching.

Both original and tracked keypoints are preserved.

## 3. Dataset Splits

### SI — Signer Independent

| Split | Samples |
|---|---:|
| Train | 5,000 |
| Dev | 500 |
| Test | 2,000 |

Purpose: Evaluate generalization to unseen signers.

### US — Unseen Sentences

| Split | Samples |
|---|---:|
| Train | 6,000 |
| Dev | 750 |
| Test | 750 |

Purpose: Evaluate generalization to unseen sentences.

SI and US reference the same 7,500 samples.

Do not combine their train/test splits in one experiment.

## 4. Metadata

Metadata files are stored in `metadata/`.

- isharah500_si_train.csv
- isharah500_si_dev.csv
- isharah500_si_test.csv
- isharah500_us_train.csv
- isharah500_us_dev.csv
- isharah500_us_test.csv

Columns:

- sample_id
- gloss
- text
- keypoints_path

Paths are relative to the project root.

## 5. Keypoints

Recommended directory:

data/processed/isharah500/hand_keypoints_tracked/

The tracked keypoints are organized according to
the SI split, even when used in US experiments.

Always load samples using the keypoints_path
column in the metadata.

## 6. Validation

- 7,500 unique samples verified.
- 15,000 raw and tracked NPY files validated.
- No missing or duplicate sample IDs within each split scheme.
- US gloss overlap across splits: 0.
- US text overlap across splits: 0.

The overlap check uses exact text matching after
whitespace normalization.

## 7. Quality Report

Reports:

- reports/isharah500_quality_report.csv
- reports/isharah500_review_required.csv

Results:

- Total samples: 7,500
- Average no-hands ratio: 9.74%
- Samples flagged for review: 1,014

Flagged sample breakdown:

- 30%–50% missing-hand frames: 833
- 50%–75% missing-hand frames: 180
- Over 75% missing-hand frames: 1

Sample 01_0280 was manually inspected.

Its hands were not visible in most inspected frames.

A visual review of additional samples suggests that
idle periods, partially visible hands, and detection
limitations may contribute to missing landmarks.

These observations are preliminary.

No samples were removed.

## 8. Recommendations for ML Team

1. Use tracked keypoints as the initial baseline.
2. Load samples through the metadata CSV files.
3. Handle NaN values before passing data to models.
4. Preserve the original SI and US evaluation splits.
5. Report performance separately for SI and US.
6. Compare training with and without low-quality
   training samples.
7. Do not remove test samples based on quality flags.
8. Consider body and facial landmarks if hand-only
   features are insufficient.

## 9. Limitations

- This dataset represents Saudi Sign Language,
  not Egyptian Sign Language.
- Hand-only keypoints omit facial expressions,
  body posture, and other non-manual features.
- Missing-hand detection does not necessarily
  indicate an invalid video.
- Temporal hand tracking may still contain errors.
- Synthetic timestamps were used during extraction.
- Structural validation does not guarantee
  landmark accuracy.

## 10. Status

Data extraction: Completed
Structural validation: Completed
SI metadata: Completed
US metadata: Completed
US sentence overlap check: Completed
Quality report: Completed
Manual quality review: Partial

The dataset is ready for initial ML experimentation,
with the documented limitations.
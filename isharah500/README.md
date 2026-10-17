# SIGNOVA — Hand Keypoint Processing

## Overview

A reusable data-processing pipeline for extracting hand landmarks
from sign-language video frames using MediaPipe Tasks Vision.

The pipeline supports parallel processing, experimental temporal
hand tracking, output validation, and resumable execution.

## Project Structure

- `src/hand_extraction.py` — MediaPipe hand-landmark extraction.
- `src/hand_tracking.py` — Experimental hand-slot tracking.
- `src/hand_pipeline.py` — Parallel extraction and tracking.
- `src/parallel_extraction.py` — Parallel extraction only.
- `src/batch_tracking.py` — Batch tracking only.
- `src/keypoint_validation.py` — NumPy output validation.
- `tests/` — Automated unit tests.
- `notebooks/` — Data exploration and experiments.
- `models/hand_landmarker.task` — MediaPipe model.
- `metadata/` — SI and US dataset split metadata.
- `reports/` — Keypoint validation and detection quality reports.
- `DATA_HANDOFF.md` — Dataset handoff documentation for the ML team.


## Requirements

- Python 3.13 (tested with 3.13.15)
- NumPy
- MediaPipe Tasks Vision
- pytest (for testing)

### MediaPipe Model

The MediaPipe Hand Landmarker model is not included in this repository.

Download the Hand Landmarker model from the official MediaPipe documentation and save it as:

`models/hand_landmarker.task`

## Input Format

Each sample is a directory containing ordered JPG frames:

```text
videos/
    02_0001/
        frame0000.jpg
        frame0001.jpg
        ...
    02_0002/
        ...
```

## Run the Pipeline


The following command demonstrates processing a single signer directory (signer `02`).

Run from the project root:

```bash
python src/hand_pipeline.py \
  --input "data/raw/isharah500/videos/02/02" \
  --model "models/hand_landmarker.task" \
  --raw-output "data/processed/isharah500/hand_keypoints/dev" \
  --tracked-output "data/processed/isharah500/hand_keypoints_tracked/dev" \
  --workers 4
```

The input directory must directly contain sample directories.

## Output

Each sample produces two NumPy arrays:

- Raw keypoints: `(frames, 2, 63)`
- Tracked keypoints: `(frames, 2, 63)`

Arrays use `float32`. Missing detections are represented by `NaN`.

## Resume and Recovery

The pipeline validates saved arrays before skipping them.

- Valid raw and tracked outputs: skip.
- Invalid or missing tracked output: rebuild from valid raw output.
- Invalid or missing raw output: rerun extraction and tracking.

Outputs are saved using temporary files and atomic replacement.

## Tests

```bash
python -m pytest tests -v
```

Current test suite: 12 tests passed.

## Benchmark

Five sample videos on the development machine:

| Workers | Time |
| --- | ---: |
| 1 | 38.91 seconds |
| 2 | 23.93 seconds |
| 4 | 13.54 seconds |

Four workers achieved approximately 2.87x speedup in this test.
Performance may differ on larger workloads.

## Dataset Scope

Isharah-500 is a Saudi Sign Language dataset for
sentence-level continuous sign-language recognition.

The processed signer `02` belongs to the signer-independent
development split, not the training split.

Do not treat this dataset as Egyptian Sign Language or
mix it directly with isolated-sign datasets.

## Known Limitations

- Hand-slot tracking is experimental and can still swap identities.
- MediaPipe handedness may depend on image mirroring.
- Validation checks array structure, not semantic correctness.
- Timestamps assume 30 FPS; original recording FPS is unverified.
- All 7,500 samples from 15 signers have been processed and structurally validated. Detection-quality limitations remain, and manual review has been performed on selected samples only.
- The pipeline does not train a recognition model.
- Dataset usage must comply with its license and team approval.

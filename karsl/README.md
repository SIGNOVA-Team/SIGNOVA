# KArSL-502 Dataset Engineering

## Project
SIGNOVA - Arabic Sign Language Dataset Engineering

## Raw Video Benchmark
- 75,514 original metadata records
- 75,095 benchmark samples
- 500 classes
- 3 signers
- 3 signer-independent evaluation folds

## Keypoints Benchmark
- 13,761 matched Kaggle samples
- 112 available classes
- 8,793 samples covering 62 common classes
- Left hand: 63 features
- Pose: 99 features
- Right hand: 63 features
- Total: 225 features per frame
- Variable sequence lengths
- No face keypoints

## Quality Control
- 8,793 keypoint samples passed structural validation
- 1,280 samples contain one or more all-zero groups
- Zero-group interpretation remains pending

## Repository Structure
- data/metadata: Dataset metadata and label mappings
- data/reports: QC and validation reports
- data/splits: Dataset split manifests
- docs: Dataset card and handoff documentation
- notebooks: Read-only verification notebooks
- scripts: Dataset validation script
- tests: Automated validation tests

## Validation
Run from the karsl directory:

    pip install pandas numpy
    python -m scripts.validate_dataset
    python -m unittest discover -s tests -v

## Data Access
Raw video archives and NPY keypoints are not included.
Raw videos originate from the official KArSL-502 dataset.
Available keypoints originate from Kaggle:
https://www.kaggle.com/datasets/yousefdotpy/blablabla

Notebook paths can be configured with:
- SIGNOVA_KARSL_ROOT
- KARSL_ARCHIVE_ROOT

The notebooks verify previously generated outputs.
They do not recreate the entire dataset pipeline.

## Limitations
- KArSL is not verified as Egyptian Sign Language.
- The raw benchmark and keypoint benchmark have different coverage.
- Each keypoint evaluation fold trains on one signer.
- Source usage rights remain unresolved.
- Kaggle keypoint redistribution rights are unverified.
- Expert label review and project approval remain pending.

## Status
Dataset integrity and structural verification completed.
No model accuracy or production readiness is claimed.

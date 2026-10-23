# SIGNOVA Isharah-500 ML module

This directory contains the Isharah-500 continuous sign language recognition
baseline and its real-data hand-keypoint pipeline. The recognition baseline is
a hand-centric BiGRU with a CTC objective evaluated first on the signer
independent (SI) protocol.

The current validated SI checkpoint is:

```text
checkpoints/si_hand_centric_resume_15ep/isharah500_si_best.pt
```

The checkpoint is distributed through the approved external artifact store:

<https://drive.google.com/file/d/1DykSTujZvhdhS0R93mNTrs7CDper66Wn/view?usp=sharing>

Download it to:

```text
isharah500/checkpoints/si_hand_centric_resume_15ep/isharah500_si_best.pt
```

Verify the downloaded file has SHA-256
`f18aabc81912c1853d2c15ebaa522cfbd0f1c5f986d1469afd849ef218aa9593`.
No retraining is required to use this frozen checkpoint.

Its corrected SI dev evaluation is GER `0.7135` and Exact Match `0.07` over
500 samples. The checkpoint was selected using SI dev only. A separate,
one-time held-out SI test evaluation was performed after the checkpoint was
frozen; its results are reported below and were not used for model selection.

## Scope and status

Completed:

- Real tracked hand keypoints with shape `(T, 2, 63)`.
- Strict dataset loading with no synthetic fallback for ML training/evaluation.
- SI and US metadata contracts.
- Training vocabulary built from the selected protocol's training split.
- Hand-centric normalization, serializable preprocessing, and mask handling.
- BiGRU + CTC model, greedy decoding, checkpoint resume, and SI training CLI.
- Canonical edit metrics and SI error analysis.
- Reusable Python inference interface and backend integration guide.
- Frozen checkpoint held-out SI test evaluation with a test-specific report.

The model is an experimental hand-only baseline. It is not a production
recognizer. Its final held-out SI test results are reported for handover and
were not used for model selection.

## Environment and dependencies

Use the existing Windows virtual environment. No dependency installation is
prescribed by this handoff.

Windows paths shown below are verified local examples, not mandatory paths.
On another machine, activate the compatible environment and use `python`, or
replace the example paths with that machine's project, data, and artifact
locations.

Core packages used by the ML module are:

- Python 3.13.15 (verified Windows reference environment)
- PyTorch 2.8.0+cu126 with CUDA runtime 12.6 (verified reference build)
- NumPy 2.5.4
- pytest 9.1.1

The extraction utilities additionally use MediaPipe 1.1.1 and Pillow 12.3.0.
The verified GPU is an NVIDIA Quadro P2000. Backend integration only needs the
inference dependencies and the trained checkpoint.

Scoped manifests are provided in `requirements-runtime.txt`,
`requirements-training.txt`, `requirements-extraction.txt`, and
`requirements-dev.txt`, and `requirements-torch-cu126.txt`. The CUDA-enabled
PyTorch package must be obtained from the approved cu126 package index or wheel
source; PyPI alone must not be assumed to reproduce it. See
[`docs/dependency_and_artifact_handoff.md`](docs/dependency_and_artifact_handoff.md)
for the verification command and checkpoint delivery procedure.

From the project root on Windows, after activating the environment:

```powershell
python --version
python -c "import numpy, torch; print(numpy.__version__, torch.__version__)"
```

The verified interpreter path
`E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe` may be used when
the existing Windows environment is not activated.

## Repository layout

| Path | Purpose |
|---|---|
| `src/isharah_dataset.py` | Metadata loading, strict keypoint validation, preprocessing, vocabulary, and CTC collator |
| `src/isharah_ctc.py` | BiGRU + CTC model, loss, and decoders |
| `src/isharah_training.py` | SI dataset construction, training, validation, checkpointing, resume, and preflight |
| `src/isharah_metrics.py` | Canonical edit-distance metrics and grouped evaluation |
| `src/isharah_inference.py` | Reusable checkpoint-backed prediction interface |
| `tests/` | Unit and contract tests |
| `metadata/` | SI and US CSV metadata |
| `checkpoints/` | Small model checkpoints and training histories |
| `reports/` | Existing validation, quality, and error-analysis reports |
| `notebooks/data_inspection.ipynb` | Configurable SI metadata, keypoint, mask, and preprocessing inspection |
| `notebooks/training.ipynb` | BiGRU + CTC architecture, recorded configurations, and non-executing command templates |
| `notebooks/experiments.ipynb` | Saved training-history comparison and charts |
| `notebooks/evaluation.ipynb` | Corrected SI dev metrics, error-pattern charts, and optional inference demonstration |
| `docs/backend_integration.md` | Python backend integration contract |
| `docs/handover_checklist.md` | Release handover verification |

## Notebook usage

The four ML handover notebooks are intentionally portable and do not execute
training automatically. Set `ISHARAH_PROJECT_ROOT` and, for real keypoint
inspection or inference, `ISHARAH_DATA_ROOT` in the notebook kernel environment.

Open them from the project root:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" -m jupyter notebook .\isharah500\notebooks
```

`data_inspection.ipynb` and the optional inference cell in
`evaluation.ipynb` use SI train/dev artifacts only. `training.ipynb` displays
configuration and Windows command templates without invoking training.
`experiments.ipynb` reads the saved histories and corrected SI dev report; it
does not recreate metrics from test data.

## Dataset paths and split rules

The original data root is external to this project. A typical Windows layout
is:

```text
E:\DEPI Project\SIGNOVA_ESL\
  data\processed\isharah500\hand_keypoints_tracked\
```

The project directory is typically:

```text
E:\DEPI Project\SIGNOVA_ESL-isharah500\
```

Metadata CSV paths are resolved relative to the configured data/project roots;
do not hardcode these paths in application code.

| Protocol | Train | Dev | Test |
|---|---:|---:|---:|
| SI | 5,000 | 500 | 2,000 |
| US | 6,000 | 750 | 750 |

SI is signer independent. US is unseen sentence. Keep protocols and split
roles separate. The current validated model uses SI train/dev only.

## Keypoint and preprocessing contract

Each input sample is a real `numpy.float32` array with shape `(T, 2, 63)`.
The 63 features are 21 hand landmarks times XYZ coordinates for each of two
hand slots. Missing detections are represented by NaNs in source arrays.

The validated hand-centric checkpoint uses:

```text
nan_strategy=interpolate
flatten_hands=false
missing_file_policy=error
coordinate_normalization=hand_centric
normalization_epsilon=1e-6
```

Interpolation fills missing values along time. The original valid-hand mask is
preserved and supplied to the model. Hand-centric normalization subtracts each
hand's wrist and scales by the sequence median wrist-to-middle-finger-MCP
distance, with safe handling for missing or degenerate references.

Training and inference restore the serialized checkpoint configuration. Missing
real keypoint files are errors; synthetic or placeholder samples are not valid
for training or evaluation.

## Training and checkpoint resume

Training is configured through `src/isharah_training.py`. Phase 3A smoke mode
uses a few real samples. Phase 3B provides full training and a standalone CTC
preflight. Full training has not been rerun as part of this handoff.

A resume must use a compatible checkpoint and preserve its optimizer, seed,
preprocessing, vocabulary, and model settings. The existing epoch 10 to 15
resume used a separate output directory:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" .\isharah500\src\isharah_training.py --help
```

Use the CLI help on the target Windows environment for the exact available
argument names before launching a new experiment. Do not overwrite existing
checkpoint directories.

## Evaluation and corrected metrics

The canonical SI dev report is
[reports/si_error_analysis_epoch15_corrected.md](reports/si_error_analysis_epoch15_corrected.md).

It evaluates all 500 SI dev samples and reports token-level Levenshtein
alignment:

- GER: `0.7135`
- Exact Match: `0.07` (35/500)
- Substitutions: `840`
- Deletions: `618`
- Insertions: `36`
- Reference glosses: `2,094`

GER is `(substitutions + deletions + insertions) / reference glosses`.
Exact Match is the number of complete sequence matches divided by sample count.

The training history for epoch 15 recorded Exact Match `0.28`. That historical
value came from the pre-refactor metric aggregation and is incorrect. The
corrected independent analysis and canonical metrics implementation report
`0.07`; the historical `0.28` must not be used for comparison or release
claims.

### One-time held-out SI test evaluation

The dedicated read-only evaluator uses only the frozen checkpoint, the
training-derived SI vocabulary, the checkpoint preprocessing configuration,
and greedy CTC decoding. It refuses placeholder files, missing keypoints,
OOV/CTC-invalid samples, existing report paths, checkpoint writes, and
training. It does not perform model selection.

The recorded one-time evaluation command was:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" .\isharah500\src\evaluate_si_test.py --project-root "." --data-root "E:\DEPI Project\SIGNOVA_ESL" --checkpoint ".\isharah500\checkpoints\si_hand_centric_resume_15ep\isharah500_si_best.pt" --json-report ".\isharah500\reports\si_test_evaluation_epoch15.json" --device cuda --batch-size 8
```

The report is test-specific and contains GER, Exact Match, substitutions,
deletions, insertions, and evaluated sample count. The command must not be
used to tune or replace the selected checkpoint.

The completed report records:

- 2,000 samples and 8,376 reference gloss tokens.
- 3,326 substitutions, 3,041 deletions, and 94 insertions.
- 6,461 total edit errors and GER `0.771370582617001`.
- 82 exact matches and Exact Match `0.041`.
- 2,000 CTC-valid samples and 0 CTC-rejected samples.

The final SI test GER is higher than the corrected SI dev GER `0.7135`, and
Exact Match is lower than the corrected dev value `0.07`. The model continues
to show short predictions and deletion-heavy errors. These metrics do not
support a production readiness claim.

## Inference

Load the predictor once during application initialization and reuse it:

```python
import numpy as np
import sys
from pathlib import Path

# Run this example from the repository root. The module is intentionally
# isolated under isharah500/src and is not imported from a global package.
sys.path.insert(0, str((Path.cwd() / "isharah500" / "src").resolve()))
from isharah_inference import load_predictor

predictor = load_predictor("path/to/isharah500_si_best.pt", device_name="cuda")
keypoints = np.load("incoming_sequence.npy", allow_pickle=False)
prediction = predictor.predict(keypoints)
response = prediction.to_dict()
```

The source-path setup above makes this example executable from a fresh
repository checkout. Backend applications should keep the equivalent source
path or package this module under their application namespace.

The output contains `gloss_tokens`, `gloss_text`, `input_length`,
`valid_hand_coverage`, `device`, and the restored preprocessing configuration.
See [docs/backend_integration.md](docs/backend_integration.md) for the full
contract and error behavior.

Real SI dev smoke command using the activated environment:

```powershell
python .\isharah500\src\isharah_inference.py --checkpoint ".\isharah500\checkpoints\si_hand_centric_resume_15ep\isharah500_si_best.pt" --keypoints "<DATA_ROOT>\data\processed\isharah500\hand_keypoints_tracked\dev\02_0001.npy" --device cuda
```

For explicit handover verification, use the CPU smoke command below. It fails
if the checkpoint, SI Dev metadata, or real SI Dev keypoint sample is missing;
the ordinary optional pytest integration test remains separate and may skip
when external artifacts are unavailable.

```powershell
python .\isharah500\src\handover_cpu_smoke.py --project-root "." --data-root "E:\DEPI Project\SIGNOVA_ESL" --checkpoint ".\isharah500\checkpoints\si_hand_centric_resume_15ep\isharah500_si_best.pt"
```

## Tests and checks

Run the full existing suite on Windows:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" -m pytest .\isharah500\tests -q
```

The latest verified Windows suite result is 80 passed, 0 skipped, and 0 failed
in 5.22s, including the real SI Dev checkpoint integration test on CPU.

## Known limitations and error patterns

- The model uses hand landmarks only and omits facial, body, and other
  non-manual signals.
- SI dev predictions remain shorter than targets on average: 3.024 versus
  4.188 glosses.
- Deletions are the dominant aligned error type: 618 deletions and a 0.2951
  deletion rate.
- Errors increase for longer target sequences; the corrected report shows zero
  exact matches for target lengths 5 and above in the analyzed groups.
- Frequent aligned confusions include `انا` to `هو`, `هو` to `الان`, and
  `رغبه` to `هو`.
- Hand detection coverage is variable, and missing detections can affect the
  representation even after interpolation.
- Greedy decoding remains the current baseline; the small prefix-beam
  comparison did not improve the evaluated sample.
- Final SI test performance is measured in
  `reports/si_test_evaluation_epoch15.json`; it is not a model-selection
  metric. US performance remains unevaluated.

## Handover blockers

Final handover still requires an owner to confirm the target deployment
environment, dependency lock/pinning policy, artifact storage policy, and a
separately authorized evaluation plan for held-out test data. These are release
process items; no production readiness claim is made by this module.

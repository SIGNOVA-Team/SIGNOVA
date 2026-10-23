# Isharah-500 dependency and artifact handoff

## Dependency manifests

The scoped manifests in this directory separate the supported workflows:

| Manifest | Use |
|---|---|
| `requirements-runtime.txt` | Loading the checkpoint and running `IsharahPredictor` |
| `requirements-torch-cu126.txt` | Verified CUDA-enabled PyTorch package selection |
| `requirements-training.txt` | Training, validation, preflight, and checkpoint resume |
| `requirements-extraction.txt` | Optional MediaPipe extraction and Pillow-based review utilities |
| `requirements-dev.txt` | Runtime/training dependencies plus pytest |

The source imports NumPy and PyTorch for inference and training, MediaPipe for
hand extraction, Pillow for review-sheet generation, and pytest in tests.
Pandas is a root project dependency but is not imported by the Isharah-500
module itself.

## Verified compatibility information

- Project metadata requires Python `>=3.12`; the verified Windows reference
  environment uses Python `3.13.15`.
- NumPy is pinned to `2.5.4`.
- PyTorch is pinned to package version `2.8.0` in the separate CUDA manifest;
  the verified installed build is `2.8.0+cu126` with CUDA runtime `12.6`.
- MediaPipe is pinned to `1.1.1` for extraction utilities.
- Pillow is pinned to `12.3.0` for review-sheet utilities.
- pytest is pinned to `9.1.1` for development and tests.
- Jupyter is pinned to `1.1.1`, Matplotlib to `3.11.2`, nbconvert to `7.17.2`,
  and ipykernel to `7.4.0` for the four handover notebooks.
- The verified GPU is an NVIDIA Quadro P2000.

The CUDA-enabled PyTorch wheel must be obtained from the approved PyTorch cu126
package index or wheel source. The manifest does not force an index URL, since
the team's package source policy has not been specified. PyPI alone must not be
assumed to reproduce the verified CUDA build.

## Windows environment verification

Run from the project root with the existing virtual environment:

```powershell
& "E:\DEPI Project\SIGNOVA_ESL\.venv\Scripts\python.exe" -c "import sys, numpy, torch, mediapipe, PIL; print('python', sys.version); print('numpy', numpy.__version__); print('torch', torch.__version__); print('torch_cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('gpu_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); print('mediapipe', mediapipe.__version__); print('pillow', PIL.__version__)"
```

Expected reference values are Python `3.13.15`, NumPy `2.5.4`, PyTorch
`2.8.0+cu126`, CUDA runtime `12.6`, MediaPipe `1.1.1`, Pillow `12.3.0`, and an
NVIDIA Quadro P2000. The command reports the installed environment; it does not
install packages or start training.

## Selected checkpoint artifact

The machine-readable manifest is [`artifact_manifest.json`](artifact_manifest.json).
The selected checkpoint is:

```text
isharah500/checkpoints/si_hand_centric_resume_15ep/isharah500_si_best.pt
```

Verified local size: `7,416,965` bytes.

Verified SHA-256:

```text
f18aabc81912c1853d2c15ebaa522cfbd0f1c5f986d1469afd849ef218aa9593
```

The repository-wide `*.pt` ignore rule means this file is not a normal Git
handover artifact. Historical checkpoints should remain excluded. Deliver the
selected checkpoint through the approved external artifact store or secure file
transfer, then verify its checksum on the receiving machine. Git LFS may be
used only after explicit team approval and repository configuration.

The companion history, CTC validation metadata, and corrected SI dev analysis
should travel with the checkpoint when experiment provenance is required.

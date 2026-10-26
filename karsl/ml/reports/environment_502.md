# KArSL-502 reproducible environments

The CPU code and tests are pinned in `../requirements_502_cpu_py312.lock.txt`: CPython 3.12 / Linux x86_64, exact direct and transitive versions, SHA-256 hashes, and PyTorch's CPU build. The lock was resolved with `uv 0.13.4`, installed into a clean virtual environment with `--require-hashes --torch-backend cpu`, and the imports plus local smoke suite passed. Verified versions included Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.4, SciPy 1.18.1, scikit-learn 1.9.1, pandas 2.3.3, and `pose-format==0.15.0`.

```bash
python -m pip install uv==0.13.4
uv venv .venv-karsl --python 3.12
uv pip sync karsl/ml/requirements_502_cpu_py312.lock.txt \
  --python .venv-karsl/bin/python --torch-backend cpu --require-hashes
.venv-karsl/bin/python -m unittest discover -s karsl/ml/tests -v
```

This exact lock is **CPU/Linux-specific**; it does not claim to reproduce the earlier Tesla T4 CUDA/Colab runtime. The full GRU replay was recorded with Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3; the CNN also replayed on Tesla T4, but a complete transitive lock for that historical GPU session was not preserved. The new CPU linear baseline run is described in `baseline_replay_502.md`.

The separate CV research adapter was exercised with Python 3.12, MediaPipe 0.10.21, NumPy 1.26.4 and OpenCV 4.11.0.86. Keep this separate from the ML CPU lock because the tested NumPy versions differ. `requirements_502.txt` remains a compatibility-range option for other compatible devices; do not call it an exact lock. Full model training/replay requires the team's controlled 502-class feature shards in addition to any Python environment.

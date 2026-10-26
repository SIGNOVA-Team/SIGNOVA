# KArSL-502 baseline reproducibility

The historical nearest-centroid and linear-softmax reference scores were recalculated using **all 25,032 frozen train PASS and 25,254 validation PASS samples**, with the frozen loader verifying all 99 train/validation shards, signer separation, SHA-256, labels and sample paths. No test shard was read. The historical linear checkpoint SHA-256 is `a2af758254de890621b1296c9167a4bacae98e2ca4eaec20e1ae8ab2b9eed7e5`, held in the team's controlled `ml_expansion/experiments/baseline_502_v1/linear_softmax_baseline.pt` storage.

| Reference | Validation accuracy | Macro F1 | Top-5 |
| --- | ---: | ---: | ---: |
| Historical centroid, independently recomputed | 19.9731% | 0.176593 | 35.4083% |
| Historical linear checkpoint, independently reloaded | 27.8213% | 0.267526 | 47.8063% |
| New 30-epoch linear training (`baseline_502_reference_v2`, best epoch 3) | 26.6651% | 0.253621 | 47.0421% |

All three historical centroid/linear metrics reproduced exactly. The train-only 4,800-channel mean and `(std + 1e-6)` matched the historical checkpoint bit for bit. For centroid, evaluating **direct negative squared-distance scores** avoids softmax underflow that changes top-5 ties. The v2 standalone trainer records its own optimizer and results; it does **not** claim byte-identical reproduction of the older training trajectory. The v2 full run used CPU, Python 3.12.14, PyTorch 2.14.1+cpu, NumPy 2.5.4, seed 42, and 30 epochs; the historical validation replay ran on Python 3.12.14, PyTorch 2.14.1+cpu and NumPy 2.3.5. These baseline results do not alter the CNN selected on validation, its calibration, or its already reported test metrics.

With private shards mounted under `$PRIVATE_FEATURES` and the historical checkpoint at `$PRIVATE_BASELINE`, run from the repository root:

```bash
python karsl/ml/training/train_baselines_502.py \
  --features-root "$PRIVATE_FEATURES" --cache "$PRIVATE_CACHE" \
  --output "$PRIVATE_OUTPUT/historical_baselines" --device cpu \
  --verify-historical-checkpoint "$PRIVATE_BASELINE"

python karsl/ml/training/train_baselines_502.py \
  --features-root "$PRIVATE_FEATURES" --cache "$PRIVATE_CACHE" \
  --output "$PRIVATE_OUTPUT/new_linear_reference" --device cpu --epochs 30
```

Both commands refuse to overwrite an existing output directory. The first reloads the private historical checkpoint and checks both baseline scores. The second trains a new baseline and saves `experiment.json`, `linear_history.csv`, `nearest_centroid.pt`, `linear_best.pt`, and `result.json`. The large feature shards and reference checkpoint are not stored in the public branch. Benchmark results are validation-only, not new test measurements.

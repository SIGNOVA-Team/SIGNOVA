# KArSL-502 controlled checkpoints

Code, tests and aggregate reports are in this public branch. The trained weights, source Pose binaries, full feature shards, and original source labels are held separately for authorized team use. A checkout alone cannot load the final CNN until the private artifacts are placed at the relative paths below. Verify SHA-256 first.

| Artifact | SHA-256 | Private path relative to the private inference package |
| --- | --- | --- |
| Selected CNN, epoch 12 | `f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254` | `ml_expansion/experiments/cnn_502_pass_v1/best_cnn_502_pass_v1.pt` |
| Frozen validation calibration | `448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4` | `ml_expansion/experiments/cnn_502_pass_v1/calibration_v1/calibration_config.json` |
| Original 502-class labels | `3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a` | `ml_expansion/source_metadata/labels_mapping.json` |
| Comparison GRU | `ba1797040c21de7b5f05039d72bac82e7fdce8e7860df02e445281af6ae577da` | `ml_expansion/experiments/gru_502_pass_v1/best_gru_502_pass_v1.pt` |

The additional historical linear baseline checkpoint SHA-256 is `a2af758254de890621b1296c9167a4bacae98e2ca4eaec20e1ae8ab2b9eed7e5`, at the separately controlled `ml_expansion/experiments/baseline_502_v1/linear_softmax_baseline.pt`. The inference ZIP does not include this baseline, nor full shards or original video media. The frozen preprocessor source SHA-256 is `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c`, and the dataset fingerprint is `5d0d43950dec9c69bef2766e48550463a027b49a44748631b976f211c77e7c4a`.

The controlled Colab directory was `/content/drive/MyDrive/SIGNOVA/KArSL/ml_expansion/`; that historic path is not a portable default. Supply explicit local `base`/`repo_root` arguments as in `../README.md`. Obtain access to the private artifacts through the team; do not commit weights or restricted data to this code branch without a release review.

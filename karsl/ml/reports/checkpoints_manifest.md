# KArSL-502 release artifacts

The project owner authorized publication of these inference and comparison artifacts on the existing `karsl` branch. The predictor verifies their SHA-256 hashes and loads them from this checkout without training or downloading a separate package.

| Artifact | SHA-256 | Repository path |
| --- | --- | --- |
| Selected CNN, epoch 12 | `f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254` | `karsl/ml/artifacts/cnn/best_cnn_502_pass_v1.pt` |
| Frozen validation calibration | `448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4` | `karsl/ml/artifacts/cnn/calibration_config.json` |
| Source 502-class labels | `3aa790f9d40c209c36a0a853a292ff7b60d351ea5e8d90dcd6d0e2652907eb7a` | `karsl/ml/artifacts/labels_mapping.json` |
| Comparison GRU | `ba1797040c21de7b5f05039d72bac82e7fdce8e7860df02e445281af6ae577da` | `karsl/ml/artifacts/gru/best_gru_502_pass_v1.pt` |
| Comparison GRU history | `1e956d7bf3c1b23e175615fa5262cb02d98e2b844bf4c24fe909fecc96abc9df` | `karsl/ml/artifacts/gru/history.csv` |

The final prediction API uses the selected CNN, calibration and labels. The GRU weights and history support experiment comparison; they do not replace the selected model. The historical linear baseline checkpoint SHA-256 is `a2af758254de890621b1296c9167a4bacae98e2ca4eaec20e1ae8ab2b9eed7e5`; it is not needed for final inference, and its training/replay results are in `baseline_replay_502.md`.

The frozen preprocessor SHA-256 is `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c`, and the dataset fingerprint is `5d0d43950dec9c69bef2766e48550463a027b49a44748631b976f211c77e7c4a`. Raw Pose, original video and full feature shards are not part of this inference release; retraining requires their source/storage as described in the README.

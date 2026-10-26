# KArSL-502 CNN full replay verification

**Run:** `cnn_502_clean_replay_20261025T221103Z` on Google Colab Tesla T4, using the frozen code-only branch candidate, controlled train/validation shards, and the original GRU checkpoint as its fixed comparison reference. The training script did not load test shards or modify the selected original checkpoint.

| Check | Independently read-back result |
| --- | --- |
| Checkpoint SHA-256 | Replay and original both `f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254`. The checkpoint files are byte-identical. |
| Model state and metadata | All 22 named tensors are equal, and every non-state checkpoint field matches. |
| Training history | All 18 recorded rows and the `history.csv` bytes match the original Drive run exactly; SHA-256 `3f954772d7282195815b064c206260b54920741a57e068649bd3bd86ccef01ab`. |
| Model selection | Best CNN epoch 12; validation accuracy 0.3870673952641166, macro F1 0.35179408229334247 and top-5 accuracy 0.6324146669834482 on 25,254 PASS validation samples. The saved comparison retains the original GRU validation macro F1 0.342525449074227. |
| Test access | The replay training script reports `Test data accessed: NO`; checkpoint metadata retains `test_evaluated=False`. |

The matching checkpoint bytes and training history provide stronger evidence than score equality alone. This validates the CNN training and validation-selection replay from a clean extracted code package plus private frozen shards. It does not independently rerun the frozen calibration or terminal test evaluation, establish label semantics or data rights, or imply production sign acceptance. The original selected CNN and calibration remain unchanged.

# KArSL-502 frozen calibration and test replay verification

**Run:** `cnn_502_frozen_eval_replay_20261025T221738Z` on the previously selected CNN. The clean-extracted code package used controlled private shards, the original source labels, the original checkpoint and copies of the original validation-calibration and terminal-test protocols. The replay recalculated calibration from validation first. The test script was invoked only after the new calibration configuration matched the original SHA-256.

| Check | Independently read-back result |
| --- | --- |
| Frozen checkpoint | Same selected CNN SHA-256 `f9f0c552b25aa4fa3a776b8bebb154718abeb1978c9717d94c269d5d36de2254`; neither replay changed the model. |
| Validation calibration | Recomputed `calibration_config.json` SHA-256 `448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4`, exactly the original. Temperature `2.4141954373370376`; no threshold met the selection target, so acceptance remains disabled. The calibration report's fields match the original except its completion time. |
| Test protocol | Replay `test_protocol.json` is byte-identical to the original, SHA-256 `b635090d1429c9d9d26fda916e0ce12bdbb16c19fd4c6bc5cfb3af2bf072eb71`. No model, threshold, ranking, or exclusion policy was changed after seeing the historical test. |
| Primary test | 24,536 PASS examples with 500 evaluated true classes and 502 output classes. Accuracy `0.5251874796217803`, macro F1 `0.48520812104660715`, top-5 `0.7429083795239648`; the complete primary and secondary metric dictionaries and subset counts match the original report. |
| Detailed outputs | The 502-row primary per-class CSV and 502 × 502 class-ID confusion matrix are byte-identical to the original exports; SHA-256 `c1d621c6681f6890b0e82bc125b2afda768e036894c5b4a1f46cd19a3bdff883` and `9616495491d6a55fc66417b375d62300d2487100392664df55fb95d1b35cea07`, respectively. |
| Policy and timing | Active-policy coverage `0.0`; accepted accuracy remains undefined. The saved reports assert `model_unchanged=True` and `calibration_unchanged=True`. Completion time and CPU/GPU benchmark measurements may vary with the run and are not reproducibility identity checks. |

These checks demonstrate a clean-artifact replay of the **already frozen** terminal research evaluation, not a new independent test or improved generalization estimate. Historical test exposure, class-27/28 exclusions, one held-out test signer, unverified source labels/Pose-video provenance, unknown-sign behavior, and release rights remain documented limitations. The model's top-k candidates remain advisory; its active policy does not accept a sign.

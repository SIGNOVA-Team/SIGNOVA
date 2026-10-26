# KArSL-502 GRU full replay verification

**Run:** `gru_502_clean_replay_20261025T220240Z` on Google Colab Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3. The standalone script in the branch-candidate code package consumed the frozen private train/validation feature shards from Drive and did not load test shards.

| Check | Recorded replay and independently read-back evidence |
| --- | --- |
| Output | New experiment directory, `result.json`, `history.csv`, `experiment.json`, and `best_gru_502_pass_v1.pt` found on Drive. |
| Selection | Epoch 17; 23 epochs run; validation accuracy 0.3775639502653045, macro F1 0.342525449074227, top-5 0.599192207175101. These values exactly equal the prior saved run. |
| Training history | All 23 CSV rows match the original private GRU `history.csv` exactly. |
| Weights | All 20 named model-state tensors have identical keys, shape, dtype, and exact values in the original and replay checkpoints. |
| Checkpoint metadata | Every common metadata field, including source fingerprint, preprocessing version, class order, selection criterion, best epoch, validation metrics and `test_evaluated=False`, matches. |
| SHA-256 | Original `ba1797040c21de7b5f05039d72bac82e7fdce8e7860df02e445281af6ae577da`; replay `d746e4d4c078585c4594e5472fe58d0a7adbe52a080ab44c454319e8b5744b92`. The replay checkpoint additionally records `experiment_config`, so bytewise checkpoint hashes differ even though all model tensors and shared metadata are identical. |

This independently verifies the saved replay artifacts against the original checkpoint and history. The 99 train/validation shard contents were checked by the replay script in Colab, but were not all downloaded and reprocessed a second time in the local audit environment. The CNN full-training replay, full frozen evaluation replay, data/provenance/semantic approvals and OOV assessment remain separate gates. The replay checkpoint is an additional private experiment artifact; the selected deployed research CNN and its frozen calibration have not been replaced.

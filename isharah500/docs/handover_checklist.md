# Isharah-500 handover checklist

Status values mean verified in the current workspace, pending external
confirmation, or blocked by a missing release decision.

## Code and data contract

| Deliverable | Status | Evidence / action |
|---|---|---|
| Strict real keypoint shape (T, 2, 63) | Verified | src/isharah_dataset.py and dataset tests |
| Float32 and invalid-value validation | Verified | Dataset and inference tests |
| Missing-file policy rejects absent real samples | Verified | Training dataset configuration uses missing_file_policy=error |
| No synthetic training/evaluation fallback | Verified | Phase 1 and Phase 3A/3B implementation |
| SI and US metadata files | Verified | metadata/isharah500_*_train/dev/test.csv |
| Training vocabulary isolated to selected train split | Verified | Dataset vocabulary contract and tests |
| Serialized preprocessing configuration | Verified | Checkpoint loader and inference interface |
| Backend Python prediction contract | Verified | docs/backend_integration.md, src/isharah_inference.py |
| HTTP service | Pending | Deliberately outside this phase |

## Model and experiment artifacts

| Deliverable | Status | Evidence / action |
|---|---|---|
| BiGRU + CTC implementation | Verified | src/isharah_ctc.py |
| Greedy CTC decoding | Verified | Existing decoder tests and inference smoke test |
| SI hand-centric epoch 15 checkpoint | Verified | checkpoints/si_hand_centric_resume_15ep/isharah500_si_best.pt |
| Optimizer/resume checkpoint support | Verified | Resume tests and epoch 10 to 15 artifacts |
| Corrected SI dev error analysis | Verified | reports/si_error_analysis_epoch15_corrected.{json,md} |
| Experiment comparison summary | Verified | reports/experiment_comparison.md |
| US training/evaluation result | Pending | No validated US result is included in this handoff |
| SI test result | Verified | reports/si_test_evaluation_epoch15.json; 2,000 samples evaluated after checkpoint freeze |
| Reproducible dependency lock | Pending | Confirm package versions in the deployment environment |
| Checkpoint artifact retention policy | Pending | Decide which historical checkpoints are archived versus shipped |

## Verification

| Check | Status | Evidence / action |
|---|---|---|
| Automated suite | Verified | Latest verified Windows result: 79 passed, 1 optional real-data test skipped in 5.77s |
| Inference smoke test on real SI dev input | Verified | CUDA smoke test passed |
| Training-data CTC preflight | Verified | Standalone SI preflight was added and used before pilot training |
| Syntax validation in WSL | Verified | Python compilation checks pass for changed Python files |
| CUDA runtime on target backend host | Pending | Run the backend environment smoke test |
| External dataset paths on deployment host | Pending | Configure data roots if offline evaluation or inference receives raw local keypoint files |
| No raw/processed dataset committed | Verified | .gitignore excludes data/raw/ and data/processed/ |
| No large untracked files detected | Verified | Current files above 20 MB: none |
| Checkpoint sizes audited | Verified | Model checkpoints are approximately 7.4 MB each |
| Historical reports preserved | Verified | Existing reports were not overwritten |

## Git and artifact review

The current .gitignore excludes raw data, processed keypoints, local MediaPipe
model assets, caches, and temporary files. It does not ignore checkpoint files.
This is appropriate while the validated checkpoint is an intentional handover
artifact, but the team should decide whether future experimental checkpoints
belong in external artifact storage.

If the team adopts external checkpoint storage, add a narrowly scoped ignore
rule for experimental outputs only, for example:

    checkpoints/experiments/

Do not add a broad rule that hides the validated checkpoint without first
moving it to the approved artifact store. No ignore-file change is required for
the current handover.

## Final blockers

1. Confirm deployment Python and PyTorch/CUDA versions.
2. Deliver the selected checkpoint through the approved artifact store and verify its checksum.
3. Decide whether the backend receives keypoints or must run extraction.
4. Approve the checkpoint and experiment artifact retention policy.
5. Evaluate the US protocol if US generalization is required.
6. Define monitoring and fallback behavior for invalid or low-coverage inputs.

The current module is suitable for engineering handoff and controlled
experimentation. The evidence does not support a production readiness claim.

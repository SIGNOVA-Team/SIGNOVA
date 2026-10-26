# SIGNOVA KArSL 502 handoff review

**Historical review snapshot.** See `../README.md`, `baseline_replay_502.md`, `environment_502.md`, and `cv_nlp_development_contract.md` for the updated branch. Earlier statements that baseline replay, a dependency lock, the 07 notebook or CV adapter are absent no longer describe this package.

**Reviewed:** 2026-10-25. **Scope:** the team delivery message, `SIGNOVA.pdf` Dataset & ML Execution Plan v1.1, the KArSL 502 package and observed Colab results, and the repository's existing CV/inference interface. This review covers Radwa's KArSL research track; it does not claim to audit other teammates' work. No GitHub branch has been submitted for this package.

## Team message

| Requested branch item | Current status | Evidence or action |
| --- | --- | --- |
| Preprocessing, training, evaluation, prediction code | Prepared for the 502-class KArSL track | Frozen `preprocessing.py` plus five standalone scripts. The GRU baseline training is in the notebook; the CNN has a standalone training/comparison script. Full rerun needs private source/shards. |
| Experiment notebook, tried models, results | Prepared | `08_KArSL_502_GRU_Training.ipynb` contains GRU training and later CNN, calibration, test, and inference runs. The standalone report compares GRU and CNN on validation. |
| Final trained model saved | Saved privately; not on a branch | CNN checkpoint, calibration config, source labels, and GRU comparison checkpoint are in the private inference archive. Code-only contains no weights. Artifact location/release policy must be agreed before branch handoff. |
| Clear prediction function and Backend input/output | ML-side contract frozen; downstream adapter missing | `KArSL502Predictor` methods and `backend_contract_502.md` define actual KArSL Python I/O. The existing repository's `(T,126)` continuous-sign contract is incompatible. No CV adapter or HTTP route is evidenced. |
| Metrics and model comparison report | Prepared with limits | Validation GRU versus CNN; restricted 500-true-class primary test; calibration, abstention, and timing in `evaluation_report_502.md`. |
| README, dependencies, usage | Prepared; clean training replay incomplete | README and `requirements_502.txt`; inference archive tested after fresh extraction. Full training from a clean checkout requires external Pose data/feature shards and uses Colab-oriented default paths. Dependencies are compatible ranges, not a complete environment lock. |
| Simple loading and prediction tests | Verified within their scope | Private archive: 2 passed, 1 skipped for absent real Pose path. Code-only: 1 passed, 2 skipped for private artifacts. A separate Colab real `.pose` decode/prediction passed. These do not test OOV or video/CV robustness. |
| Separate organized folder and identical training/inference preprocessing | Prepared in archives; repo handoff pending | `SIGNOVA_KArSL_502/` archive root and frozen source SHA-256 `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c`. Predictor verifies that file before loading; no branch or PR has been made. |

## Plan v1.1: what this track does and does not satisfy

| Plan item | KArSL evidence / limit |
| --- | --- |
| Dataset QC and fixed splits | 75,515 source entries, technical QC, signer-disjoint train/validation/test, version fingerprint, duplicate/conflict audit. All 502 label IDs and source text fields now match the original KArSL label spreadsheet. Sign-to-label semantics, original Pose/video provenance, and redistribution/model-release rights remain unresolved. Technical PASS is not label approval. |
| Temporal baseline and fair alternative | GRU502 and CNN502 used the same PASS train/validation data. CNN selected by validation macro F1: 0.351794 versus GRU 0.342525. |
| Frozen evaluation and detailed metrics | Selected CNN primary test: 24,536 PASS samples, 500 evaluated true classes, 502 output classes; 52.5187% accuracy, 0.485208 macro F1, 74.2908% top-5. Source classes 27/28, REVIEW entries, and prior exposed paths were excluded under a written protocol. Per-class and confusion-matrix reports exist in private Drive but are not included in the compact handoff archive. This is not a clean full-502-class score. |
| Calibration, abstention, OOV | Validation temperature fitted; no candidate threshold met the exploratory acceptance rule. Active acceptance coverage is 0% and accepted accuracy undefined. Unknown/OOV evaluation set and detection are missing. |
| Inference performance | Ready-made feature latency and checkpoint size recorded; no video-to-response or HTTP latency. One real source Pose compatibility smoke test passed; it does not validate the prediction's correctness. |
| Reproducibility and experiment tracking | Checkpoints, scripts, notebook, report, partial machine-readable summaries, seed and run information exist. Clean checkout full train/eval, exact dependency lock, shared experiment-log CSV, and reviewed checkpoint-location manifest remain to be completed or linked. |
| Plan's primary ESL and secondary WLASL objectives | Not covered by this KArSL track. The plan explicitly lists KArSL as a separate Arabic-language research reference, not Egyptian Sign Language. The team must assess ESL and ASL handoffs separately. |
| Gate E: approved handoff to CV | Not passed. Dataset rights/provenance, labels, source-compatible CV extraction, the distinct `(32,150)` versus `(T,126)` interface, and project-lead review remain unresolved. Production translation claims are unsupported. |

## Decision before a public branch handoff

Use the frozen KArSL Python contract in `backend_contract_502.md` as a separate research path. CV still needs a validated source-compatible producer, and Backend still needs an actual service adapter; neither is a prerequisite to documenting this ML-side contract. Decide controlled checkpoint access and rights, and state clearly that KArSL is a non-ESL isolated-sign reference. A code-only branch can carry code, tests, notebook, reports, and a private-artifact manifest after repository path conventions are reconciled. Do not commit the private inference archive as a public repository file without rights review.

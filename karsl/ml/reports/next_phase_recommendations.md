# KArSL-502 next-phase recommendations

The code-only KArSL ML branch has standalone preprocessing/training/evaluation/prediction, model comparison and notebooks, reproducible CPU dependencies, smoke tests, a controlled checkpoint manifest, a research CV adapter and the provisional Backend/NLP candidate schema. The project lead can review it as an isolated Arabic Sign Language research baseline.

1. Give authorized teammates controlled access to the private selected CNN checkpoint, calibration and source labels using `checkpoints_manifest.md`. Keep full feature shards and original media separate from the public code branch.
2. Backend can implement an explicit research-preview path using `cv_nlp_development_contract.md`; NLP must retain `confirmed=false` on candidate-derived text. The original predictor still returns `uncertain` on every input.
3. CV can test the pinned Holistic adapter against the existing original KArSL recordings. Broaden sample identity and feature parity checks beyond the selected pilot pairs, then freeze a video evaluation protocol before reporting camera-input accuracy.
4. If confirmed-sign acceptance is a product goal, train/evaluate a new version with valid validation-only acceptance criteria, and evaluate genuine unknown signs separately. The frozen test scores must not be used to choose a threshold. Qualified semantic/source rights and project-lead review remain separate from code correctness.

No additional ESL/WLASL work is part of this KArSL handoff. The code does not claim end-to-end live camera translation or OOV rejection.

# KArSL original-frame to Pose pilot

**Historical diagnostic:** This report preserves the preselected pilot results and its limits. The branch's `../integration/karsl_cv_adapter.py` was rechecked on 15 original RGB frames with the frozen preprocessor, but no fresh independent video-input accuracy or OOV validation has been established. The code-only branch can be reviewed separately from publication of source media/weights.

Date: 2026-10-25. Scope: KArSL isolated signs and the frozen 502-output CNN representation. This is an extraction and identity diagnostic, not a production acceptance test.

## Result

Twelve candidate original-frame / third-party Pose pairs were checked across source classes 1, 2, 3, 10, 50, and 70 and signers S01, S02, and S03. All twelve original frame counts matched the paired Pose counts. With MediaPipe Holistic 0.10.21 run independently on the original 256 × 256 JPEG frames, the available left-hand landmarks were near-identical to the Pose file coordinates. The first-pass full-model/static-image setting exposed a material body and feature shift. A comparison on S03 training examples favored the heavy pose model with video tracking (`model_complexity=2`, `static_image_mode=False`); three S02 validation pairs also had small feature differences. This follow-up was motivated after seeing the initial S01 test-exposed diagnostic failure, so it is exploratory despite not selecting the setting by S01 numerical scores. It strengthens the sample-level candidate correspondences, while the earlier one-pair overlay conflict remains to be reconciled.

The frozen 32 × 150 preprocessing function was applied unchanged to the new CV landmarks. In the selected heavy/tracking diagnostic, visibility-mask agreement ranged from 0.934375 to 1.00, and mean absolute difference across all 4,800 feature values ranged from 0.000741 to 0.064446 (median 0.002432). The larger S01 exceptions remain important for model prediction stability.

The unchanged SignCNN502 checkpoint and validation-only temperature calibration were loaded from the private package with PyTorch 2.11.0+cpu. With the first-pass full-model/static setting, the Pose-feature and original-frame/CV-feature paths agreed on top-1 in **6 of 12** pairs, with mean top-5 overlap **3 of 5** and mean Jensen–Shannon divergence **0.146235 nats**. With the heavy/tracking setting selected on S03 and checked on S02, top-1 agreement improved to **11 of 12**, mean top-5 overlap to **4.583 of 5**, and mean divergence fell to **0.008114 nats**. The remaining changed top-1 was two-handed `sign_0003_s1_sess1_001.pose`: source top-1 class 3 versus CV top-1 class 132. All 24 heavy/tracking responses remained `uncertain` under the unchanged disabled-acceptance policy. The two `karsl_cv_model_consistency_12pairs_*.json` files record each pair's ranks, confidence, and divergence. These deliberately selected, historically test-exposed examples must not be turned into a new accuracy estimate or used to tune the frozen model. Eleven-way agreement is promising diagnostic evidence, not proof of general video-input equivalence.

| Candidate Pose file | Signer | Frames | Body median | Left-hand median | Feature mask agreement | Feature MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sign_0001_s1_sess1_001.pose` | S01 | 15 | 0.010362 | 0.000074 | 0.98125 | 0.024031 |
| `sign_0001_s1_sess1_002.pose` | S01 | 26 | 0.009493 | 0.000100 | 0.984375 | 0.020176 |
| `sign_0002_s1_sess1_001.pose` | S01 | 18 | 0.008285 | 0.000054 | 1.00000 | 0.006339 |
| `sign_0010_s1_sess1_001.pose` | S01 | 20 | 0.012641 | 0.000070 | 0.934375 | 0.072171 |
| `sign_0050_s1_sess1_001.pose` | S01 | 29 | 0.007836 | 0.000286 | 0.96000 | 0.041301 |
| `sign_0070_s1_sess1_001.pose` | S01 | 14 | 0.011613 | 0.000033 | 0.94000 | 0.060089 |
| `sign_0003_s1_sess1_001.pose` | S01 | 24 | 0.012582 | 0.000066 | 0.97375 | 0.043126 |
| `sign_0001_s2_sess2_001.pose` | S02 | 13 | 0.008405 | 0.000397 | 0.92000 | 0.079601 |
| `sign_0010_s2_sess2_001.pose` | S02 | 11 | 0.020588 | 0.000192 | 0.96000 | 0.052550 |
| `sign_0050_s2_sess2_001.pose` | S02 | 16 | 0.008988 | 0.000412 | 0.94750 | 0.055609 |
| `sign_0001_s3_sess3_001.pose` | S03 | 24 | 0.015429 | 0.000240 | 0.960625 | 0.045468 |
| `sign_0003_s3_sess3_001.pose` | S03 | 24 | 0.017932 | 0.000154 | 0.961875 | 0.043316 |

The table above shows the initial full-model/static-image baseline; it is retained to explain the failure and setting change. The selected heavy/tracking measurements for all twelve pairs are in `karsl_video_pose_pilot_12pairs.json` under `heavy_tracking`. For the two-handed S01 class-3 pair, the selected setting had body median 0.002190, left-hand median 0.000041, and right-hand median 0.001646 of image width, but 2.625% of feature visibility masks still disagreed.

Distances compare the same indexed landmarks on corresponding frames after both coordinate systems are expressed on a 1000 × 1000 normalized source image. The hand median uses only jointly visible landmarks. One two-handed pair covered both hands; most checked pairs did not. The first available PASS candidates were deliberately selected from the mapping by source order; this is a small diagnostic sample, not a random or independent audit. Pair IDs remain `CANDIDATE_UNVERIFIED` for the whole dataset until broader checks and the earlier overlay discrepancy are resolved.

## Reproduction

The source artifacts were the original KArSL-502 Drive archives `01/test/0001-0070.7z`, `02/test/0001-0070.7z`, and `03/test/0001-0070.7z`, and the pinned Kaggle source `tfmohamedyahia/kars502-pose-75/versions/1`. The archive SHA-256 values in signer order were `6d9f04a4fd9bf82bd7de92fb5861f59ba22ff291a2d6da17f83d2ea5250eaf64`, `55699d206944f4719220580c6999c91fe60fb2aa42820f8b5f2d818b89d7f4d1`, and `b8df7ecde598f99f6f67433333690b3aa9c922dc76086a0369de3e6ce88d61ba`. The frozen `preprocessing.py` SHA-256 was `4ba3f84a0157fd217a6fbafbcbcf933a3e94ddb3e03fe35d0950d20689203c1c`.

Run with Python 3.12, `mediapipe==0.10.21`, `pose-format==0.15.0`, `py7zr`, NumPy 1.26, and OpenCV 4.11. Each source sample is an archive directory of numbered JPEG frames. `karsl_video_pose_probe.py` extracts those frames, decodes the candidate `.pose`, compares body and hand points and invokes the frozen preprocessing on both. `karsl_video_pose_pilot_12pairs.json` contains every pair's results and the precise original member and Pose name. The original images, third-party Pose files, and feature arrays are excluded from the pilot package.

`karsl_cv_adapter.features_from_rgb_frames` applies MediaPipe Holistic with the selected heavy/tracking settings, maps 33 body + 42 hand landmarks into the trained 543-slot layout, leaves unused face slots empty, and calls the exact frozen `preprocess_pose` function. `predict_rgb_frames` passes the resulting 32 × 150 features and preprocessing version to the existing `KArSL502Predictor.predict_features` function. Its input is a list of RGB `uint8` frames of one isolated sign, and its output is the existing uncertain-result dictionary. It does not accept the repository's separate 126-channel continuous-sign feature stream. This adapter is still experimental and is not an approved backend/video integration.

The two adapter tests passed locally, including a real original-frame extraction, feature shape/finite checks, and a mock prediction contract check. The full CNN also loaded and ran on all twelve pairs. Earlier CNN validation/test metrics remain metrics on third-party Pose features, not performance claims for this video adapter.

## Frozen development follow-up across additional classes

Before extracting additional frames, `karsl_cv_development_manifest_20.json` froze ten source class IDs (5, 6, 15, 20, 27, 28, 35, 40, 45, 60) and the first PASS candidate for each of S03 (the model's training signer) and S02 (its validation signer). The manifest SHA-256 is `f3140942718bf93a2516f42b5984916209fbea13113bba091297c8e88ea10e57`. No S01 test samples were included in this follow-up. The model, calibration, heavy/tracking extraction, and preprocessing were unchanged.

The video and Pose branches agreed on top-1 in **19/20** pairs: **10/10 S03** and **9/10 S02**. Their mean top-5 overlap was **4.4/5** and mean full-distribution Jensen–Shannon divergence **0.013314 nats**. The median feature MAE was 0.004824 for S03 and 0.008164 for S02; median visibility-mask agreement was 0.99625 and 0.99250 respectively. The S02 class-20 pair changed top-1 from source-Pose class 9 to video class 38; its two top-5 sets contained the same five classes. All pairs remained `uncertain`. `karsl_cv_development_consistency_20.json` preserves the per-pair evidence. Classes 5/6 and 27/28 remain source-label review cases; agreement in a class ID cannot settle their semantic meaning.

These are model training/validation signer examples with candidate video/Pose correspondence, not a fresh video-input accuracy test. The earlier 12-pair pilot was exploratory and contained historically test-exposed S01 records. None of these consistency numbers should be reported as the recognition accuracy of a deployed camera pipeline.

## Remaining KArSL work before release

1. Expand training-only video/Pose matching beyond the first 70 source classes and to more both-hand sequences; investigate the S01 class-3 decision mismatch, the S02 class-20 mismatch, and the prior overlay discrepancy. Freeze an adapter version and QC protocol without choosing more settings using the historically test-exposed S01 diagnostic pairs.
2. Evaluate the exact frozen video extraction path on genuinely fresh held-out original videos before stating video-input accuracy. If the residual feature shift changes decisions materially, extract new train and validation features and train a separately versioned model; a fresh final test protocol is required.
3. Obtain rights and derivative-use clarification from the original source and conduct qualified semantic review of disputed labels. The 502-row publisher mapping has been transcribed exactly, but the sign meanings have not been independently validated.
4. Collect genuine consented unknown-sign examples and evaluate an OOV/abstention policy. The current CNN policy accepts none of its predictions; `top_k` is advisory and must not be sent to NLP as a recognized word.

No GitHub branch or public dataset/checkpoint upload is authorized by this pilot result.

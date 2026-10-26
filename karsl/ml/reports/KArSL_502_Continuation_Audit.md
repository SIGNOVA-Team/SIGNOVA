# KArSL 502 continuation audit

Review date: 2026-10-26. Evidence inspected: uploaded `07_KArSL_Colab_Verification (3).ipynb`, `08_KArSL_502_GRU_Training (1).ipynb`, and `files (1).zip`, plus the frozen five scripts and existing 502/CV handoff packages. This is a technical evidence review, not a new full training run.

## Verified from the uploaded notebook outputs

| Item | Recorded outcome | Consequence |
| --- | --- | --- |
| Pinned source download | 75,515/75,515 nonempty Pose paths; frozen manifest matched | All expected paths were present in that Colab runtime. This does not prove semantic or video identity. |
| Full-file exact-duplicate audit | 75,515 SHA-256 hashes; 75,464 unique; 51 duplicate pairs | No exact-byte duplicate group crossed the frozen train/validation/test split in any of three folds. Near-duplicates were not checked. |
| Conflicting labels | 50 of 51 pairs carry class IDs 27 and 28, within signer-1 test in fold 1 | Do not silently train/evaluate these as independent correct labels. The frozen CNN test excluded both classes' 150 samples. A separate class-395 duplicate pair in validation had one label. |
| Checkpoint hashes | CNN `f9f0c552…5d36de2254`; GRU `ba179704…2e445281af6ae577da`; linear baseline `a2af7582…20e1ae8ab2b9eed7e5` | The notebook log matches the supplied checkpoint manifest. Hash equality was checked in the Colab runtime, not by independently fetching all Drive files here. |
| Calibration config hash | `448fcd35f55e26eede15fd78b7f09ed3d21d40a2f13fe9e0e1cff0ec9fdda5e4` | The previously blank manifest field is filled in the uploaded documentation. |
| Model selection | CNN validation macro F1 0.351794 vs GRU 0.342525 | CNN was selected on validation. Test was not used for this comparison. |
| Test scope | 24,536 PASS examples, 500 true classes, 502 outputs | 52.5187% top-1, 0.485208 macro F1, 74.2908% top-5; historical test exposure and two excluded classes constrain the claim. |

The uploaded ZIP contains **seven documentation and CSV files, no Python scripts or checkpoints**. The five exact scripts in the existing 502 code-only package have SHA-256 hashes matching the previously frozen script hashes recorded in `checkpoints_manifest.md`. The current code-only archive carries the scripts and a smoke test; the private inference archive additionally carries the checkpoint, calibration file, and publisher labels. A full clean-checkout retraining run still needs the controlled Pose data, manifests, feature shards and original Drive experiment metadata. Relocating or editing frozen scripts changes their hashes and may invalidate cache/protocol identity checks, so preserve the originals and test any wrapper separately.

## Video/CV continuation

The original-frame adapter compared candidate videos with third-party Pose features using the frozen CNN and preprocessing. Across three existing pilots, there were 52 distinct original RGB-frame samples: video/Pose top-1 agreed on 49, while the video top-1 equaled the unverified source folder class in 21. The two frozen 20-pair follow-ups each agreed on top-1 in 19 pairs. S01 had historical test exposure. In its 20-pair follow-up, source Pose top-1 equaled the folder class in only 4 pairs. These are selected original-video diagnostics, not a fresh-video recognition benchmark. All outputs remained `uncertain` because the acceptance policy is disabled. No new recordings are needed to inspect the original KArSL frames; a controlled fresh-video protocol and evaluator are available for a later independent generalization claim.

Error attribution on the saved pairs found only three video-versus-Pose top-1 changes. S02 and S03 share ten reviewed-in-code folder class IDs in the development follow-up: video top-1 equaled the folder class on 4/10 S02 and 9/10 S03; after excluding disputed IDs 5, 6, 27 and 28, the counts are 3/6 and 5/6. Each class has only one selected example per signer. Six top video candidates had confidence at least 0.5 but differed from their source folder IDs, including a disputed class-27 record. The model must not turn confidence into an accepted sign on this evidence. These patterns justify checking source pairing and labels before deciding whether separately versioned model training is needed; changing the adapter or threshold against the historical test material would be unsound.

The private original-video/Pose candidate mapping has 75,514 rows. One pair has a frame-count discrepancy (source 41, Pose 40), and 33 S01 rows have original class label 6 paired with Pose sign ID 5. The mapping itself remains entirely unverified at the instance-identity level. These findings narrow the expert review request; they do not license automatic relabeling or invalidate the separately frozen Pose test as a numerical computation.

## Remaining gates before a public branch or camera handoff

1. Resolve rights for original videos, third-party Pose derivatives, metadata and model weights. A Kaggle license field alone does not settle upstream permissions. No inquiry was sent on the user's behalf.
2. Have a qualified reviewer resolve the 27/28 duplicate-label conflict, class-5/6 assignments, proposed gloss edits and representative video/Pose identity. Source text matches the spreadsheet, but sign meanings are unverified.
3. Obtain genuinely new, consented and reviewed known-sign videos from an independent signer and, if OOV behavior is required, genuine labeled unknown signs. Freeze their manifest before inference and report candidate accuracy separately from accepted coverage. The current acceptance policy has 0% coverage; OOV safety cannot be claimed from that alone.
4. Reconcile the existing repo layout and relative paths with the frozen script hashes. Run code-only and private inference checks after extraction, and record controlled artifact pointers. Full training replay remains dependent on private shards.

No public repository upload, checkpoint publication, production acceptance, or sign-language translation claim follows from this audit.

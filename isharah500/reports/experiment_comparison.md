# Isharah-500 SI experiment comparison

The development rows use the SI train/dev protocol and real tracked keypoints.
The final held-out test row was evaluated only after the checkpoint was frozen;
it was not used for model selection. The epoch 15 Exact Match shown in the
training history is retained as historical context only; the corrected error
analysis is the authoritative dev value for that checkpoint.

## Aggregate comparison

| Experiment | Training | Preprocessing | Dev CTC loss | Dev GER | Dev Exact Match | Source |
|---|---:|---|---:|---:|---:|---|
| SI pilot | 5 epochs | Original preprocessing | 4.9031 | 0.8902 | 0.00 | checkpoints/si_pilot/isharah500_si_training_history.json |
| Hand-centric | 5 epochs | Hand-centric normalization | 4.1717 | 0.8548 | 0.024 | checkpoints/si_hand_centric_5ep/isharah500_si_training_history.json |
| Hand-centric resume | 10 total epochs | Hand-centric normalization | 3.3536 | 0.7722 | 0.12 | checkpoints/si_hand_centric_resume_10ep/isharah500_si_training_history.json |
| Hand-centric resume | 15 total epochs | Hand-centric normalization | 3.0335 | 0.7135 | **0.07 corrected** | Corrected error analysis |

The epoch 15 training history recorded Exact Match 0.28. That value was
produced by the pre-refactor batch metric aggregation and is incorrect for the
500-sample evaluation. The independent canonical analysis evaluated all 500 SI
dev samples with correct target boundaries and reports 35 exact matches,
35 / 500 = 0.07. GER agrees because the aggregate edit counts were already
consistent.

## Corrected epoch 15 error profile

The trusted checkpoint is:

    checkpoints/si_hand_centric_resume_15ep/isharah500_si_best.pt

The corrected SI dev report contains:

| Metric | Value |
|---|---:|
| Samples | 500 |
| Reference glosses | 2,094 |
| Substitutions | 840 |
| Deletions | 618 |
| Insertions | 36 |
| Edit errors | 1,494 |
| GER | 0.713467 |
| Exact matches | 35 |
| Exact Match | 0.07 |
| Mean prediction length | 3.024 |
| Mean target length | 4.188 |
| Deletion rate | 0.2951 |

The GER identity is:

    (840 + 618 + 36) / 2,094 = 0.713467...

## Interpretation

Hand-centric normalization improved the five-epoch dev GER from 0.8902 to
0.8548, and continued resumed training reduced corrected epoch 15 GER to
0.7135. Exact Match also improved relative to the hand-centric five-epoch
run, but the corrected epoch 15 value is 0.07, not the historical 0.28.

The model still underpredicts sequence length. Deletions dominate the corrected
epoch 15 error profile, especially on longer target sequences. Frequent
alignment-based substitutions include انا to هو, هو to الان, and رغبه to هو.
Prefix beam search did not improve the focused comparison, so greedy decoding
remains the current baseline.

These results support the hand-centric preprocessing choice for this baseline.
The final held-out SI test GER is higher than the corrected dev GER and Exact
Match is lower, indicating a measurable generalization gap. The test result is
reported for handover and does not justify a production readiness claim.

## Research-only experiments after the frozen baseline

These experiments were not used to replace the selected checkpoint. They are
documented for provenance and must not be treated as production model results.

| Experiment | Calibration/training split | Evaluation split | Result | Decision |
|---|---|---|---|---|
| Target-length-weighted CTC v2 | SI Train | SI Dev | GER `0.747373`, Exact Match `0.024` at epoch 15 | Rejected; worse than frozen baseline |
| Blank-offset calibration `-2.0` | SI Train calibration subset (100) | SI Dev (500) | GER `0.723496`, Exact Match `0.052`, substitutions `964`, deletions `441`, insertions `110`, mean prediction length `3.526` | Rejected; fewer deletions but higher total errors |
| Global Position Features v1 | Not trained | Not evaluated | Feature schema and commands only; no validated metrics | Untrained research experiment |

The weighted CTC result is detailed in
`reports/experiment_target_length_weighted_ctc_15ep_v2_analysis.md` and the
blank-offset result is recorded in
`reports/blank_offset_calibration_si_dev.json`. The Global Position Features
definition is in `reports/experiment_global_position_v1.md`. None of these
experiments changes the frozen 63-feature checkpoint or its inference contract.

## Final held-out SI test evaluation

The epoch 15 checkpoint was frozen before this one-time evaluation. The saved
report is
`reports/si_test_evaluation_epoch15.json`. It uses the checkpoint vocabulary
and preprocessing configuration, strict real keypoint loading, and greedy CTC
decoding. It was not used to select or tune the model.

| Metric | SI test value |
|---|---:|
| Samples | 2,000 |
| Reference gloss tokens | 8,376 |
| Substitutions | 3,326 |
| Deletions | 3,041 |
| Insertions | 94 |
| Total edit errors | 6,461 |
| GER | 0.771370582617001 |
| Exact matches | 82 |
| Exact Match | 0.041 |
| CTC-valid samples | 2,000 |
| CTC-rejected samples | 0 |

The test GER identity is:

    (3,326 + 3,041 + 94) / 8,376 = 0.771370582617001

The model remains a hand-only baseline with short sequence predictions and
substantial deletion errors. These results establish the frozen checkpoint's
SI test measurement, not production-level accuracy or generalization to US.

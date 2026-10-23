# Isharah-500 SI Error Analysis

- Split: `dev`
- Samples: `500`
- Checkpoint: `E:\DEPI Project\SIGNOVA_ESL-isharah500\isharah500\checkpoints\si_hand_centric_resume_15ep\isharah500_si_best.pt`

## Aggregate metrics

| Metric | Value |
|---|---:|
| GER | 0.7135 |
| Exact match | 0.0700 |
| Substitutions | 840 |
| Deletions | 618 |
| Insertions | 36 |
| Mean prediction length | 3.024 |
| Mean target length | 4.188 |
| Deletion rate | 0.2951 |

## Target length groups

| Target length | Samples | GER | Exact match | Deletion rate |
|---:|---:|---:|---:|---:|
| 1 | 7 | 0.8571 | 0.5714 | 0.0000 |
| 2 | 38 | 1.0000 | 0.1053 | 0.0000 |
| 3 | 116 | 0.5747 | 0.1897 | 0.0546 |
| 4 | 163 | 0.7178 | 0.0307 | 0.2439 |
| 5 | 99 | 0.6909 | 0.0000 | 0.3394 |
| 6 | 41 | 0.7724 | 0.0000 | 0.4837 |
| 7 | 26 | 0.7527 | 0.0000 | 0.5330 |
| 8 | 6 | 0.8750 | 0.0000 | 0.6250 |
| 9 | 2 | 0.9444 | 0.0000 | 0.6111 |
| 11 | 2 | 0.7273 | 0.0000 | 0.6818 |

## Valid-hand coverage groups

| Coverage | Samples | GER | Exact match | Deletion rate |
|---|---:|---:|---:|---:|
| 25-49% | 1 | 1.0000 | 0.0000 | 0.2500 |
| 50-74% | 280 | 0.7468 | 0.0607 | 0.2793 |
| 75-99% | 219 | 0.6745 | 0.0822 | 0.3133 |

## Most frequent aligned substitutions

- `انا` → `هو`: 9
- `هو` → `الان`: 6
- `رغبه` → `هو`: 5
- `هو` → `سوال`: 5
- `ولد` → `هو`: 5
- `ذهاب` → `هو`: 4
- `سوال` → `هو`: 3
- `مدرسه` → `سوال`: 3
- `هو` → `لا`: 3
- `هو` → `وقت`: 3
- `اسم` → `وقت`: 2
- `اكل` → `سوال`: 2
- `اكل` → `شهر`: 2
- `الان` → `هو`: 2
- `انتظار` → `سوال`: 2
- `بحر` → `كوفي`: 2
- `برد` → `صباح`: 2
- `بيع` → `هو`: 2
- `تاريخ` → `هو`: 2
- `تفاحه` → `هو`: 2

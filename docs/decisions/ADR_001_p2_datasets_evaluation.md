# ADR-001: Evaluation, Policy, and Integration Strategy for P2 Datasets (How2Sign & ArabSign)

**Status:** Accepted  
**Date:** 2026-10-11  
**Deciders:** SIGNOVA Dataset & Machine Learning Engineering Workstreams  
**Consulted:** ML Lead, Data Engineer, NLP Engineer, Project Lead  

---

## 1. Context & Problem Statement

The SIGNOVA project aims to build an assistive communication prototype connecting sign language, speech, and text. The core mission is **Egypt-first**: Egyptian Sign Language (ESL) paired with Egyptian Arabic as the primary track, with American Sign Language (ASL) and English as the secondary track.

The team has been tasked with evaluating and preparing two Priority 2 (P2) continuous sign language datasets:
1. **How2Sign** (Continuous ASL, ~80 hours, multimodal)
2. **ArabSign** (Continuous ArSL, 10 hours, 9,335 samples, Kinect multimodal)

We must define the precise boundaries, data governance rules, split policies, and technical integration strategies for these two datasets to avoid scope creep, linguistic mislabeling, and data leakage.

---

## 2. Decision Drivers

1. **Linguistic Fidelity:** Egyptian Sign Language (ESL) is linguistically distinct from Gulf/Saudi Sign Language (ArSL). Unverified data mixing damages model validity.
2. **Phased Scope Discipline:** Core Phase 1 and Phase 2 focus on verified isolated-sign baselines (`P0 — ESL` and `P1 — WLASL`). Continuous sentence translation is an advanced, non-blocking goal.
3. **Generalization & Leakage Prevention:** The SIGNOVA quality principle mandates group-aware, signer-independent splitting whenever signer metadata allows.
4. **Compute Feasibility:** Raw video across both datasets exceeds 2.2 TB, which is impractical for immediate CI/CD and rapid prototyping.

---

## 3. Decisions & Policy Rules

### Decision 1: P2 Datasets Are Exploratory References, Not Phase 1 Blockers
- How2Sign and ArabSign are designated as **Phase 3+ continuous research references**.
- Neither dataset shall delay or alter the acceptance gates for Phase 1/2 isolated sign recognition baselines.

### Decision 2: ArabSign Must Never Be Relabeled as Egyptian Sign Language (ESL)
- ArabSign represents Saudi / Pan-Arab Sign Language recorded at KFUPM.
- **Rule:** ArabSign data and annotations shall strictly retain the label `sign_language: ArSL`. They must never be merged into ESL training pools or presented to users as Egyptian Sign Language.
- ArabSign will be utilized to benchmark continuous Arabic sentence decoding architectures (e.g., Encoder-Decoder, CTC), studying the transition from ArSL gloss syntax to standard Arabic text.

### Decision 3: Adoption of Group-Aware Signer-Independent Splits
- **ArabSign:** We establish a canonical **Signer-Independent Split**:
  - **Train:** Signers 01, 02, 03, 04 (6,323 samples, 67.7%)
  - **Validation:** Signer 05 (1,500 samples, 16.1%)
  - **Test:** Signer 06 (1,512 samples, 16.2%)
  - *Result:* Zero signer overlap (0% leakage), establishing an honest benchmark for generalization to novel signers.
- **How2Sign:** We catalog the official topic-based split while formally documenting that 5 signers overlap across train, val, and test. Any claim of generalization to unseen signers must use a custom signer-partitioned evaluation set.

### Decision 4: Representation-First Feature Strategy
- ML baselines exploring P2 continuous data shall prioritize lightweight keypoint/skeleton representations:
  - How2Sign: 2D OpenPose Body-Face-Hands keypoints (80 GB tarball).
  - ArabSign: 25 3D Kinect skeleton joint coordinates (~5 GB) and MediaPipe extractions.
- Full multi-terabyte raw video downloads are deferred until continuous modeling is formally scheduled in Phase 3.

---

## 4. Consequences & Impact

### Positive:
- **Zero Scope Creep:** The ML and CV teams remain focused on delivering the primary ESL isolated baseline without distraction.
- **Linguistic Integrity:** Avoids deceptive claims about ESL coverage using Gulf sign data.
- **Reproducible Pipeline:** All 35,263 How2Sign sentences and 9,335 ArabSign samples are versioned in canonical schemas with automated tests.
- **Rigorous Evaluation:** Eliminates optimistic bias from signer leakage by establishing a true signer-independent benchmark on ArabSign.

### Negative / Trade-offs:
- Continuous sentence translation will not ship in the initial v0.1 release (as explicitly planned in the 8-Week Roadmap).
- Researchers working on continuous ASL translation must account for signer overlap on the official How2Sign benchmark.

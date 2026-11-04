# Results Synthesis - Robustness of Document Layout Analysis Under Compound Degradation

**Status:** results for the first three detectors, 2026-11-04. The rerun for 8 detectors is pending. Source of every number: the phase notes
(`docs/phases/P*/NOTES_*.md`), which cite the result files under `RDLD_data/results/`.
Workstream B (Khizar) notes are on branch `khizar`; Workstream A (Alishba) notes on branch `alishba`.

## Setup in one paragraph
300 DocLayNet v1.2 validation pages (50 × 6 document categories, 3,530 boxes, 11 classes). Three released
detectors: YOLO (`hantian/yolo-doclaynet`, CNN one-stage), heron (`docling-layout-heron`, RT-DETR v2),
egret (`docling-layout-egret-medium`, D-FINE). Eight image-only RoDLA degradations × 3 severities: 24
singles + 84 same-severity pairs, applied in a fixed physical order with common random numbers →
32,400 images. 519 prediction sets in total (incl. reversed-order and second-seed replicates). Inference
on a 1,000-resample paired page bootstrap with one shared resample index; BH-FDR. Downstream: one
Tesseract pass per page, text read through GT vs predicted regions.

## Clean baseline (P1)

| | YOLO | heron | egret |
|---|---|---|---|
| mAP@[.5:.95] [95% CI] | **68.8** [64.0, 72.5] | 64.2 [60.4, 68.4] | 61.5 [58.3, 65.8] |
| mAP@.5 | 87.6 | **89.1** | 86.1 |
| cross-label duplicates (score ≥ 0.3) | 3.3% | 28.0% | 26.3% |
| correct regions at τ = 0.5 (P4) | 87.9% | 87.6% | 88.0% |
| WER of the region pipeline (P5) | **0.022** | 0.112 | 0.102 |

## RQ1 - Do degradations compound?

**Answer: mostly near-additive, but systematically worse than multiplicative, growing with severity, with
a small set of strongly super-additive pairs; compounding is created by the detector, not the pixels.**

1. Negative control (21 `defocus@1` pairs): interaction exactly 0.00 - the apparatus is exact. *(P3-K)*
2. Additive null: 50–60 of 77 pairs per detector n.s. Multiplicative null: median I_mult +0.56 (YOLO),
   +1.09 (heron), +1.48 (egret); egret s1 → s3: −0.05 → +1.63 → +4.33. *(P3-K)*
3. 18 detector × pair cells super-additive under **both** nulls - speckle with anything, and blur
   (defocus / vibration) with ink or speckle. YOLO, ink_holdout@3 + defocus@3: singles 0.9 and 2.0
   points, pair 12.8. *(P3-K)*
4. Singles predict pairs well (ρ ≈ 0.98); the multiplicative rule is best (MAE 1.8–2.7 points) but
   under-predicts on average for every detector. *(P3-K)*
5. Mechanism: only 18% of pairs compound at pixel level; for both transformers pixel and detector
   interactions are **negatively** correlated (ρ = −0.26, −0.33). *(P3-A)*
6. Robust to the seed: seed noise ≈ 0.28–0.41 × page noise, no sign flips. *(P4-K)*
7. Robust to order **except texture**: texture pairs median order effect 4.7 points (up to 23.4 for YOLO);
   other pairs 0.4. *(P3-A)*

## RQ2 - Does mAP predict downstream quality?

**Answer: partially. mAP ranks detectors correctly on WER most of the time, but it is blind to duplicated
output, understates clean-data gaps, and the proposed UW-mAP does not improve on it.**

1. Within condition, ρ(mAP, WER) = −0.83; best detector by mAP = best by WER in 81% of 37 conditions
   (full order 59%); disagreements concentrate under speckle. *(P5-K)*
2. ρ(mAP, text precision) within condition = **−0.02**; ρ(duplicate share, text precision) = **−0.85**.
   The transformers' label hedging raises mAP (P1: class-agnostic dedup *lowers* their mAP by 4.3–4.5
   points) and duplicates text downstream. *(P1-K, P5-K)*
3. On clean pages a ≤ 7-point mAP gap is a 5× WER gap (0.022 vs 0.10–0.11). *(P5-A)*
4. Failure type predicts downstream symptom: YOLO misses regions (+25.7 pp at s3 pairs) → text recall
   0.78 on pairs; transformers merge regions (+20.5 / +26.6 pp) → recall kept (≥ 0.89), precision and
   order lost. *(P4-A, P5-A)*
5. Threshold: text recall is flat above ≈ 39% mAP retention (slope 0.11) and collapses below it
   (slope 1.99; F = 131.8). *(P5-K)*
6. UW-mAP: robust to its weights (drop ratio 1.09, range 1.00–1.18) but ρ with WER −0.90 vs −0.92 for
   mAP - **no improvement** (negative result). mAP75 tracks WER best pooled (−0.93). *(P5-K)*

## RQ3 - Are detector rankings preserved?

**Answer: no, not under compound degradation - and the "most robust" detector depends on the
robustness metric.**

1. Kendall τ vs clean order: singles 0.69, pairs 0.45. Clean order holds in 18 / 24 singles but only
   46 / 84 pairs; heron > egret > YOLO in 31 pairs. *(P4-K)*
2. Significant reversals (both gaps significant, opposite signs), all YOLO vs egret: 5 singles, **27
   pairs** - the clean loser beats the clean winner. *(P4-K)*
3. Robustness ranking by retention is stable (heron > YOLO > egret on both benchmarks), but the compound
   benchmark nearly doubles the heron–egret gap (3.8 → 6.5 points). *(P4-K)*
4. By RoDLA's mRD the ranking is YOLO (7.8) > heron (8.9) > egret (9.9) - mRD scores absolute error and
   rewards clean accuracy; retention scores relative loss. *(P4-K)*
5. Architectures fail on different things: transformers' drops correlate ρ = 0.98 with each other, 0.90
   with YOLO. YOLO: texture-fragile (≈ 43 points at every severity), blur-robust (defocus s3: 2.0 vs
   12–17), most speckle-fragile (59.9 at s3). *(P2-K, P4-K)*

## Secondary findings
- Class vulnerability is not explained by box size (ρ = −0.28, p = 0.44); for the transformers, **large**
  regions degrade more than small ones on pairs. Formula and Picture drop most; Table and Footnote
  compound most. *(P4-A)*
- Image-quality damage predicts detector damage (ρ(mPE_IQA, D) = 0.84–0.88); PSNR does not
  (−0.36 to −0.65). *(P3-A)*

## Deviations from the proposal (must be stated in the paper)
1. mPE is IQA-only (MS-SSIM + CW-SSIM), not RoDLA's IQA + Faster R-CNN term - model-independent by design.
2. heron/egret have 17 classes (not 12 as in the Sept notes); 6 unmapped classes dropped, counts recorded.
3. `defocus@1` is a pixel no-op in RoDLA's code; used as a negative control and excluded from mRD.
4. Downstream covers severity 2 only (37 conditions).
5. Same-severity pairs only (no cross-severity pairs, no triples).

## Limitations
- 300 pages, one dataset (DocLayNet), three detectors (two share a training pipeline and class space).
- Two composition orders tested, not all; texture results are order-dependent.
- One OCR engine (Tesseract 5.5.3); pages with light-on-photo text are unreadable to it (7 / 300 clean).
- 111 downstream points share conditions; pooled CIs are optimistic (within-condition analysis used as
  the primary test).
- Released weights only; no detector was fine-tuned or augmented.

## Integrity checks that held
Byte-reproducible cache · deterministic detectors · exact negative control (after fixing a precision
mismatch it exposed) · fast bootstrap verified equal to pycocotools · seed replicate · order replicate ·
OCR fail-loud guard (after a silent-empty first run was caught).

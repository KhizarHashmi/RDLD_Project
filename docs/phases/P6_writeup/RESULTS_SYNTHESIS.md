# Results Synthesis - Robustness of Document Layout Analysis Under Compound Degradation

**Status:** results for all 8 detectors, 2026-11-07. Still preliminary because some methodology checks are open
(`docs/phases/P7_extension/NOTES_khizar.md`, Section 7). Source of every number: the phase notes
(`docs/phases/P*/NOTES_*.md`) and the result files under `RDLD_data/results/`.
Workstream B (Khizar) notes are on branch `khizar`; Workstream A (Alishba) notes on branch `alishba`.
Where a result differs from the 3-detector run, the earlier number is given.

## Setup in one paragraph
300 DocLayNet v1.2 validation pages (50 × 6 document categories, 3,530 boxes, 11 classes). Seven released
detectors: YOLO (`hantian/yolo-doclaynet`, CNN one-stage), heron and heron-101 (RT-DETR v2), egret and egret-xlarge
(D-FINE), DETR, Deformable DETR. One fine-tuned by us: RF-DETR (DINOv2 backbone, about one sixth of the training
split). Eight image-only RoDLA degradations × 3 severities: 24 singles + 84 same-severity pairs, applied in a fixed
physical order with common random numbers → 32,400 images. 173 prediction sets per detector (incl. reversed-order and
second-seed replicates). Inference on a 1,000-resample paired page bootstrap with one shared resample index; BH-FDR.
Downstream: one Tesseract pass per page, text read through GT vs predicted regions.

## Clean baseline (P1)

| | mAP [95% CI] | mAP@.5 | cross-label duplicates (score ≥ 0.3) | correct regions at τ = 0.5 | WER of the region pipeline |
|---|---|---|---|---|---|
| YOLO | **68.8** [64.0, 72.5] | 87.6 | 3.3% | 87.9% | **0.022** |
| Deformable DETR | 66.2 [63.4, 69.7] | 87.6 | 7.5% | 82.2% | 0.058 |
| heron-101 | 65.9 [62.4, 70.3] | 88.3 | 22.6% | 85.8% | 0.098 |
| heron | 64.2 [60.4, 68.4] | **89.1** | 28.0% | 87.6% | 0.112 |
| egret-xlarge | 62.3 [58.5, 66.6] | 87.1 | 27.6% | 88.0% | 0.105 |
| egret | 61.5 [58.3, 65.8] | 86.1 | 26.3% | 88.0% | 0.102 |
| DETR | 53.0 [49.3, 57.1] | 77.9 | 0.2% | 83.6% | 0.185 |
| RF-DETR | 48.9 [45.8, 53.7] | 69.8 | 19.6% | 76.2% | 0.084 |

RF-DETR is not like for like because it had far less training data. Title has only 6 boxes on these pages; see P7 notes.

## RQ1 - Do degradations compound?

**Answer: mostly near-additive, but on average worse than multiplicative, with a set of strongly super-additive
pairs. Growth with severity depends on the detector. Compounding is created by the detector, not the pixels.**

1. Negative control (56 `defocus@1` detector × pair cells): interaction exactly 0.00 - the apparatus is exact. *(P3-K)*
2. Additive null: 48–60 of 77 pairs per detector n.s. Multiplicative null: median I_mult positive for all eight:
   +0.21 (Deformable DETR), +0.44 (RF-DETR), +0.47 (DETR), +0.56 (YOLO), +0.85 (heron-101), +1.09 (heron),
   +1.20 (egret-xlarge), +1.48 (egret). *(P3-K)*
3. It grows with severity in the heron and egret families and RF-DETR (egret s1 → s3: −0.05 → +1.63 → +4.33). YOLO, DETR
   and Deformable DETR show no steady rise. *(P3-K)*
4. 43 of 616 detector × pair cells super-additive under **both** nulls (18 of 231 with 3 detectors), none of them from RF-DETR.
   Speckle and blur (defocus, vibration) pairs recur. YOLO, ink_holdout@3 + defocus@3: singles 0.9 and 2.0 points, pair 12.8. *(P3-K)*
5. Singles predict pairs well (ρ ≈ 0.98–0.99); the multiplicative rule is best (MAE 1.0–2.7 points) and under-predicts on
   average for seven of eight detectors. *(P3-K)*
6. Mechanism: only 18% of pairs compound at pixel level. Pixel and detector interactions are negatively correlated for egret
   (ρ = −0.33), heron (−0.26) and egret-xlarge (−0.25); no detector shows a significant positive correlation. *(P3-A)*
7. Robust to the seed: seed noise ≈ 0.28–0.41 × page noise, no opposite significant calls. *(P4-K)*
8. Robust to order **except texture**: texture pairs median order effect 3.1 points (up to 23.4 for YOLO);
   other pairs 0.3. 108 of 224 order effects are significant, median 0.47 points. *(P3-A)*
9. Without Title, medians stay similar and more pairs become significant. *(P7-K)*

## RQ2 - Does mAP predict downstream quality?

**Answer: partially. mAP often picks the best detector on WER, but it is blind to duplicated
output, misses clean-data gaps, and the proposed UW-mAP gives almost no gain.**

1. Within condition, ρ(mAP, WER) = −0.68 (−0.83 with 3 detectors); best detector by mAP = best by WER in 76% of 37 conditions
   (81% with 3). The full order never agrees across 8 detectors, which is not comparable with the 59% for 3. *(P5-K)*
2. ρ(mAP, text precision) within condition = **0.00**; ρ(duplicate share, text precision) = **−0.79**. *(P5-K)*
   Class-agnostic dedup lowers mAP by 4.0–4.7 points in the heron and egret families and 3.0 for Deformable DETR. *(P1-K)*
3. On clean pages small mAP gaps hide large WER gaps: YOLO 0.022 vs 0.098–0.112 for the heron and egret families, within
   7.3 mAP points. RF-DETR has the lowest mAP but the third-lowest WER (0.084). *(P5-A)*
4. Failure type predicts downstream symptom. Severity-3 pairs, change in missed regions: Deformable DETR +28.3, YOLO +25.7,
   RF-DETR +24.3 pp, and these have the lowest text recall on pairs (0.56, 0.78, 0.80). Heron and egret families mostly merge
   regions (+20.5 to +34.0 pp) and keep recall (≥ 0.89). *(P4-A, P5-A)*
5. Text recall against mAP retention has a breakpoint (F = 149.7), but it moved from about 39% retention with 3 detectors to about
   24% with 8. It depends on the detector set, so we should not claim a fixed threshold. *(P5-K)*
6. UW-mAP: robust to its weights (drop ratio 1.06, range 1.02–1.10, new estimator) but ρ with WER pooled −0.87 for both and within
   condition −0.69 vs −0.68. Text recall within condition 0.52 vs 0.46. Plain mAP tracks WER best pooled (mAP75 −0.86). *(P5-K)*

## RQ3 - Are detector rankings preserved?

**Answer: no, rankings break down with damage. But pairing itself does not seem to scramble them, and single and compound
benchmarks give the same robustness order. The choice of robustness metric matters much more.**

This changes the 3-detector reading, which was that pairs scramble rankings.

1. Kendall τ vs clean order, controls excluded: singles 0.67, pairs 0.46, gap 0.21 [0.13, 0.24]. The drop also holds for distinct
   architectures only (0.25 [0.18, 0.28]) and the original three (0.25 [0.05, 0.29]). Seed noise floor τ 0.94–0.96. *(P4-K)*
2. At equal damage, the pair effect on τ is +0.04 [−0.01, 0.06] within retention bands, and +0.05 [−0.00, 0.07] linear. Negative would
   mean pairs scramble more, so we see no sign of it. For distinct architectures: +0.02 [−0.02, 0.07] and +0.04 [0.01, 0.08]. *(P4-K)*
3. Clean order of all 8 holds in 5 of 23 singles and 13 of 77 pairs. For the original three: 17 of 23 and 41 of 77. Top detector on
   pairs: YOLO 37, heron 20, heron-101 13, Deformable DETR 4, egret-xlarge 2, RF-DETR 1. *(P4-K)*
4. Significant reversals (both gaps significant, opposite signs, after FDR): 40 of 460 tests on singles (8.7%), 238 of 1,540 on pairs
   (15.5%). YOLO vs egret alone: 5 singles, 27 pairs. *(P4-K)*
5. Robustness order by retention is the same on single and pair benchmarks (τ 1.0 [0.79, 1.0]): heron, heron-101, RF-DETR, egret-xlarge, YOLO,
   egret, DETR, Deformable DETR. Heron is first on both. Pairs widen gaps (heron–egret 3.8 → 6.5 points). *(P4-K)*
6. By RoDLA's mRD: YOLO 7.8, heron-101 8.8, Deformable DETR 8.8, heron 8.9, egret-xlarge 9.5, egret 9.9, DETR 11.8, RF-DETR 12.2. mRD scores absolute
   error and rewards clean accuracy; retention scores relative loss. The two orders barely agree (τ 0.07). Deformable DETR is last by
   retention and third by mRD; RF-DETR is third by retention and last by mRD. *(P4-K)*
7. Two groups fail alike: heron and egret families correlate ρ = 0.97–0.99 on per-configuration drops, and YOLO, Deformable DETR and RF-DETR
   0.95–0.97. Between the groups ρ ≈ 0.90–0.92. DETR is closer to the heron and egret families (0.92–0.95). YOLO is texture-fragile, and Deformable DETR scores above clean on `defocus@3`. *(P2-K, P4-K)*

## Secondary findings
- Class vulnerability is not explained by box size (ρ = −0.25, p = 0.49). Formula (25.9) and Picture (23.7) drop most at severity 3;
  Table compounds most at severity 3 (+3.10). *(P4-A)*
- Image-quality damage predicts detector damage (ρ(mPE_IQA, D) = 0.78–0.91); PSNR does less well
  (−0.46 to −0.66). *(P3-A)*
- Selection vs test pages: no detector does unusually well on the test pages. *(P7-K)*

## Deviations from the proposal (must be stated in the paper)
1. mPE is IQA-only (MS-SSIM + CW-SSIM), not RoDLA's IQA + Faster R-CNN term - model-independent by design.
2. heron/egret have 17 classes (not 12 as in the Sept notes); 6 unmapped classes dropped, counts recorded.
3. `defocus@1` is a pixel no-op in RoDLA's code; used as a negative control and excluded from mRD.
4. Downstream covers severity 2 only (37 conditions).
5. Same-severity pairs only (no cross-severity pairs, no triples).
6. 3 detectors became 8, one of them fine-tuned by us.
7. RQ3 gained a test of whether ranking changes come from pairing or from damage, added after the first results.

## Limitations
- 300 pages, one dataset (DocLayNet). The 8 detectors are not independent: two heron, two egret and several DETR-family models.
- The test pages come from the DocLayNet validation split, not the official test split. Released detectors may have used them to pick checkpoints.
- Only 6 Title boxes on the test pages. Report mAP with and without Title.
- RF-DETR is not like for like, having about one sixth of the training pages.
- Two composition orders tested, not all; texture results are order-dependent.
- One OCR engine (Tesseract 5.5.3); pages with light-on-photo text are unreadable to it (7 / 300 clean).
- 296 downstream points share conditions; pooled CIs are optimistic (within-condition analysis used as
  the primary test).
- Released weights for seven detectors, none of them fine-tuned or augmented by us.

## Integrity checks that held
Byte-reproducible cache · deterministic detectors · exact negative control (after fixing a precision
mismatch it exposed) · fast bootstrap verified equal to pycocotools · seed replicate · order replicate ·
OCR fail-loud guard (after a silent-empty first run was caught).

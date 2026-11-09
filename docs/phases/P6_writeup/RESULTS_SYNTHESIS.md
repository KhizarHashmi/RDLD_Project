# Results Synthesis - Robustness of Document Layout Analysis Under Compound Degradation

**Status:** results for 11 detectors on two page sets, 2026-11-10. Methodology checks in
`docs/phases/P7_extension/NOTES_khizar.md`, Section 7 (three small items still open). Source of every number: the phase
notes (`docs/phases/P*/NOTES_*.md`) and the result files under `RDLD_data/results/` (validation pages),
`RDLD_data_test/results/` (test pages) and the two document-bootstrap roots. Workstream B (Khizar) notes are on branch
`khizar`; Workstream A (Alishba) notes on branch `alishba`. Where a result changed from the 8-detector run, the earlier
number is given.

## Setup in one paragraph
Two page sets of 300 DocLayNet v1.2 pages each, 50 per document category, 11 classes: the **validation pages** (23
documents, 3,530 boxes) and the **test pages** from the official test split (44 documents, 4,351 boxes, no document shared
with the validation split or with RF-DETR's training data). Ten released detectors: four YOLO from `hantian/yolo-doclaynet`
(YOLO26 medium and large, YOLOv11 medium and large: a size × generation design), heron and heron-101 (RT-DETR v2),
egret and egret-xlarge (D-FINE), DETR, Deformable DETR. One fine-tuned by us: RF-DETR (DINOv2 backbone, about one sixth
of the training split). Eight image-only RoDLA degradations × 3 severities: 24 singles + 84 same-severity pairs, applied
in a fixed physical order with common random numbers. Validation pages also have reversed-order and second-seed
replicates (173 prediction sets per detector; 109 on test pages). Inference: a 1,000-resample paired page bootstrap,
and next to it a two-stage document bootstrap (documents, then pages within documents, stratified by category); BH-FDR.
A claim is made only when it holds under both. Downstream: one Tesseract pass per page, text read through GT vs
predicted regions, severity 2.

## Clean baseline (P1)

| | Val mAP [95% CI] | Test mAP | correct regions at τ = 0.5 (val / test) | duplicates % of predictions (val / test) | WER (val / test) |
|---|---|---|---|---|---|
| YOLOv11-l | **69.4** [66.6, 72.3] | 72.5 | 84.8 / 82.6 | 0.8 / 0.7 | 0.041 / 0.052 |
| YOLOv11-m | 68.9 [65.0, 72.6] | 71.3 | 84.8 / 81.9 | 1.1 / 1.4 | 0.063 / 0.060 |
| YOLO26-m | 68.8 [64.0, 72.5] | 71.4 | 87.8 / 84.2 | 0.6 / 0.7 | **0.022** / **0.029** |
| YOLO26-l | 68.2 [63.5, 72.2] | **73.1** | 87.9 / 83.4 | 0.5 / 0.4 | 0.035 / 0.040 |
| Deformable DETR | 66.2 [63.4, 69.7] | 70.4 | 82.2 / 78.4 | 0.0 / 0.1 | 0.058 / 0.072 |
| heron-101 | 65.9 [62.4, 70.3] | 66.1 | 85.8 / 82.3 | 3.2 / 8.1 | 0.098 / 0.175 |
| heron | 64.2 [60.4, 68.4] | 65.7 | 87.6 / 81.9 | 5.1 / 7.8 | 0.112 / 0.187 |
| egret-xlarge | 62.3 [58.5, 66.6] | 63.9 | 88.0 / 82.7 | 4.9 / 8.6 | 0.105 / 0.164 |
| egret | 61.5 [58.3, 65.8] | 64.4 | 88.0 / 82.6 | 4.5 / 8.2 | 0.102 / 0.171 |
| DETR | 53.0 [49.3, 57.1] | 41.3 | 83.6 / 75.8 | 5.1 / 7.6 | 0.185 / 0.237 |
| RF-DETR | 48.9 [45.8, 53.7] | 49.7 | 76.2 / 69.0 | 1.5 / 1.3 | 0.084 / 0.111 |

RF-DETR is not like for like because it had far less training data. The four YOLO models are within 1.2 mAP points on
validation and their order changes between page sets. DETR drops 11.7 points on the test pages, which we have not
explained. Title has 6 boxes on the validation pages and 19 on the test pages; see P7 notes.

## RQ1 - Do degradations compound?

**Answer: mostly near-additive, but on average worse than multiplicative, with a set of strongly super-additive pairs
from every detector family. Growth with severity depends on the detector. Compounding is created by the detector, not
the pixels. Holds on both page sets and under both bootstrap schemes.**

1. Negative control (77 `defocus@1` detector × pair cells per page set): interaction exactly 0.00 - the apparatus is exact. *(P3-K)*
2. Median I_mult positive for all 11 detectors: +0.21 to +1.48 on validation (Deformable DETR lowest, egret highest), +0.24 to
   +2.00 on test (egret lowest, egret-xlarge highest). Additive null: most pairs n.s. *(P3-K, P7-K)*
3. It grows with severity in the heron and egret families and RF-DETR (egret validation s1 → s3: −0.05 → +1.63 → +4.33). The YOLO
   models, DETR and Deformable DETR show no steady rise. *(P3-K)*
4. Multiplicative null, FDR, 924 cells: super / sub = 228 / 105 (validation, page bootstrap), 188 / 75 (validation,
   document), 299 / 134 (test, page), 219 / 86 (test, document). Super-additive under **both** nulls: 83 and 57 (validation),
   111 and 61 (test). With 8 detectors: 43 of 616 on validation, page bootstrap. Speckle and blur pairs recur. *(P3-K, P7-K)*
5. Singles predict pairs well (ρ 0.92–0.99); the multiplicative rule is best for all 11 detectors (MAE 1.0–2.7 points on
   validation, 1.1–3.0 on test) and under-predicts for 10 of 11. *(P3-K, P7-K)*
6. Mechanism: only 18% (validation) and 17% (test) of pairs compound at pixel level. Pixel and detector interactions are
   negatively correlated for egret (ρ = −0.33), heron (−0.26) and egret-xlarge (−0.25) on validation; egret-xlarge (−0.28) and
   RF-DETR (−0.22) on test. Positive correlations are weak: YOLO26-l +0.30 on validation (p = 0.005) and Deformable DETR +0.22 on test
   (p = 0.047), neither replicated on the other page set. *(P3-A)*
7. Robust to the seed: seed noise 0.28–0.41 × page noise for all 11 detectors, no opposite significant calls. *(P4-K)*
8. Robust to order **except texture**: texture pairs median order effect 3.2 points (up to 23.4 for YOLO26-m); other pairs
   0.3. 164 of 308 order effects significant, median 0.52 points (validation). *(P3-A)*
9. Without Title, medians stay similar and more pairs become significant. *(P7-K)*
10. The document bootstrap widens the median interval by 16% (validation) and 34% (test); about a fifth to a third of the
    significant cells become n.s. No conclusion changes. *(P7-K)*

## RQ2 - Does mAP predict downstream quality?

**Answer: partially. mAP tracks WER within a condition (ρ ≈ −0.73 on both page sets) but often picks a different best
detector, is blind to duplicated output, and the proposed UW-mAP gives no gain. At a lower score cut-off its link to WER
disappears.**

1. Within condition, ρ(mAP, WER) = −0.73 on both page sets (−0.68 with 8 detectors). Best detector by mAP = best by WER in
   40.5% (validation) and 27% (test) of 37 conditions (76% with 8). Four YOLO models within 1.2 mAP points make the top spot
   a near tie. The full order never agrees. *(P5-K)*
2. ρ(mAP, text precision) within condition = −0.10 (validation) and 0.03 (test); ρ(duplicate share, text precision) = −0.91 and
   −0.89. *(P5-K)* Class-agnostic dedup lowers mAP by 4.0–4.7 points in the heron and egret families, 3.0 for Deformable DETR
   and 3.1 for YOLOv11-m. *(P1-K)*
3. On clean pages small mAP gaps hide large WER gaps: YOLO26-m 0.022 vs 0.098–0.112 for the heron and egret families on
   validation, 0.029 vs 0.164–0.187 on test. RF-DETR has the lowest validation mAP but a lower WER than any Docling model. *(P5-A)*
4. Failure type predicts downstream symptom. Change in missed regions, severity-3 pairs (validation): YOLOv11-m and -l +42.2,
   Deformable DETR +28.3, YOLO26-m +25.7, RF-DETR +24.3, YOLO26-l +23.8 pp. These have the lowest text recall on pairs
   (0.56–0.80). The heron and egret families mostly merge (+20.5 to +34.0 pp) and keep recall (≥ 0.89). The same split
   holds on the test pages. *(P4-A, P5-A)*
5. The absolute failure labels depend on the score cut-off (at 0.3 nearly every detector mostly merges, at 0.7 nearly every
   detector mostly misses), but the relative order holds (rank correlation 0.75–0.94 between cut-offs). State it as
   "relatively more merging / missing". *(P7-K)*
6. At cut-off 0.3 the Docling models and DETR read duplicated text, their WER on pairs rises to 0.62–0.70, and within-condition
   ρ(mAP, WER) falls to −0.00 (ρ(duplicate share, WER) = 0.81). At 0.7: −0.65. *(P7-K)*
7. Text recall against mAP retention has a breakpoint, but it moved from 39% retention (3 detectors) to 24% (8) and 38% (11, both
   page sets). It depends on the detector set, so we should not claim a fixed threshold. *(P5-K)*
8. UW-mAP: robust to its weights (drop ratio 1.06 [1.02, 1.09] validation, 1.05 [1.01, 1.09] test) but no better than mAP:
   within condition ρ with WER −0.73 vs −0.73 (validation), −0.74 vs −0.73 (test); pooled −0.91 vs −0.92 and −0.90 vs −0.91. *(P5-K)*

## RQ3 - Are detector rankings preserved?

**Answer: no, rankings break down with damage, on both page sets, under every interval method and detector set. Pairing
itself does not consistently scramble them beyond what the extra damage explains, and single and compound benchmarks give
nearly the same robustness order. The choice of robustness metric matters much more.**

1. Kendall τ vs clean order, controls excluded, all 11: singles 0.49, pairs 0.22, gap 0.27 [0.19, 0.29] on validation;
   singles 0.59, pairs 0.32, gap 0.27 [0.23, 0.29] on test (page percentile). The gap excludes zero under page percentile, page
   BCa and document intervals, and for all 11, distinct architectures, without RF-DETR and the first 8 on both page sets. Seed noise
   floor τ 0.92 (pairs) and 0.94 (singles) for all 11 (0.94–0.96 with 8), far above the damage values. *(P4-K, P7-K)*
2. At equal damage, the pair effect on τ lies between −0.024 and +0.051 across estimators, detector sets and page sets. Sign
   and significance change with the estimator, interval method and page set. No consistent pair effect. *(P7-K)*
3. Clean order of all 11 holds in 2 of 23 singles and 0 of 77 pairs on validation (1 and 1 on test). For the original three:
   17 of 23 and 41 of 77 (validation), 18 and 45 (test). Top detector on pairs, validation: YOLO26-l 26, heron 17, heron-101 13,
   YOLOv11-l 8, YOLO26-m 6; test: YOLO26-l 28, heron 17, YOLO26-m 14, heron-101 13. *(P4-K)*
4. Significant reversals after FDR: 15.0% of single tests and 25.4% of pair tests on validation; 16.7% and 28.3% on test
   (8.7% and 15.5% with 8 detectors). *(P4-K)*
5. Robustness order by retention is identical on single and pair benchmarks on validation (τ 1.0) and close on test (0.89):
   heron, heron-101, RF-DETR, YOLO26-l, egret-xlarge, YOLO26-m, egret, DETR, Deformable DETR, YOLOv11-l, YOLOv11-m (validation).
   Heron is first in every detector set, page set and bootstrap scheme. *(P4-K, P7-K)*
6. By RoDLA's mRD: YOLO26-l 7.7, YOLO26-m 7.8, YOLOv11-l 8.3, YOLOv11-m 8.3, heron-101 8.8, Deformable DETR 8.8, heron 8.9,
   egret-xlarge 9.5, egret 9.9, DETR 11.8, RF-DETR 12.2 (validation). mRD and retention orders disagree: τ = −0.09 (validation)
   and −0.16 (test) for 11 detectors, 0.07 for the first 8. The YOLOv11 models are third and fourth by mRD and last two by
   retention. *(P4-K, P7-K)*
7. Generation, not size: YOLOv11 keeps about 7 points less of its mAP on singles and 11 on pairs than YOLO26 at equal clean
   accuracy, on both page sets; large vs medium changes retention by at most 1.7 points. Confounded with training length
   (100 vs 40 epochs). *(P7-K)*
8. Failure patterns: the heron and egret families correlate ρ = 0.97–0.99 on per-configuration drops. YOLO26 correlates with Deformable DETR and RF-DETR
   (0.95–0.96) more than with the Docling models (0.89–0.91); YOLOv11 correlates more with the Docling models (0.95–0.96).
   YOLO is texture-fragile, and Deformable DETR scores above clean on `defocus@3`. *(P2-K, P4-K)*

## Secondary findings
- Class vulnerability is not explained by box size (validation ρ = −0.20, p = 0.58; test ρ = −0.48, p = 0.16). Formula drops most
  at severity 3 on both page sets (26.7 and 26.6 points); Table compounds most (median I_mult at s3 2.14 and 2.24). *(P4-A)*
- On pairs at severity 3, large regions lose more than small ones for every non-YOLO detector on both page sets
  (+5.4 to +25.5 points). YOLO26 shows no size effect. *(P4-A)*
- Image-quality damage predicts detector damage (ρ(mPE_IQA, D) = 0.78–0.91 validation, 0.80–0.88 test); PSNR does less well
  (−0.44 to −0.68). *(P3-A)*
- Selection vs test pages: no detector does unusually well on the test pages. *(P7-K)*

## Deviations from the proposal (must be stated in the paper)
1. mPE is IQA-only (MS-SSIM + CW-SSIM), not RoDLA's IQA + Faster R-CNN term - model-independent by design.
2. heron/egret have 17 classes (not 12 as in the Sept notes); 6 unmapped classes dropped, counts recorded.
3. `defocus@1` is a pixel no-op in RoDLA's code; used as a negative control and excluded from mRD.
4. Downstream covers severity 2 only (37 conditions).
5. Same-severity pairs only (no cross-severity pairs, no triples).
6. 3 detectors became 11, one of them fine-tuned by us.
7. RQ3 gained a test of whether ranking changes come from pairing or from damage, added after the first results.
8. A second page set from the official test split, and a document bootstrap next to the page bootstrap, both added after
   the first results.

## Limitations
- Two page sets of 300 pages, one dataset (DocLayNet). The validation pages come from 23 documents (three categories have
  one document each); the test pages from 44.
- The 11 detectors are not independent: four YOLO, two heron, two egret, several DETR-family models. Distinct-architecture
  results are reported.
- Released detectors may have used the validation pages to pick checkpoints; the test pages remove this.
- YOLO generation is confounded with training length.
- Few Title boxes (6 and 19). Report mAP with and without Title.
- RF-DETR is not like for like, having about one sixth of the training pages.
- Two composition orders tested, not all; texture results are order-dependent. Order and seed replicates exist on the
  validation pages only.
- One OCR engine (Tesseract 5.5.3); pages with light-on-photo text are unreadable to it (7 / 300 clean validation pages return no words; see P5-A for test pages).
- 407 downstream points share 37 conditions; pooled CIs are optimistic (within-condition analysis is the primary test).
- Released weights for ten detectors, none of them fine-tuned or augmented by us.

## Integrity checks that held
Byte-reproducible cache · deterministic detectors · exact negative control on both page sets (after fixing a precision
mismatch it exposed) · fast bootstrap verified equal to pycocotools · page scheme of the new bootstrap code bit-identical
to the old tables · 8-detector results reproduced row by row inside the 11-detector run · seed replicate · order
replicate · OCR fail-loud guard (after a silent-empty first run was caught) · test pages rebuilt byte for byte from the
extraction script · no test document in the validation split or RF-DETR training data.

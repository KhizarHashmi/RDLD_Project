# P3 - Main Experiment - Khizar (Workstream B)

**Dates:** 2026-11-04

## 1. Goal
Answer RQ1 for three detectors: for each same-severity pair of degradations, is the mAP drop of the
pair equal to what the two singles predict, and if not, which pairs compound and how strongly?

## 2. What was done

| Step | Command | Output |
|---|---|---|
| Paired page-bootstrap tables, 1,000 resamples, one shared resample index (seed 20261103) | `python scripts/p3_bootstrap.py` | `RDLD_data/results/p3/boot/<model>/<config>.npy` (1001 × 11 per-class AP) |
| Interaction terms, tests, prediction from singles, figures | `python scripts/p3_rq1_interaction.py` | `RDLD_data/results/p3/rq1/` |

Log: `RDLD_data/results/p3/logs/p3_rq1_interaction.log`, `p3_bootstrap.log`.

**Interaction terms** (D = mAP drop from clean in points; R = mAP / mAP_clean):

- additive null: I_add = D(A+B) − D(A) − D(B)
- multiplicative null: I_mult = 100 · [R(A)·R(B) − R(A+B)]

I > 0 means the pair hurts more than the null predicts (compounding). Both nulls are reported because
mAP is bounded: two large drops cannot add past zero, so the additive null alone labels saturation as
"sub-additive".

**Inference.** Every term is computed on the full corpus (point estimate) and on each of 1,000 page
resamples. Because the same resample index is used for all configurations and models, D(A), D(B) and
D(A+B) are evaluated on the same pages, and page-sampling noise cancels in I. 95% percentile CI and a
two-sided bootstrap p; Benjamini–Hochberg FDR across the 84 pairs per detector and null. A pair is
"super" / "sub" only if q < 0.05.

**Negative control.** `defocus@1` is a pixel no-op, so its 7 pairs per detector (21 in total) must give
I = 0 exactly; they are excluded from the counts below.

## 3. Results (evidence)

### 3.1 Negative control (`p3_rq1_interaction.log`)
max |I_add| over the 21 `defocus@1` pairs = **0.00** → PASS. (Failed at 0.026 before the precision fix
recorded in the P2 note; this is the check that found it.)

### 3.2 Counts of pairs by class, q < 0.05 (`p3/rq1/rq1_summary.csv`; 77 non-control pairs per detector)

| Detector | additive: super / sub / n.s. | multiplicative: super / sub / n.s. | median I_mult (points) |
|---|---|---|---|
| YOLO | 9 / 18 / 50 | 15 / 12 / 50 | +0.56 |
| heron | 7 / 12 / 58 | 22 / 5 / 50 | +1.09 |
| egret | 3 / 14 / 60 | 29 / 9 / 39 | +1.48 |

By severity, median I_mult (points), s1 → s2 → s3:
YOLO 0.46 → 0.71 → 0.67 · heron 0.30 → 1.29 → 1.62 · **egret −0.05 → 1.63 → 4.33**.

### 3.3 Which degradations compound (`p3/rq1/rq1_mean_interaction_by_degradation.csv`, mean I_mult over partners and models)

| Degradation | s1 | s2 | s3 |
|---|---|---|---|
| speckle | 1.90 | 5.17 | 4.11 |
| vibration | 1.33 | 5.01 | 3.68 |
| defocus | - | 2.03 | 3.76 |
| illumination | 0.51 | 1.48 | 1.89 |
| watermark | −0.50 | 0.82 | 2.07 |
| texture | −0.96 | 0.02 | 1.64 |
| ink_bleeding | 0.12 | 0.76 | 0.66 |
| ink_holdout | 0.18 | 0.13 | −0.69 |

### 3.4 Super-additive under BOTH nulls (`p3/rq1/rq1_super_under_both_nulls.csv`) - 18 detector × pair cells
Strongest cases (points):

| Detector | Pair | D(A) | D(B) | D(A+B) | I_add | I_mult |
|---|---|---|---|---|---|---|
| YOLO | speckle@1 + vibration@1 | 24.65 | −0.05 | 37.85 | 13.24 | 19.22 |
| YOLO | speckle@2 + defocus@2 | 50.33 | −0.11 | 60.28 | 10.06 | 14.50 |
| YOLO | ink_holdout@3 + defocus@3 | 0.91 | 2.02 | 12.82 | 9.89 | 14.40 |
| heron | speckle@2 + defocus@2 | 30.64 | 1.32 | 35.58 | 3.62 | 6.61 |
| heron | defocus@2 + vibration@2 | 1.32 | 24.73 | 29.72 | 3.67 | 6.51 |

The pattern: **speckle with anything**, and **a blur (defocus / vibration) with ink or speckle**. Two
degradations that are each harmless to YOLO (ink_holdout@3: 0.9, defocus@3: 2.0) together cost 12.8 points.

### 3.5 Can the singles predict the pair? (`p3/rq1/rq1_prediction_from_singles.csv`)

| Detector | predictor | MAE (pts) | bias (pts) | Spearman |
|---|---|---|---|---|
| YOLO | additive | 4.37 | +2.36 | 0.978 |
| YOLO | **multiplicative** | **2.70** | −0.27 | 0.976 |
| YOLO | max | 3.54 | −1.37 | 0.958 |
| heron | additive | 2.05 | +0.74 | 0.988 |
| heron | **multiplicative** | **1.83** | −1.26 | 0.988 |
| heron | max | 3.65 | −3.57 | 0.972 |
| egret | additive | 2.51 | +1.38 | 0.984 |
| egret | **multiplicative** | **2.22** | −1.47 | 0.979 |
| egret | max | 4.58 | −4.40 | 0.946 |

Figures: `fig_single_curves.png` (RoDLA-style single-degradation curves), `fig_interaction_heatmaps.png`
(8 × 8 I_mult per detector × severity), `fig_observed_vs_predicted.png`.

## 4. Decisions and why
1. **Two nulls, not one.** The additive null is the intuitive one; the multiplicative null respects the
   bound on mAP. A pair is called compounding with confidence only when it is super-additive under both.
2. **Paired bootstrap over pages, one shared index.** P1 showed a page-sampling SD of ≈ 2 points on clean
   mAP alone; an unpaired comparison of four mAPs would drown interactions of 1–5 points.
3. **BH-FDR within detector × null** (84 tests each): the study asks "which pairs compound", a
   many-tests question; FDR controls the share of false discoveries among those reported.
4. **Fast bootstrap engine** (`rdld/fastcoco.py`) made 1,000 resamples × 519 prediction sets feasible
   (≈ 25 s per table for YOLO, ≈ 120–180 s for the transformers, whose clean box counts at the 0.01 floor are ≈ 16× higher: 79,839 vs 4,842).

## 5. Problems found
- **The negative control caught a real bug** (clean vs degraded prediction precision; see the P2 note).
  Every number above is from after the fix.
- **Bootstrap p has a floor of 2/1001 ≈ 0.002.** With 84 tests, BH still allows q < 0.05 for strong
  effects (the smallest q reported is 0.01), but very strong and moderately strong effects cannot be told
  apart by p - use the CI and effect size, not p, to rank.
- **Multiplicative predictor bias is negative for every detector** (−0.3 to −1.5 points): even the best
  cheap estimator under-predicts compound damage on average. This is the RQ1 finding, not a defect.

## 6. Main takeaway
RQ1 answer: **compounding is mostly near-additive, but systematically super-multiplicative and
growing with severity** (median I_mult +0.6 to +1.5 points overall; +4.3 for egret at s3), with
**specific super-additive pairs** (18 cells under both nulls) centred on speckle and blur × ink. Single-
degradation benchmarks therefore under-estimate compound damage, most for the D-FINE head (egret).

## 7. Hand-off
- P4 (Khizar): the same bootstrap tables feed RQ3 ranking tests; `r1-*` tables feed seed stability.
- P3 (Alishba): `rq1_interactions.csv` (I_mult per detector × pair) for image-level vs detector-level
  interaction; `rev-*` bootstrap tables for the order check.
- P5: `metrics_long.csv` + bootstrap tables for UW-mAP.

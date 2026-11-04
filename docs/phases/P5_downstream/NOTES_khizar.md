# P5 - Downstream Link - Khizar (Workstream B)

**Dates:** 2026-11-04

## 1. Goal
Validate the layout metrics against what a document pipeline delivers (Workstream A's OCR pipeline):
does mAP track downstream quality, does a consequence-weighted mAP (UW-mAP) or a structural error rate
track it better, and does the per-condition "best detector" by mAP match the best by WER?

## 2. What was done

| Step | Command | Output |
|---|---|---|
| Metric-validity analysis | `python scripts/p5_rq2_metric_validity.py` | `RDLD_data/results/p5/rq2/` |

Inputs: `p2/evaluation/metrics_long.csv`, `p3/boot/*` (per-class AP), `p4/structural/structural_rates.csv`
(Workstream A), `p5/downstream/downstream_configs.csv` (Workstream A). Log:
`RDLD_data/results/p5/logs/p5_rq2_metric_validity.log`.

**Metric ladder.** Tier 1 - mAP, mAP50, mAP75 and **UW-mAP** (per-class AP weighted by downstream
consequence; weights fixed on Sept 15 before any result: Table 3, Formula 2.5, Picture 2, Section-header
1.5, Title 1.5, List-item 1.2, Caption 1, Text 1, Footnote 0.6, Page-header 0.3, Page-footer 0.3).
Tier 2 - structural rates at τ = 0.5: share correct, structural error (merged + split), duplicate share.
Tier 3 - downstream: WER, text recall, text precision, table recall.

**Analyses.** (1) UW-mAP vs mAP, with weight sensitivity: 2,000 draws, each weight × U(0.5, 1.5).
(2) Spearman ρ between every layout metric and every downstream outcome over 111 detector × condition
points (3 × [clean + 36 severity-2 conditions]), with a 500-resample bootstrap CI over points - and the
same **within condition** (condition means removed), which isolates "which detector is better on the
same images" from "which condition is harder". (3) Per condition, does the best detector by mAP also have
the lowest WER? (4) Two-segment fit of text recall on mAP retention, F-test against a straight line.

## 3. Results (evidence)

### 3.1 Pooled: layout metric vs downstream (`p5/rq2/metric_vs_downstream_correlation.csv`, n = 111)

| Layout metric | ρ with WER [95% CI] | ρ with text precision [95% CI] |
|---|---|---|
| mAP75 | **−0.932** [−0.955, −0.891] | 0.544 [0.377, 0.700] |
| mAP | −0.919 [−0.954, −0.866] | 0.463 [0.259, 0.647] |
| UW-mAP | −0.900 [−0.943, −0.835] | 0.426 [0.239, 0.621] |
| mAP50 | −0.824 [−0.886, −0.727] | 0.317 [0.128, 0.491] |
| share correct (τ 0.5) | −0.799 [−0.867, −0.700] | 0.260 [0.064, 0.454] |
| structural error | 0.670 [0.571, 0.754] | −0.345 [−0.486, −0.183] |
| duplicate share | 0.321 [0.089, 0.525] | **−0.885** [−0.910, −0.840] |

### 3.2 Within condition (`p5/rq2/within_condition_correlation.csv`)

| Layout metric | WER | text recall | text precision | table recall |
|---|---|---|---|---|
| mAP | **−0.832** | 0.563 | **−0.022** | 0.706 |
| UW-mAP | −0.826 | 0.579 | −0.071 | 0.657 |
| mAP75 | −0.775 | 0.345 | 0.263 | 0.769 |
| mAP50 | −0.642 | 0.635 | −0.394 | 0.324 |
| duplicate share | −0.188 | 0.695 | **−0.853** | −0.155 |
| structural error | 0.296 | −0.143 | 0.208 | 0.036 |

**mAP ranks detectors correctly on WER within a condition (ρ = −0.83), but it is blind to text
precision (ρ = −0.02), which is governed almost entirely by duplicate boxes (ρ = −0.85).** This is the P1
hedging effect measured downstream: duplicates cost mAP almost nothing and cost the reader repeated text.

### 3.3 Ranking agreement (`p5/rq2/ranking_agreement.csv`, 37 conditions)
Best detector by mAP = best by WER in **81%** of conditions; full 3-detector order agrees in **59%**.
All 7 disagreements but one are **speckle** conditions (speckle@2 and its pairs with ink_bleeding,
ink_holdout, defocus, vibration, watermark), where heron leads on mAP and egret on WER; the seventh is
texture@2 + ink_holdout@2 (heron by mAP, YOLO by WER).

### 3.4 Clean condition: size of the downstream gap (`p5/downstream/downstream_configs.csv`, P1)
mAP 68.8 / 64.2 / 61.5 (YOLO / heron / egret) vs WER 0.022 / 0.112 / 0.102: a ≤ 7-point mAP gap is a
**5× WER gap**.

### 3.5 UW-mAP (`p5/rq2/uw_map.csv`, `uw_weight_sensitivity.csv`)
UW-mAP is 2.9–4.8 points below mAP on clean pages (YOLO −2.9, heron −4.8, egret −4.7), and the gap
narrows under degradation (−1.1 to −2.7 at severity 3). Its drop at severity 3 is **1.09×** mAP's drop
(2,000 weight draws: 95% range 1.00–1.18) - robust to the weights, but UW-mAP **does not predict
downstream quality better than plain mAP** (§3.1, §3.2).

### 3.6 Threshold (`p5/rq2/threshold_fit.csv`)
Text recall vs mAP retention is strongly non-linear: breakpoint at **retention 0.39**; slope 1.99 below,
**0.11 above** (F = 131.8 vs a straight line, n = 111). Above ≈ 40% of clean mAP, losing mAP barely costs
text recall; below it, recall collapses.

Figures: `p5/rq2/fig_map_vs_downstream.png`, `fig_metric_correlation.png`.

## 4. Decisions and why
1. **Within-condition correlation as the decisive test.** Pooled correlations are dominated by condition
   difficulty (every metric gets worse on harder images); model selection is a within-condition question.
2. **UW-mAP weights fixed before results** (Sept 15) to avoid tuning the metric to the outcome; the
   sensitivity analysis shows the conclusion does not depend on the exact weights.
3. **Report the negative result for UW-mAP.** The proposed metric is not better than mAP for WER; the
   metric that adds information is the duplicate share, which mAP ignores.

## 5. Problems found
- The first run of this analysis used an empty OCR cache (Workstream A's P5 note, §5) and was stopped
  before any result was used; all numbers here are from the corrected downstream table.
- Downstream covers severity 2 only (37 conditions); the threshold fit therefore relies on the range of
  retentions that severity-2 conditions produce.
- 111 points are not independent (3 detectors share each condition); the bootstrap CI over points is
  therefore optimistic. The within-condition analysis and ranking agreement do not have this problem.

## 6. Main takeaway
RQ2 answer: **mAP predicts downstream quality partially.** It tracks WER across and within
conditions, but (a) is blind to duplication, which drives text precision; (b) disagrees with WER on the
best detector under speckle; (c) has a threshold relationship with text recall (flat above 39%
retention). The consequence-weighted UW-mAP does not improve on mAP - a negative result worth reporting.

## 7. Hand-off
- P6: the three RQ answers (P3, P4, P5 notes) go into the results synthesis; the duplicate-share
  recommendation goes into the discussion.

# P7 - Detector Extension and Methodology Checks - Khizar (Workstream B)

**Dates:** 2026-11-04 – 2026-11-10
**Status:** done. 11 detectors on two page sets (DocLayNet validation and official test split), every analysis rerun,
and the methodology checks from the first version of these notes closed (Section 7).

## 1. Goal
1. Widen the comparison from 3 to 11 detectors, each through the same pipeline as the original three.
2. Repeat the study on pages from the official DocLayNet test split, which no released detector could have used to
   pick a checkpoint.
3. Test whether the conclusions depend on analysis choices: the detector set, the interval method, the resampling
   unit and the 0.5 score cut-off.

## 2. What was done

### 2.1 Detectors

| Key | Model | Source |
|---|---|---|
| `yolo` | YOLO26 medium | `hantian/yolo-doclaynet` (original) |
| `heron`, `egret` | RT-DETR v2, D-FINE | `docling-project` (original) |
| `heron101` | RT-DETR v2, ResNet-101 | `docling-project/docling-layout-heron-101` |
| `egretxl` | D-FINE x-large | `docling-project/docling-layout-egret-xlarge` |
| `detr` | DETR, ResNet-50 | `cmarkea/detr-layout-detection` |
| `ddetr` | Deformable DETR | `Aryn/deformable-detr-DocLayNet` |
| `rfdetr` | RF-DETR medium, DINOv2 backbone | fine-tuned by us from `Roboflow/rf-detr-medium` |
| `yolo26l` | YOLO26 large | `hantian/yolo-doclaynet` |
| `yolo11m` | YOLOv11 medium | `hantian/yolo-doclaynet` |
| `yolo11l` | YOLOv11 large | `hantian/yolo-doclaynet` |

The four YOLO models form a 2 × 2 of size (medium, large) × generation (YOLO26, YOLOv11). All use image size 1024.
The model card says YOLO26 was trained for 100 epochs and YOLOv11 for 40, so generation is confounded with training
length. The weights are AGPL-3.0. In the distinct-architecture analysis the four YOLO models count as one family.

### 2.2 Official test split
300 pages, 50 per category, from the DocLayNet v1.2 test split, extracted with `scripts/p0_prepare_doclaynet.py`
(split option on branch `alishba`). Data root `RDLD_data_test` (`RDLD_DATA` environment variable), so every script
runs unchanged. 109 configurations per detector: clean, 24 singles, 84 pairs (no order or seed replicates).

`scripts/p7_test_split_leakage.py` (`RDLD_data_test/results/p7/test_split_leakage.csv`):

| Checked against | Pages | Documents | Test documents shared | Test pages shared |
|---|---|---|---|---|
| RF-DETR training shards (12 of 72) | 11,563 | 1,431 | 0 | 0 |
| Validation split (all 7 shards) | 6,489 | 160 | 0 | 0 |

The test pages come from 44 documents (validation pages: 23) and have 19 Title boxes (validation: 6).

### 2.3 Methodology checks

| Check | Code | Output |
|---|---|---|
| Results with and without RF-DETR, distinct architectures, first 8 | `scripts/p4_rq3_robustness.py` (detector sets) | `p4/rq3_robustness/` |
| BCa intervals for RQ1 and RQ3 | `rdld/intervals.py`, `scripts/p3_jackknife.py` (300 leave-one-page-out tables) | `_bca` columns in `rq1_interactions.csv`, `tau_gap.csv`, `pair_effect_at_equal_damage.csv` |
| Document bootstrap next to the page bootstrap | `BOOT_SCHEME=doc python scripts/p3_bootstrap.py` | separate roots `RDLD_data_docboot`, `RDLD_data_test_docboot` |
| Score cut-off 0.3 and 0.7 (main: 0.5) | `TAU=` on the structural, OCR and RQ2 scripts (validation pages) | `p4/structural_tau*`, `p5/downstream_tau*`, `p5/rq2_tau*` |

**Document bootstrap.** The validation pages come from only 23 documents, and pages of one document look alike, so
resampling pages may give intervals that are too narrow. The document scheme is two-stage and stratified by
category: draw documents with replacement, then pages with replacement within each drawn document (Davison and
Hinkley, Section 3.8). Seed 20261108, 1,000 resamples, 202–439 pages per resample. It is on the conservative side.
The page scheme is the liberal bound, the document scheme the conservative one, and both are reported. Only
percentile intervals are computed for the document scheme. The page scheme was checked to reproduce the old tables
bit for bit.

Every run is logged in `RDLD_data/results/p7/logs/` (`chain2.log`, `checks.log`, `docboot.log`); all three chains
ended with no unexpected failures.

## 3. Results

### 3.1 Integrity
- The original 8 detectors reproduce the 8-detector backup (`results/_backup_8det`) row by row in `metrics_long`,
  `rq1_interactions`, `single_degradation_drops`, `structural_rates`, `downstream_configs`, `uw_map` and
  `per_class_single_drops`. Tau gap for the first 8: 0.207 [0.131, 0.243], the same as before.
- Test split: all 11 detectors × 109 configurations, no missing mAP. Image quality complete for 108 configurations.
- Point estimates are identical under both bootstrap schemes; only the intervals change.
- Negative control (`defocus@1` pairs, 77 detector × pair cells per page set): interaction exactly 0.00 on both page sets.

### 3.2 Clean mAP

| Detector | Validation mAP [95% CI] | Test mAP | Test mAP@.5 |
|---|---|---|---|
| `yolo11l` | 69.4 [66.6, 72.3] | 72.5 | 87.0 |
| `yolo11m` | 68.9 [65.0, 72.6] | 71.3 | 85.9 |
| `yolo` (26m) | 68.8 [64.0, 72.5] | 71.4 | 84.9 |
| `yolo26l` | 68.2 [63.5, 72.2] | **73.1** | 86.1 |
| `ddetr` | 66.2 [63.4, 69.7] | 70.4 | 88.4 |
| `heron101` | 65.9 [62.4, 70.3] | 66.1 | 86.7 |
| `heron` | 64.2 [60.4, 68.4] | 65.7 | 85.7 |
| `egretxl` | 62.3 [58.5, 66.6] | 63.9 | 86.4 |
| `egret` | 61.5 [58.3, 65.8] | 64.4 | 85.6 |
| `detr` | 53.0 [49.3, 57.1] | **41.3** | 64.5 |
| `rfdetr` | 48.9 [45.8, 53.7] | 49.7 | 71.4 |

Most detectors score higher on the test pages. DETR drops 11.7 points, which we have not explained. The clean order
changes between page sets (YOLOv11-l first on validation, YOLO26-l first on test), so the clean ranking of the four
YOLO models is within noise.

### 3.3 RQ1 on both page sets
- The multiplicative rule is the best predictor of pairs for all 11 detectors on both page sets (MAE 1.0–2.7 points on
  validation, 1.1–3.0 on test; Spearman 0.92–0.99), and it under-predicts for 10 of 11.
- Median I_mult per detector: +0.21 to +1.48 on validation, +0.24 to +2.00 on test, positive for every detector.
- FDR-classified cells, all 924 detector × pair cells:

| | Page bootstrap | Document bootstrap |
|---|---|---|
| Validation, multiplicative null: super / sub / n.s. | 228 / 105 / 591 | 188 / 75 / 661 |
| Test, multiplicative null: super / sub / n.s. | 299 / 134 / 491 | 219 / 86 / 619 |
| Validation, super under both nulls | 83 | 57 |
| Test, super under both nulls | 111 | 61 |

The document bootstrap widens the median I_mult interval by 16% on validation and 34% on test. About a fifth to a third
of the significant cells lose significance, but the direction of the findings is unchanged. Super-additive cells under
both nulls come from all detector families; on validation the largest counts are YOLOv11-l (15), YOLO26-l (14),
YOLOv11-m (11) and egret-xlarge (10).

### 3.4 RQ3: tau gap, singles vs pairs (controls excluded)

| Page set, detectors | τ singles | τ pairs | Gap | Page pct. CI | Page BCa CI | Document CI |
|---|---|---|---|---|---|---|
| Validation, all 11 | 0.494 | 0.222 | 0.272 | [0.195, 0.292] | [0.248, 0.311] | [0.194, 0.293] |
| Validation, distinct architectures | 0.670 | 0.420 | 0.250 | [0.178, 0.284] | [0.206, 0.295] | [0.164, 0.295] |
| Validation, without RF-DETR | 0.469 | 0.195 | 0.274 | [0.183, 0.293] | [0.250, 0.312] | [0.181, 0.291] |
| Validation, first 8 | 0.671 | 0.464 | 0.207 | [0.131, 0.243] | [0.153, 0.258] | [0.123, 0.253] |
| Test, all 11 | 0.589 | 0.323 | 0.266 | [0.231, 0.287] | [0.244, 0.302] | [0.217, 0.292] |
| Test, distinct architectures | 0.739 | 0.510 | 0.229 | [0.191, 0.241] | [0.218, 0.256] | [0.184, 0.247] |
| Test, without RF-DETR | 0.563 | 0.301 | 0.262 | [0.218, 0.286] | [0.234, 0.304] | [0.204, 0.295] |
| Test, first 8 | 0.671 | 0.466 | 0.205 | [0.176, 0.232] | [0.176, 0.232] | [0.172, 0.241] |

The gap excludes zero under every interval method, detector set and page set. This is the most stable result of the study.

### 3.5 RQ3: pair effect at equal damage
Positive means pairs keep the clean order better than singles at the same retention.

| Page set, detectors, estimator | Estimate | Page pct. | Page BCa | Document |
|---|---|---|---|---|
| Validation, all 11, linear | +0.031 | [−0.012, 0.053] | [0.005, 0.065] | [−0.017, 0.054] |
| Validation, all 11, bands | +0.008 | [−0.030, 0.031] | [−0.011, 0.040] | [−0.034, 0.036] |
| Validation, distinct, linear | +0.042 | [0.009, 0.078] | [0.006, 0.075] | [−0.001, 0.076] |
| Test, all 11, linear | +0.017 | [−0.001, 0.045] | [−0.009, 0.035] | [−0.007, 0.048] |
| Test, all 11, bands | −0.024 | [−0.034, 0.033] | [−0.045, −0.012] | [−0.034, 0.041] |
| Test, distinct, linear | +0.036 | [0.022, 0.072] | [0.003, 0.049] | [0.015, 0.077] |

All estimates for every detector set are in `pair_effect_at_equal_damage.csv`. They lie between −0.024 and +0.051. Their sign and
significance change with the estimator, the interval method and the page set, and no single direction holds across
them. Conclusion: no consistent pair effect at equal damage. The ranking breakdown under pairs is explained by
the larger damage. We do not claim that pairs help or hurt rankings beyond that.

BCa and the percentile method disagree most here and for the original three (validation BCa for the original three:
[−0.206, 0.070]). Kendall τ over 3–11 detectors is a discrete statistic, and BCa assumes a smooth one, so for RQ3 the
percentile intervals under both schemes are the primary report and BCa is a check.

### 3.6 RQ3: single vs pair benchmark
Robustness order by mean retention, singles vs pairs: τ = 1.0 on validation (identical orders) and 0.89 on test. Heron
is the most robust detector in every detector set, page set and bootstrap scheme. The top detector differs between the
two benchmarks in 4.1% (page) and 5.8% (document) of validation resamples, 10.6% and 13.3% on test.

Validation order: heron > heron101 > RF-DETR > YOLO26-l > egret-xlarge > YOLO26-m > egret > DETR > Deformable DETR >
YOLOv11-l > YOLOv11-m.

### 3.7 Retention vs mRD
mRD order (lower = more robust), validation: YOLO26-l 7.7, YOLO26-m 7.8, YOLOv11-l 8.3, YOLOv11-m 8.3, heron101 8.8,
Deformable DETR 8.8, heron 8.9, egret-xlarge 9.5, egret 9.9, DETR 11.8, RF-DETR 12.2. Test: the same top four, then Deformable DETR, heron,
heron101.

τ(mRD order, retention order): 0.07 for the first 8 on both page sets; −0.09 (validation) and −0.16 (test) for all 11.
The YOLOv11 models show the gap most clearly: last two by retention, third and fourth by mRD. mRD scores absolute error and
rewards clean accuracy. The choice of robustness metric changes the answer more than single vs pair benchmarks do.

`p2_evaluate.py --mrd-only` was run on the test root after its image quality finished (`logs/p2_evaluate_mrd.log`).

### 3.8 YOLO size × generation

| Model | Clean val / test | Retention singles val / test | Retention pairs val / test | Missed regions, s3 pairs (pp change, val / test) |
|---|---|---|---|---|
| YOLO26-m | 68.8 / 71.4 | 0.794 / 0.801 | 0.620 / 0.630 | +25.7 / +25.0 |
| YOLO26-l | 68.2 / 73.1 | 0.805 / 0.800 | 0.637 / 0.634 | +23.8 / +22.5 |
| YOLOv11-m | 68.9 / 71.3 | 0.731 / 0.729 | 0.517 / 0.517 | +42.2 / +36.9 |
| YOLOv11-l | 69.4 / 72.5 | 0.737 / 0.735 | 0.530 / 0.523 | +42.2 / +37.7 |

- **Generation matters, size barely does.** At equal clean accuracy, YOLOv11 keeps about 7 points less of its mAP on singles
  and 11 points less on pairs, on both page sets, and misses far more regions. Large vs medium changes retention by at
  most 1.7 points.
- **The four models do not fail in the same way.** On validation, YOLO26 drops correlate with Deformable DETR and RF-DETR (ρ 0.95–0.96) and less with
  the Docling models (ρ 0.89–0.91); YOLOv11 drops correlate 0.95–0.96 with the Docling models and 0.93–0.94 with Deformable
  DETR and RF-DETR. The two YOLO generations fail differently.
- **Confound.** YOLO26 trained 100 epochs, YOLOv11 40. The generation effect may be partly a training-length effect;
  we can only claim "these released checkpoints", not "this architecture".
- Within the YOLO family, clean mAP and robustness disagree: YOLOv11-l has the best clean mAP on validation and the
  worst retention but one.

### 3.9 RQ2 on both page sets

| | Validation | Test |
|---|---|---|
| Within-condition ρ(mAP, WER) | −0.73 | −0.73 |
| Within-condition ρ(UW-mAP, WER) | −0.73 | −0.74 |
| Pooled ρ(mAP, WER) / ρ(UW-mAP, WER) | −0.92 / −0.91 | −0.91 / −0.90 |
| Within-condition ρ(mAP, text precision) | −0.10 | 0.03 |
| Within-condition ρ(duplicate share, text precision) | −0.91 | −0.89 |
| Best detector by mAP = best by WER | 40.5% of 37 | 27% of 37 |
| UW drop / mAP drop at severity 3 | 1.06 [1.02, 1.09] | 1.05 [1.01, 1.09] |
| Breakpoint of text recall on retention | 0.38 (F = 250) | 0.38 (F = 153) |

The best-by-mAP agreement fell from 76% with 8 detectors to 40.5% with 11 because four YOLO models now sit within 1.2 mAP
points of each other. The breakpoint moved from about 39% retention (3 detectors) to 24% (8) and back to 38% (11), so it
depends on the detector set, and we should not claim a fixed threshold.

### 3.10 Score cut-off (validation pages)

| Cut-off | Within ρ(mAP, WER) | Best by mAP = best by WER | Share of GT regions missed / merged, all pairs |
|---|---|---|---|
| 0.3 | −0.00 | 29.7% | 8.4% / 17.7% |
| 0.5 | −0.73 | 40.5% | 20.7% / 14.8% |
| 0.7 | −0.65 | 45.9% | 39.8% / 8.9% |

- At 0.3 the Docling models and DETR emit many duplicate boxes, WER triples for them (heron 0.20 → 0.70 on pairs), and
  within a condition WER follows the duplicate share (ρ 0.81), not mAP. This strengthens the RQ2 finding that mAP is
  blind to duplicates: at a lower cut-off, mAP's link to downstream quality disappears.
- The absolute failure labels depend on the cut-off: at 0.3 almost every detector mostly merges, at 0.7 almost every
  detector mostly misses. The relative ordering holds: rank correlation of (merged − missed) between cut-offs is 0.75–0.94,
  the Docling models are always the most merge-prone and YOLOv11 always the most miss-prone. The exception is Deformable DETR,
  which becomes merge-prone at 0.3. The paper should say "relatively more merging" and "relatively more missing".

### 3.11 Other analyses at 11 detectors
- Seed: seed noise 0.28–0.41 × page noise for all 11 detectors, no opposite significant calls. Seed-to-seed τ of the
  ranking: 0.92 on pairs and 0.94 on singles (0.94 and 0.96 with 8).
- Title (test pages, 19 boxes): τ singles 0.589 → 0.529 and pairs 0.323 → 0.267 without Title; RQ1 medians similar. Holds.
- Select vs test pages for the new YOLO models: YOLO26-l +6.3 [0.8, 11.9], YOLOv11-m +2.8 [−2.3, 8.6], YOLOv11-l +1.6
  [−3.0, 6.4] mAP points better on the selection pages, the same pattern as YOLO26-m (+5.7).

## 4. Decisions and why
1. **Clean training only, no flips** for RF-DETR. The study measures robustness a detector did not train for.
2. **DETR uses its detection sub-model only.** The segmentation head needed about 25 GB of VRAM.
3. **Batch size 1 for DETR and Deformable DETR.** Batching was slower for these two.
4. **Tau is reported with controls excluded.** `defocus@1` is a pixel no-op.
5. **Weight sensitivity uses the ratio of mean drops.**
6. **Both page sets are reported.** The test split is the cleaner sample (44 documents, no checkpoint selection on it); the
   validation pages carry the order and seed replicates.
7. **Page and document bootstrap side by side**, as liberal and conservative bounds (agreed 2026-11-08). A claim is
   made only when it holds under both.
8. **BCa as a check, not the primary interval, for RQ3**, because τ is discrete. For RQ1 BCa and percentile agree
   (399 percentile vs 418 BCa cells excluding zero on validation, 479 vs 490 on test).
9. **Cut-off check on validation pages only**, where the earlier downstream work was done.

## 5. Problems found
- The data loader could not pickle a local collate function under Windows spawn. Fixed with a top-level class.
- The laptop GPU reached 95–96 °C and throttled.
- Analysis scripts assumed 3 detectors. Fixed and dry-run on 7.
- DETR's config looks up `resnet50` on the Hub, so it cannot load offline.
- `p5_rq2_metric_validity.py` weight sensitivity averaged per-configuration ratios and blew up. Now the ratio of mean drops.
- The test image-quality step uses the GPU but was scheduled beside inference, which slowed YOLO inference about 5× for
  about 13 hours. It was stopped at 63 of 108 configurations and resumed after the GPU work (it resumes per configuration).
- Test mRD was missing because image quality finished after evaluation. Fixed with `--mrd-only`.
- `fig_interaction_heatmaps.png` has an empty band above the first row with more than 3 detectors. Values are correct. To fix for the paper.

## 6. Limitations this adds
- RF-DETR had about one sixth of the training pages, so it is not like for like.
- YOLO generation is confounded with training length (100 vs 40 epochs).
- Detectors are not independent: four YOLO, two heron, two egret, several DETR-family models. The distinct-architecture set
  addresses this.
- The validation pages come from 23 documents; three categories have one document each. The document bootstrap covers
  this, and the test split has 44 documents.
- Downstream and cut-off checks are on severity 2 only.

## 7. Methodology checks: status

| # | Check | Status |
|---|---|---|
| 1 | Results with and without RF-DETR | Done. No conclusion changes (3.4, 3.5) |
| 2 | Official test split | Done. Findings replicate (3.2–3.9) |
| 3 | BCa intervals | Done. Agree for RQ1; used as a check for RQ3 (3.5) |
| 4 | Score cut-off sensitivity | Done (3.10). Changes the wording of the failure-type claim |
| 5 | Damage as a confound in RQ1 and RQ2 | Open. RQ2 within-condition analysis already holds damage fixed; RQ1 not yet |
| 6 | Deformable DETR clean mAP 66.20 vs 66.17 in the baseline rerun | Open, small |
| 7 | Reproducibility: pinned versions, one-command rerun, RF-DETR weights after licence check | Open, before release |
| 8 | YOLO size × generation | Done (3.8) |
| 9 | Document-level resampling | Done (2.3, 3.3–3.6) |

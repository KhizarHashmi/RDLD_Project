# P4 - Rankings & Per-Region Analysis - Alishba (Workstream A)

**Dates:** 2026-11-04

## 1. Goal
Open up the mAP number: which layout classes break, and *how* the detectors break - do they miss
regions, merge them, split them, or mislabel them - at the operating threshold a real pipeline uses.
Also generate the second-seed replicate that tests whether RQ1 depends on one random draw.

## 2. What was done

| Step | Command | Output |
|---|---|---|
| Second-seed replicate of every severity-2 single and pair (rep = 1) | `python scripts/p4_seed_replicate.py` | `RDLD_data/degraded/r1-<config>/` (36 configs × 300 pages), `configs_rep.csv` |
| Per-class drops, per-class interaction, size buckets | `python scripts/p4_per_class.py` | `RDLD_data/results/p4/per_class/` |
| Structural error taxonomy at τ = 0.5 | `WORKERS=8 python scripts/p4_structural_errors.py` | `RDLD_data/results/p4/structural/` |

Logs: `RDLD_data/results/p4/logs/`.

**Structural taxonomy** (adapted from the LED benchmark; per page, class-agnostic geometry first, boxes
with score ≥ 0.5). Each ground-truth region gets exactly one label, in priority order: *correct*
(one-to-one IoU ≥ 0.5, same class), *misclassified* (IoU ≥ 0.5, other class), *merged* (one prediction
covers it and ≥ 1 other GT region), *split* (≥ 2 predictions lying mostly inside it), *localisation*
(best IoU 0.1–0.5), *missed* (best IoU < 0.1). Each prediction is additionally checked for
*duplicate* and *hallucination*. The classifier was unit-tested on hand-built pages for every category.

## 3. Results (evidence)

### 3.1 Class vulnerability (`p4/per_class/class_vulnerability.csv`; mean over degradations and models, severity 3)

| Class | mean AP drop s3 (pts) | median I_mult s2 / s3 | share of small boxes | n boxes |
|---|---|---|---|---|
| Formula | 29.2 | −0.21 / 0.39 | 0.00 | 50 |
| Picture | 25.6 | 1.85 / 2.81 | 0.00 | 119 |
| Caption | 21.6 | 2.11 / 2.03 | 0.14 | 66 |
| List-item | 20.4 | 0.79 / 2.24 | 0.03 | 380 |
| Page-header | 19.0 | 0.81 / 0.73 | 0.21 | 312 |
| Title | 17.2 | 0.06 / 0.15 | 0.00 | 6 |
| Table | 14.8 | 1.55 / 3.27 | 0.00 | 102 |
| Text | 14.4 | 1.61 / 1.70 | 0.02 | 1,661 |
| Page-footer | 14.4 | 0.43 / 0.45 | 0.70 | 215 |
| Section-header | 14.0 | 1.18 / 1.56 | 0.17 | 578 |
| Footnote | 8.6 | 2.76 / 3.44 | 0.00 | 41 |

- **Box size does not explain vulnerability:** Spearman(share of small boxes, drop) = −0.28, p = 0.44
  (Title excluded, n = 6). Page-footer - 70% small boxes, the weakest class on clean data - is among the
  *least* degraded.
- **Formula and Picture drop most** but barely compound; **Table and Footnote compound most**
  (median I_mult ≈ 3.3–3.4 at s3).

### 3.2 Drop by size bucket, severity 3 (`p4/per_class/drop_by_size_bucket.csv`, points)

| Detector | kind | small | medium | large |
|---|---|---|---|---|
| YOLO | single | 19.2 | 16.2 | 17.8 |
| YOLO | pair | 32.6 | 28.2 | 33.0 |
| heron | single | 13.1 | 16.1 | 17.1 |
| heron | pair | 22.2 | 29.0 | 34.0 |
| egret | single | 19.6 | 18.0 | 18.9 |
| egret | pair | 31.7 | 30.7 | 37.7 |

For the transformers, **large regions lose more than small ones** under compound degradation - the
opposite of the usual "small objects are fragile" assumption.

### 3.3 Clean error profile at τ = 0.5 (`p4/structural/structural_rates.csv`, % of GT regions)

| Detector | correct | misclassified | merged | split | localisation | missed | duplicate (% preds) |
|---|---|---|---|---|---|---|---|
| YOLO | 87.85 | 3.00 | 5.04 | 0.88 | 0.74 | 2.49 | 0.60 |
| heron | 87.59 | 3.57 | 3.99 | 0.76 | 0.71 | 3.37 | 5.06 |
| egret | 87.99 | 4.22 | 2.44 | 0.96 | 0.99 | 3.40 | 4.49 |

On clean pages the three are near-identical at the operating point (≈ 88% correct) even though their
mAPs differ by 7 points (68.8 / 64.2 / 61.5). Tables: correct 70.6% (YOLO), 73.5% (heron), 57.8% (egret).

### 3.4 Change from clean under degradation (`p4/structural/structural_vs_clean.csv`, percentage points)

| Detector | kind | sev | correct | merged | missed | table merged |
|---|---|---|---|---|---|---|
| YOLO | single | 3 | −21.3 | +5.4 | **+15.1** | +7.2 |
| YOLO | pair | 3 | −39.3 | +13.3 | **+25.7** | +17.2 |
| heron | single | 3 | −13.8 | **+7.1** | +2.0 | +8.2 |
| heron | pair | 3 | −30.6 | **+20.5** | +4.0 | +22.3 |
| egret | single | 3 | −16.9 | **+8.7** | +3.8 | +5.9 |
| egret | pair | 3 | −38.8 | **+26.6** | +7.7 | +21.8 |

**Which degradation drives which failure** (single, severity 3, change in percentage points):

| | merged: YOLO / heron / egret | missed: YOLO / heron / egret |
|---|---|---|
| vibration | 9.3 / **43.9** / **36.4** | 24.0 / 2.2 / 8.2 |
| speckle | 22.0 / 5.7 / 15.2 | **51.8** / 13.9 / 17.7 |
| texture | 8.2 / 3.1 / 6.1 | **36.7** / −1.1 / 4.7 |
| defocus | 0.1 / 2.8 / 6.4 | 1.9 / 0.7 / 0.3 |

## 4. Decisions and why
1. **Operating threshold τ = 0.5**, matching a downstream system that takes boxes as-is; mAP integrates
   over all thresholds and is reported separately by Workstream B.
2. **Class-agnostic geometry before class.** A merged paragraph+table is a structural failure whatever
   its label; mislabelling is counted separately (*misclassified*).
3. **Mutually exclusive labels per GT region** so the categories sum to 100% and can be compared as
   shares across detectors and conditions.
4. **Second seed only at severity 2** (36 configurations): the middle severity carries the most
   interactions while staying below saturation; a full replicate would have doubled the cache.

## 5. Problems found
- The first version of the matching loop in `p4_structural_errors.py` had a garbled condition; rewritten
  (`cand = np.where((iou[:, j] >= 0.5) & ~gt_taken)[0]`) and the unit tests re-run before any results
  were produced.
- Title has only 6 boxes, so its per-class AP swings widely; it is excluded from the size correlation.
- Structural rates use one threshold; `TAU=0.3 python scripts/p4_structural_errors.py` gives the
  sensitivity check if a reviewer asks.

## 6. Main takeaway
Evidence for RQ2 - mAP drops of similar size hide qualitatively different failures with
different downstream costs (a missed region loses its text; a merged table+text region corrupts
reading order and table extraction). Class vulnerability is not explained by object size
(ρ = −0.28, n.s.), and large regions degrade *more* for transformer detectors.

## 7. Hand-off
- P4 (Khizar): `r1-*` caches → predictions → seed-stability analysis.
- P5 (Khizar, RQ2): `structural_rates.csv` (gt_correct, structural_error = merged + split,
  pred_duplicate) as tier-2 layout metrics against downstream quality.
- P5 (Alishba): the merge/miss split predicts that YOLO will lose *recall* of text while the
  transformers will keep the text but damage *order* (WER) - check this in the OCR pipeline.

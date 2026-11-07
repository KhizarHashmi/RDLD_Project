# P4 - Rankings & Per-Region Analysis - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to all eight detectors on 2026-11-08)

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

**Detectors:** YOLO, heron, egret, heron101, egretxl, DETR, Deformable DETR and RF-DETR. The analyses were
first run on the first three and re-run unchanged on all eight; the per-detector numbers for the first
three are the same. Section 3.1 is a mean over detectors, so it now averages over all eight.

**Structural taxonomy** (adapted from the LED benchmark; per page, class-agnostic geometry first, boxes
with score ≥ 0.5). Each ground-truth region gets exactly one label, in priority order: *correct*
(one-to-one IoU ≥ 0.5, same class), *misclassified* (IoU ≥ 0.5, other class), *merged* (one prediction
covers it and ≥ 1 other GT region), *split* (≥ 2 predictions lying mostly inside it), *localisation*
(best IoU 0.1–0.5), *missed* (best IoU < 0.1). Each prediction is additionally checked for
*duplicate* and *hallucination*. The classifier was unit-tested on hand-built pages for every category.

## 3. Results (evidence)

### 3.1 Class vulnerability (`p4/per_class/class_vulnerability.csv`; mean over degradations and all eight detectors, severity 3)

| Class | mean AP drop s3 (pts) | median I_mult s2 / s3 | share of small boxes | n boxes |
|---|---|---|---|---|
| Formula | 25.9 | 0.44 / 0.81 | 0.00 | 50 |
| Picture | 23.7 | 1.10 / 2.06 | 0.00 | 119 |
| Page-header | 20.4 | 0.41 / 0.59 | 0.21 | 312 |
| List-item | 20.3 | 0.72 / 1.74 | 0.03 | 380 |
| Caption | 18.9 | 1.37 / 2.01 | 0.14 | 66 |
| Title | 18.7 | −0.30 / −1.80 | 0.00 | 6 |
| Table | 15.1 | 1.25 / 3.10 | 0.00 | 102 |
| Text | 14.9 | 1.43 / 1.95 | 0.02 | 1,661 |
| Section-header | 14.6 | 0.76 / 0.94 | 0.17 | 578 |
| Page-footer | 13.5 | 0.80 / 0.56 | 0.70 | 215 |
| Footnote | 11.9 | 1.60 / 1.62 | 0.00 | 41 |

- **Box size does not explain vulnerability:** Spearman(share of small boxes, drop) = −0.25, p = 0.49
  (Title excluded, n = 6). Page-footer - 70% small boxes, the weakest class on clean data - is among the
  *least* degraded.
- **Formula drops most but barely compounds** (median I_mult 0.81 at s3); **Table compounds most**
  (3.10 at s3), followed by Picture, Caption and Text (≈ 2.0).

### 3.2 Drop by size bucket, severity 3 (`p4/per_class/drop_by_size_bucket.csv`, points)

| Detector | kind | small | medium | large |
|---|---|---|---|---|
| YOLO | single | 19.2 | 16.2 | 17.8 |
| YOLO | pair | 32.6 | 28.2 | 33.0 |
| heron | single | 13.1 | 16.1 | 17.1 |
| heron | pair | 22.2 | 29.0 | 34.0 |
| egret | single | 19.6 | 18.0 | 18.9 |
| egret | pair | 31.7 | 30.7 | 37.7 |
| heron101 | single | 18.0 | 16.3 | 17.9 |
| heron101 | pair | 29.1 | 29.2 | 34.5 |
| egretxl | single | 19.3 | 16.4 | 19.3 |
| egretxl | pair | 30.6 | 28.9 | 36.2 |
| DETR | single | 5.4 | 15.0 | 18.2 |
| DETR | pair | 10.4 | 25.1 | 31.9 |
| Deformable DETR | single | 16.2 | 20.5 | 25.4 |
| Deformable DETR | pair | 26.9 | 34.4 | 43.0 |
| RF-DETR | single | 10.3 | 10.2 | 12.7 |
| RF-DETR | pair | 18.1 | 18.8 | 24.1 |

Under compound degradation **large regions lose more than small ones for all seven transformer-based
detectors** (pairs: large − small = +5.4 to +21.5 points) - the opposite of the usual "small objects are
fragile" assumption. YOLO is the exception: its drop is about the same for every size.

### 3.3 Clean error profile at τ = 0.5 (`p4/structural/structural_rates.csv`, % of GT regions)

| Detector | clean mAP | correct | misclassified | merged | split | localisation | missed | duplicate (% preds) |
|---|---|---|---|---|---|---|---|---|
| YOLO | 68.8 | 87.85 | 3.00 | 5.04 | 0.88 | 0.74 | 2.49 | 0.60 |
| heron | 64.2 | 87.59 | 3.57 | 3.99 | 0.76 | 0.71 | 3.37 | 5.06 |
| egret | 61.5 | 87.99 | 4.22 | 2.44 | 0.96 | 0.99 | 3.40 | 4.49 |
| heron101 | 65.9 | 85.75 | 3.51 | 3.94 | 1.05 | 0.76 | 4.99 | 3.24 |
| egretxl | 62.3 | 87.96 | 3.77 | 2.35 | 1.02 | 1.33 | 3.57 | 4.87 |
| DETR | 53.0 | 83.57 | 4.19 | 7.31 | 0.99 | 1.39 | 2.55 | 5.12 |
| Deformable DETR | 66.2 | 82.15 | 1.16 | 5.78 | 0.85 | 0.74 | 9.32 | 0.00 |
| RF-DETR | 48.9 | 76.18 | 4.87 | 7.62 | 1.53 | 1.70 | 8.10 | 1.54 |

On clean pages YOLO and the four Docling models (heron, egret, heron101, egretxl) are near-identical at the
operating point (86–88% correct) even though their mAPs differ by 7 points (61.5–68.8). The ranking at the
operating point does not follow mAP: Deformable DETR has the second-highest mAP (66.2) but only 82% correct,
because it misses 9% of regions; it produces no duplicate boxes at all. Tables correct: YOLO 70.6%,
heron 73.5%, egret 57.8%, heron101 61.8%, egretxl 59.8%, DETR 59.8%, Deformable DETR 61.8%, RF-DETR 59.8%.

### 3.4 Change from clean under degradation (`p4/structural/structural_vs_clean.csv`, percentage points)

| Detector | kind | sev | correct | merged | missed | table merged |
|---|---|---|---|---|---|---|
| YOLO | single | 3 | −21.3 | +5.4 | **+15.1** | +7.2 |
| YOLO | pair | 3 | −39.3 | +13.3 | **+25.7** | +17.2 |
| heron | single | 3 | −13.8 | **+7.1** | +2.0 | +8.2 |
| heron | pair | 3 | −30.6 | **+20.5** | +4.0 | +22.3 |
| egret | single | 3 | −16.9 | **+8.7** | +3.8 | +5.9 |
| egret | pair | 3 | −38.8 | **+26.6** | +7.7 | +21.8 |
| heron101 | single | 3 | −13.6 | **+7.2** | +2.4 | +8.0 |
| heron101 | pair | 3 | −30.4 | **+20.7** | +5.0 | +18.9 |
| egretxl | single | 3 | −17.1 | **+12.5** | +2.6 | +15.6 |
| egretxl | pair | 3 | −37.4 | **+34.0** | +2.0 | +34.9 |
| DETR | single | 3 | −17.4 | +5.2 | +7.8 | −2.0 |
| DETR | pair | 3 | −34.5 | +12.2 | +16.7 | +1.5 |
| Deformable DETR | single | 3 | −29.7 | +10.0 | **+19.5** | +11.0 |
| Deformable DETR | pair | 3 | −49.6 | +21.3 | **+28.3** | +23.0 |
| RF-DETR | single | 3 | −15.9 | +3.3 | **+12.3** | +7.8 |
| RF-DETR | pair | 3 | −32.3 | +8.4 | **+24.3** | +14.0 |

The detectors fall into two failure types. **The four Docling models fail by merging** (pairs: merged
+20.5 to +34.0, missed at most +7.7). **YOLO, Deformable DETR and RF-DETR fail by missing** (pairs: missed
+24.3 to +28.3). DETR does both to a similar degree. The failure type follows the model family, not
whether the detector is a transformer.

**Which degradation drives which failure** (single, severity 3, change in percentage points):

*Merged*

| Detector | vibration | speckle | texture | defocus |
|---|---|---|---|---|
| YOLO | 9.3 | 22.0 | 8.2 | 0.1 |
| heron | **43.9** | 5.7 | 3.1 | 2.8 |
| egret | **36.4** | 15.2 | 6.1 | 6.4 |
| heron101 | **39.5** | 5.8 | 5.7 | 5.0 |
| egretxl | **46.7** | 22.2 | 19.0 | 4.9 |
| DETR | 20.6 | 12.2 | 5.4 | 2.2 |
| Deformable DETR | 29.9 | **48.8** | −1.7 | 0.0 |
| RF-DETR | 8.8 | 17.7 | −0.6 | 0.2 |

*Missed*

| Detector | vibration | speckle | texture | defocus |
|---|---|---|---|---|
| YOLO | 24.0 | **51.8** | **36.7** | 1.9 |
| heron | 2.2 | 13.9 | −1.1 | 0.7 |
| egret | 8.2 | 17.7 | 4.7 | 0.3 |
| heron101 | 6.7 | 13.2 | −2.0 | 2.6 |
| egretxl | 12.0 | 6.4 | −2.2 | 2.7 |
| DETR | 24.7 | 23.0 | 8.5 | 3.0 |
| Deformable DETR | **46.0** | 34.2 | **55.3** | 0.8 |
| RF-DETR | 19.6 | **46.6** | 17.9 | 3.9 |

Vibration makes the Docling models merge; speckle and texture make YOLO, Deformable DETR and RF-DETR miss.
Defocus barely changes the structure for any detector.

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
- Title has only 6 boxes, so its per-class AP swings widely (its median interaction across the eight
  detectors is negative); it is excluded from the size correlation.
- Structural rates use one threshold; `TAU=0.3 python scripts/p4_structural_errors.py` gives the
  sensitivity check if a reviewer asks.
- The per-class figure title said "mean of 3 detectors"; it now takes the number of detectors from the
  data.

## 6. Main takeaway
Evidence for RQ2 - mAP drops of similar size hide qualitatively different failures with
different downstream costs (a missed region loses its text; a merged table+text region corrupts
reading order and table extraction). The failure type is a property of the model family: the Docling
models merge, YOLO, Deformable DETR and RF-DETR miss. Class vulnerability is not explained by object size
(ρ = −0.25, n.s.), and large regions degrade *more* for every transformer-based detector.

## 7. Hand-off
- P4 (Khizar): `r1-*` caches → predictions → seed-stability analysis.
- P5 (Khizar, RQ2): `structural_rates.csv` (gt_correct, structural_error = merged + split,
  pred_duplicate) as tier-2 layout metrics against downstream quality.
- P5 (Alishba): the merge/miss split predicts that the detectors that miss will lose *recall* of text
  while the detectors that merge will keep the text but damage *order* (WER) - check this in the OCR
  pipeline.

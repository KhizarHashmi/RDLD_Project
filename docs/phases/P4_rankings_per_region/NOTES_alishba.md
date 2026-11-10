# P4 - Rankings & Per-Region Analysis - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to eight detectors on 2026-11-08, and to eleven detectors, the test pages and the
score cut-off check on 2026-11-10)

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
| Score cut-off check, τ = 0.3 and 0.7 (validation pages) | `TAU=0.3 WORKERS=12 python scripts/p4_structural_errors.py` | `RDLD_data/results/p4/structural_tau0.3/`, `structural_tau0.7/` |
| Per-class and structural analysis on the test pages | the same with `RDLD_DATA=<test root>` | `RDLD_data_test/results/p4/` |

Logs: `RDLD_data/results/p4/logs/` and `RDLD_data_test/results/logs/`.

**Detectors (11):** YOLO26-m (`yolo`), YOLO26-l, YOLOv11-m, YOLOv11-l, heron, heron101, egret, egretxl, DETR,
Deformable DETR and RF-DETR. The analyses were first run on three detectors, then on eight, then on eleven, with the
scripts unchanged; the per-detector numbers for earlier detectors stay the same. Section 3.1 is a mean over detectors, so
it now averages over all eleven.

**Page sets:** the 300 validation pages, and 300 pages from the official DocLayNet test split (P7). The seed replicate and
the cut-off check exist for the validation pages only.

**Structural taxonomy** (adapted from the LED benchmark; per page, class-agnostic geometry first, boxes
with score ≥ 0.5). Each ground-truth region gets exactly one label, in priority order: *correct*
(one-to-one IoU ≥ 0.5, same class), *misclassified* (IoU ≥ 0.5, other class), *merged* (one prediction
covers it and ≥ 1 other GT region), *split* (≥ 2 predictions lying mostly inside it), *localisation*
(best IoU 0.1–0.5), *missed* (best IoU < 0.1). Each prediction is additionally checked for
*duplicate* and *hallucination*. The classifier was unit-tested on hand-built pages for every category.

## 3. Results (evidence)

### 3.1 Class vulnerability (`p4/per_class/class_vulnerability.csv`; mean over degradations and all eleven detectors)

| Class | val drop s3 (pts) | val median I_mult s2 / s3 | test drop s3 | test median I_mult s2 / s3 | share of small boxes val / test | n boxes val / test |
|---|---|---|---|---|---|---|
| Formula | 26.7 | 0.04 / 0.66 | 26.6 | 0.72 / 0.45 | 0.00 / 0.00 | 50 / 59 |
| Picture | 24.7 | 1.10 / 1.79 | 21.6 | 1.03 / 1.06 | 0.00 / 0.03 | 119 / 157 |
| Title | 22.6 | −0.69 / −1.98 | 15.8 | 0.06 / 0.00 | 0.00 / 0.00 | 6 / 19 |
| Page-header | 22.4 | 0.64 / 0.59 | 20.6 | 0.92 / 0.47 | 0.21 / 0.43 | 312 / 189 |
| Caption | 20.7 | 1.07 / 1.15 | 23.4 | 1.29 / 0.86 | 0.14 / 0.14 | 66 / 50 |
| List-item | 20.6 | 0.66 / 1.59 | 18.3 | 1.77 / 1.92 | 0.03 / 0.00 | 380 / 1,040 |
| Table | 17.5 | 1.25 / **2.14** | 22.1 | 1.63 / **2.24** | 0.00 / 0.00 | 102 / 117 |
| Section-header | 15.7 | 1.05 / 1.01 | 16.4 | 0.84 / 1.20 | 0.17 / 0.08 | 578 / 502 |
| Text | 15.4 | 1.41 / 1.55 | 14.9 | 1.19 / 2.16 | 0.02 / 0.04 | 1,661 / 1,882 |
| Page-footer | 15.3 | 0.97 / 0.53 | 18.7 | 0.11 / −0.02 | 0.70 / 0.50 | 215 / 307 |
| Footnote | 13.2 | 1.26 / 0.87 | 23.5 | 0.86 / 0.85 | 0.00 / 0.00 | 41 / 29 |

- **Box size does not explain vulnerability:** Spearman(share of small boxes, drop), Title excluded: ρ = −0.20, p = 0.58
  (validation); ρ = −0.48, p = 0.16 (test). Page-footer, with the most small boxes, is among the least degraded on validation.
- **Formula drops most on both page sets** (26.7 and 26.6 points at s3) but compounds little; **Table compounds most at
  s3 on both** (median I_mult 2.14 and 2.24). The middle of the order moves between page sets: Footnote has 41 and 29 boxes,
  so its drop is noisy.

### 3.2 Drop by size bucket, severity 3 (`p4/per_class/drop_by_size_bucket.csv`, points)

| Detector | val single S / M / L | val pair S / M / L | test pair S / M / L |
|---|---|---|---|
| YOLO26-m | 19.2 / 16.2 / 17.8 | 32.6 / 28.2 / 33.0 | 40.2 / 32.0 / 37.1 |
| YOLO26-l | 20.6 / 16.7 / 18.4 | 34.7 / 28.2 / 32.2 | 34.5 / 33.8 / 34.3 |
| YOLOv11-m | 27.2 / 22.9 / 22.6 | 43.4 / 37.0 / 42.1 | 42.3 / 43.5 / 49.3 |
| YOLOv11-l | 26.4 / 21.8 / 28.3 | 43.7 / 36.3 / 47.2 | 40.2 / 42.0 / 51.5 |
| heron | 13.1 / 16.1 / 17.1 | 22.2 / 29.0 / 33.9 | 21.7 / 23.6 / 36.5 |
| heron101 | 18.0 / 16.3 / 17.9 | 29.1 / 29.2 / 34.5 | 24.2 / 23.7 / 37.7 |
| egret | 19.6 / 18.0 / 18.9 | 31.7 / 30.7 / 37.6 | 20.8 / 27.0 / 46.3 |
| egretxl | 19.3 / 16.4 / 19.3 | 30.6 / 28.9 / 36.2 | 20.7 / 24.9 / 43.5 |
| DETR | 5.4 / 15.0 / 18.2 | 10.4 / 25.1 / 31.9 | 9.3 / 15.5 / 29.0 |
| Def. DETR | 16.2 / 20.5 / 25.4 | 26.9 / 34.4 / 43.0 | 31.8 / 37.9 / 46.3 |
| RF-DETR | 10.3 / 10.2 / 12.7 | 18.1 / 18.8 / 24.1 | 12.8 / 18.5 / 26.8 |

Under compound degradation **large regions lose more than small ones for all seven non-YOLO detectors on both page
sets** (pairs: large − small = +5.4 to +21.5 points on validation, +13.5 to +25.5 on test) - the opposite of the usual
"small objects are fragile" assumption. YOLO26 shows no size effect on either page set. YOLOv11 has none on validation
(−1.4 and +3.5) but the transformer pattern on test (+7.0 and +11.3).

### 3.3 Clean error profile at τ = 0.5 (`p4/structural/structural_rates.csv`, % of GT regions)

Validation pages:

| Detector | clean mAP | correct | misclassified | merged | split | localisation | missed | duplicate (% preds) | tables correct |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26-m | 68.8 | 87.85 | 3.00 | 5.04 | 0.88 | 0.74 | 2.49 | 0.60 | 70.6 |
| YOLO26-l | 68.2 | 87.93 | 2.38 | 4.84 | 0.91 | 0.71 | 3.23 | 0.46 | 67.6 |
| YOLOv11-m | 68.9 | 84.84 | 2.86 | 4.73 | 0.74 | 0.68 | 6.15 | 1.12 | 67.6 |
| YOLOv11-l | 69.4 | 84.76 | 2.29 | 5.10 | 0.79 | 0.57 | 6.49 | 0.81 | 68.6 |
| heron | 64.2 | 87.59 | 3.57 | 3.99 | 0.76 | 0.71 | 3.37 | 5.06 | 73.5 |
| heron101 | 65.9 | 85.75 | 3.51 | 3.94 | 1.05 | 0.76 | 4.99 | 3.24 | 61.8 |
| egret | 61.5 | 87.99 | 4.22 | 2.44 | 0.96 | 0.99 | 3.40 | 4.49 | 57.8 |
| egretxl | 62.3 | 87.96 | 3.77 | 2.35 | 1.02 | 1.33 | 3.57 | 4.87 | 59.8 |
| DETR | 53.0 | 83.57 | 4.19 | 7.31 | 0.99 | 1.39 | 2.55 | 5.12 | 59.8 |
| Def. DETR | 66.2 | 82.15 | 1.16 | 5.78 | 0.85 | 0.74 | 9.32 | 0.00 | 61.8 |
| RF-DETR | 48.9 | 76.18 | 4.87 | 7.62 | 1.53 | 1.70 | 8.10 | 1.54 | 59.8 |

Test pages:

| Detector | clean mAP | correct | misclassified | merged | split | localisation | missed | duplicate (% preds) | tables correct |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26-m | 71.4 | 84.21 | 5.84 | 3.93 | 0.34 | 0.21 | 5.47 | 0.70 | 87.2 |
| YOLO26-l | 73.1 | 83.38 | 4.53 | 4.05 | 0.23 | 0.14 | 7.68 | 0.43 | 88.0 |
| YOLOv11-m | 71.3 | 81.91 | 3.70 | 3.31 | 0.28 | 0.37 | 10.43 | 1.44 | 87.2 |
| YOLOv11-l | 72.5 | 82.60 | 4.21 | 3.13 | 0.21 | 0.21 | 9.65 | 0.73 | 85.5 |
| heron | 65.7 | 81.91 | 11.19 | 4.02 | 1.54 | 0.37 | 0.97 | 7.78 | 82.9 |
| heron101 | 66.1 | 82.30 | 11.12 | 3.72 | 1.31 | 0.28 | 1.26 | 8.06 | 80.3 |
| egret | 64.4 | 82.60 | 7.72 | 4.92 | 1.59 | 0.44 | 2.74 | 8.21 | 81.2 |
| egretxl | 63.9 | 82.69 | 10.30 | 2.92 | 1.49 | 0.60 | 2.00 | 8.63 | 81.2 |
| DETR | 41.3 | 75.82 | 9.15 | 8.25 | 0.37 | 2.23 | 4.18 | 7.62 | 83.8 |
| Def. DETR | 70.4 | 78.35 | 2.21 | 3.19 | 0.18 | 0.18 | 15.88 | 0.06 | 80.3 |
| RF-DETR | 49.7 | 69.04 | 6.78 | 9.33 | 0.64 | 1.52 | 12.69 | 1.28 | 71.8 |

On clean validation pages YOLO26 and the four Docling models are near-identical at the operating point (86–88% correct)
even though their mAPs differ by 7 points. The ranking at the operating point does not follow mAP: Deformable DETR has a
high mAP but the most missed regions (9% validation, 16% test) and no duplicate boxes. YOLOv11 misses about twice as many
regions as YOLO26 on clean pages (6.2–6.5% vs 2.5–3.2% on validation). On the test pages the Docling models mislabel more
(7.7–11.2% misclassified) and duplicate more (7.8–8.6% of predictions) than on validation.

### 3.4 Change from clean under degradation (`p4/structural/structural_vs_clean.csv`, severity 3, percentage points)

| Detector | kind | correct val / test | merged val / test | missed val / test | table merged val / test |
|---|---|---|---|---|---|
| YOLO26-m | single | −21.3 / −21.1 | +5.4 / +5.6 | **+15.1** / **+15.8** | +7.2 / +4.3 |
| YOLO26-m | pair | −39.3 / −37.1 | +13.3 / +12.9 | **+25.7** / **+25.0** | +17.2 / +11.9 |
| YOLO26-l | single | −22.0 / −21.2 | +9.1 / +7.7 | **+12.5** / **+13.8** | +10.9 / +8.0 |
| YOLO26-l | pair | −39.3 / −36.1 | +15.7 / +14.6 | **+23.8** / **+22.5** | +18.6 / +13.0 |
| YOLOv11-m | single | −32.1 / −31.2 | +5.1 / +7.5 | **+27.6** / **+24.2** | +8.0 / +6.0 |
| YOLOv11-m | pair | −54.6 / −52.8 | +14.1 / +17.5 | **+42.2** / **+36.9** | +17.2 / +14.1 |
| YOLOv11-l | single | −30.8 / −29.8 | +5.8 / +9.3 | **+24.9** / **+21.6** | +4.4 / +9.4 |
| YOLOv11-l | pair | −53.1 / −51.6 | +11.7 / +15.9 | **+42.2** / **+37.7** | +9.7 / +14.9 |
| heron | single | −13.8 / −13.9 | **+7.1** / **+9.9** | +2.0 / +2.6 | +8.2 / +8.7 |
| heron | pair | −30.6 / −30.6 | **+20.5** / **+24.6** | +4.0 / +5.1 | +22.3 / +23.9 |
| heron101 | single | −13.6 / −15.6 | **+7.2** / **+10.8** | +2.4 / +3.2 | +8.0 / +12.0 |
| heron101 | pair | −30.4 / −32.5 | **+20.7** / **+25.6** | +5.0 / +6.6 | +18.9 / +27.3 |
| egret | single | −16.9 / −18.6 | **+8.7** / **+10.5** | +3.8 / +4.7 | +5.9 / +10.4 |
| egret | pair | −38.8 / −38.8 | **+26.6** / **+28.6** | +7.7 / +8.0 | +21.8 / +30.2 |
| egretxl | single | −17.1 / −15.4 | **+12.5** / **+14.8** | +2.6 / +1.6 | +15.6 / +18.7 |
| egretxl | pair | −37.4 / −36.6 | **+34.0** / **+37.4** | +2.0 / +2.2 | +34.9 / +42.0 |
| DETR | single | −17.4 / −14.9 | +5.2 / +6.6 | +7.8 / +5.8 | −2.0 / +4.5 |
| DETR | pair | −34.5 / −29.7 | +12.2 / +13.9 | +16.7 / +13.1 | +1.5 / +11.6 |
| Def. DETR | single | −29.7 / −26.4 | +10.0 / +11.3 | **+19.5** / **+15.1** | +11.0 / +7.9 |
| Def. DETR | pair | −49.6 / −45.6 | +21.3 / +23.3 | **+28.3** / **+22.4** | +23.0 / +17.9 |
| RF-DETR | single | −15.9 / −15.3 | +3.3 / +4.4 | **+12.3** / **+11.5** | +7.8 / +2.7 |
| RF-DETR | pair | −32.3 / −30.0 | +8.4 / +9.7 | **+24.3** / **+22.0** | +14.0 / +5.7 |

The detectors fall into two failure types, on both page sets. **The four Docling models fail by merging** (pairs: merged
+20.5 to +37.4, missed at most +8.0). **The four YOLO models, Deformable DETR and RF-DETR fail by missing** (pairs: missed
+22.0 to +42.2). DETR does both to a similar degree. YOLOv11 is the most extreme: it loses more than half of its correct
regions on severity-3 pairs. On test pages Deformable DETR misses and merges about equally on pairs (+22.4 vs +23.3). The
failure type follows the model family, not whether the detector is a transformer.

**Which degradation drives which failure** (single, severity 3, change in percentage points, validation / test):

*Merged*

| Detector | vibration | speckle | texture | defocus |
|---|---|---|---|---|
| YOLO26-m | 9.3 / 17.2 | 22.0 / 20.3 | 8.2 / 5.3 | 0.1 / 1.2 |
| YOLO26-l | 5.4 / 11.5 | **53.4** / **44.1** | 7.9 / 5.0 | 1.8 / 0.0 |
| YOLOv11-m | 16.0 / 37.9 | 14.3 / 11.8 | −1.1 / 0.7 | 6.9 / 9.3 |
| YOLOv11-l | 11.1 / 42.6 | 28.2 / 23.0 | 1.4 / 0.3 | 3.6 / 2.9 |
| heron | **43.9** / **64.1** | 5.7 / 5.1 | 3.1 / 1.3 | 2.8 / 5.1 |
| heron101 | **39.5** / **63.2** | 5.8 / 5.9 | 5.7 / 2.1 | 5.0 / 12.5 |
| egret | **36.4** / **60.5** | 15.2 / 7.3 | 6.1 / 6.0 | 6.4 / 6.0 |
| egretxl | **46.7** / **67.9** | 22.2 / 16.1 | 19.0 / 21.4 | 4.9 / 5.1 |
| DETR | 20.6 / 24.8 | 12.2 / 14.0 | 5.4 / 4.2 | 2.2 / 5.7 |
| Def. DETR | 29.9 / 48.3 | **48.8** / 38.0 | −1.7 / 1.9 | 0.0 / −0.5 |
| RF-DETR | 8.8 / 15.1 | 17.7 / 16.1 | −0.6 / −0.5 | 0.2 / 0.7 |

*Missed*

| Detector | vibration | speckle | texture | defocus |
|---|---|---|---|---|
| YOLO26-m | 24.0 / 32.8 | **51.8** / 39.7 | **36.7** / 33.6 | 1.9 / 4.4 |
| YOLO26-l | 30.5 / 30.5 | 32.7 / 36.3 | 30.0 / 26.0 | 1.6 / 2.5 |
| YOLOv11-m | **53.8** / 39.5 | **56.9** / **53.7** | **69.6** / **52.1** | **33.4** / **38.5** |
| YOLOv11-l | **58.0** / 36.1 | **51.0** / 47.7 | **56.2** / 49.4 | **27.4** / **33.6** |
| heron | 2.2 / 4.2 | 13.9 / 10.2 | −1.1 / 2.1 | 0.7 / 2.4 |
| heron101 | 6.7 / 5.6 | 13.2 / 13.1 | −2.0 / 2.7 | 2.6 / 0.6 |
| egret | 8.2 / 5.3 | 17.7 / 18.4 | 4.7 / 5.9 | 0.3 / 1.7 |
| egretxl | 12.0 / 6.6 | 6.4 / 4.3 | −2.2 / −0.5 | 2.7 / −0.0 |
| DETR | 24.7 / 22.1 | 23.0 / 13.5 | 8.5 / 6.9 | 3.0 / 3.1 |
| Def. DETR | 46.0 / 26.6 | 34.2 / 39.5 | **55.3** / 41.3 | 0.8 / −0.2 |
| RF-DETR | 19.6 / 22.1 | **46.6** / 39.1 | 17.9 / 14.3 | 3.9 / 5.2 |

Vibration makes the Docling models merge (more so on the test pages); speckle and texture make the YOLO models, Deformable
DETR and RF-DETR miss. Defocus barely changes the structure for any detector **except YOLOv11**, which misses 27–39% more
regions under defocus@3 on both page sets. YOLO26 does not (1.6–4.4). This is the clearest structural difference between the
two YOLO generations.

### 3.5 Score cut-off (validation pages, `p4/structural_tau0.3/`, `p4/structural_tau0.7/`)

Mean over all pairs and all eleven detectors (% of GT regions; duplicates % of predictions):

| τ | merged | missed | duplicate |
|---|---|---|---|
| 0.3 | 17.7 | 8.4 | 9.9 |
| 0.5 | 14.8 | 20.7 | 3.3 |
| 0.7 | 8.9 | 39.8 | 0.3 |

Severity-3 pairs, change in merged − change in missed (percentage points; positive = mostly merging):

| Detector | τ = 0.3 | τ = 0.5 | τ = 0.7 |
|---|---|---|---|
| egretxl | +28.6 | +32.0 | +4.2 |
| egret | +25.6 | +18.9 | −5.7 |
| heron | +18.0 | +16.5 | −6.3 |
| heron101 | +18.4 | +15.7 | −14.5 |
| DETR | +13.0 | −4.6 | −22.4 |
| Def. DETR | +25.3 | −7.1 | −41.8 |
| YOLO26-l | +7.0 | −8.1 | −24.4 |
| YOLO26-m | +7.5 | −12.4 | −32.9 |
| RF-DETR | +13.1 | −15.9 | −32.4 |
| YOLOv11-m | −2.9 | −28.1 | −47.6 |
| YOLOv11-l | +4.0 | −30.4 | −50.6 |

A lower cut-off keeps more weak boxes, so more regions are covered by an over-large or duplicate box (merged) and fewer are
missed; a higher cut-off does the opposite. The absolute labels therefore depend on τ: at 0.3 nearly every detector mostly
merges, at 0.7 nearly every detector mostly misses. **The relative order holds**: the rank correlation of the column above
between cut-offs is 0.75–0.94, the Docling models are the most merge-prone and YOLOv11 the most miss-prone at every cut-off.
The exception is Deformable DETR, which moves to the merge side at 0.3. The failure-type claim should be stated as "relatively
more merging" and "relatively more missing", with the cut-off given.

## 4. Decisions and why
1. **Operating threshold τ = 0.5**, matching a downstream system that takes boxes as-is; mAP integrates
   over all thresholds and is reported separately by Workstream B. 0.3 and 0.7 are reported as a sensitivity check.
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
- Title has only 6 boxes on the validation pages (19 on test), so its per-class AP swings widely; it is excluded from the
  size correlation.
- The failure type depends on the score cut-off in absolute terms (§3.5). Only the relative tendency is a finding.
- The per-class figure title said "mean of 3 detectors"; it now takes the number of detectors from the data.

## 6. Main takeaway
Evidence for RQ2 - mAP drops of similar size hide qualitatively different failures with
different downstream costs (a missed region loses its text; a merged table+text region corrupts
reading order and table extraction). The failure tendency is a property of the model family and holds on both page sets
and at every cut-off in relative terms: the Docling models merge more, the YOLO models, Deformable DETR and RF-DETR miss
more. Within the YOLO family, generation matters: YOLOv11 misses far more than YOLO26, even under defocus. Class
vulnerability is not explained by object size, and large regions degrade *more* for every non-YOLO detector.

## 7. Hand-off
- P4 (Khizar): `r1-*` caches → predictions → seed-stability analysis.
- P5 (Khizar, RQ2): `structural_rates.csv` (gt_correct, structural_error = merged + split,
  pred_duplicate) as tier-2 layout metrics against downstream quality.
- P5 (Alishba): the merge/miss split predicts that the detectors that miss will lose *recall* of text
  while the detectors that merge will keep the text but damage *order* (WER) - check this in the OCR
  pipeline.

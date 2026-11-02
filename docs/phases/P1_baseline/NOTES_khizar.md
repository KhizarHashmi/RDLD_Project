# P1 - Clean Baseline - Khizar (Workstream B)

**Dates:** 2026-11-03

## 1. Goal
Establish the clean-condition reference every degradation effect is measured against: mAP with an
uncertainty band, per class and per document category, the clean label-instability floor flagged in
P0, and a fixed post-processing policy.

## 2. What was done

| Item | File |
|---|---|
| One interface over the 3 detectors + label mapping | `rdld/detectors.py` |
| COCO mAP, per-class AP, instability, dedup | `rdld/evaluation.py` |
| Fast resampled mAP (paired bootstrap engine) | `rdld/fastcoco.py` |
| Baseline run | `python scripts/p1_clean_baseline.py` → `RDLD_data/results/p1/baseline/` |

Settings: score floor 0.01 (COCO mAP integrates the full PR curve), batch 8, YOLO at imgsz 1024 (its
training size), max 300 detections/page. Predictions cached to `RDLD_data/predictions/<model>/clean.json`.
Log: `RDLD_data/results/p1/logs/p1_clean_baseline.log`.

## 3. Results (evidence)

### 3.1 Clean accuracy (`baseline/summary.csv`), 300 pages, 1,000-resample page bootstrap

| Detector | Family | mAP@[.5:.95] | 95% CI | mAP@.5 | mAP@.75 | ms/page* | VRAM |
|---|---|---|---|---|---|---|---|
| YOLO (`hantian/yolo-doclaynet`) | CNN one-stage | **68.8** | 64.0 – 72.5 | 87.6 | 70.8 | 50 | 1.83 GB |
| heron (`docling-layout-heron`) | RT-DETR v2 | 64.2 | 60.4 – 68.4 | **89.1** | 64.2 | 75 | 1.21 GB |
| egret (`docling-layout-egret-medium`) | D-FINE | 61.5 | 58.3 – 65.8 | 86.1 | 62.1 | 51 | 0.98 GB |

\* end-to-end incl. PNG decode, batch 8, RTX 3080 Laptop, machine otherwise idle.

**The ranking depends on the metric already on clean data:** YOLO leads on mAP@[.5:.95] and mAP@.75,
heron leads on mAP@.5. Transformers find regions but localise them less tightly.

### 3.2 Per class (`baseline/per_class_ap.csv`, `fig_per_class_ap.png`)
Weakest classes for all three: **Page-footer** (YOLO 56.5, heron 32.7, egret 27.4) and Footnote (43–53).
Strongest: Picture (81–87), Text (75–79). Title (n = 6) swings from 43.5 to 73.2 between models - noise.

### 3.3 Per document category (`baseline/per_category_map.csv`, `fig_per_category_map.png`)
Scientific articles easiest (77–90). **Manuals** are hardest for the transformers (heron 42.2, egret 37.8;
YOLO 59.2); **financial reports** hardest for YOLO (54.6). Read with the class-mix caveat from Alishba's
P1 EDA.

### 3.4 Label-instability floor (`baseline/instability.csv`)
Share of predictions sitting on the same region (IoU ≥ 0.9) as a differently-labelled prediction:

| Detector | score ≥ 0.3 | score ≥ 0.5 |
|---|---|---|
| YOLO | 3.3% | 0.9% |
| heron | **28.0%** | 9.7% |
| egret | **26.3%** | 8.4% |

### 3.5 Dedup policy (`summary.csv: mAP_dedup`)
Class-agnostic suppression of same-region boxes (IoU ≥ 0.9, keep highest score) **lowers** mAP for every
model: YOLO −0.8, heron −4.5, egret −4.3 points (Title: heron 60.7 → 31.1, egret 43.5 → 13.5).

### 3.6 Determinism
Second inference pass on 48 pages: identical prediction set (scores to 4 dp, boxes to 0.1 px) and
identical mAP for all three detectors. The pipeline is deterministic, so differences between conditions
are not run-to-run noise.

### 3.7 Unmapped predictions
heron and egret share docling's **17-class** space (not 12 as the Sept. notes said); 6 classes have no
DocLayNet equivalent and are dropped: 9,509 (heron) and 13,323 (egret) boxes at the 0.01 floor, mostly
`key_value_region` (≈146 of 2,230 on an 8-page sample).

## 4. Decisions and why
1. **Evaluate raw output, no dedup.** It is standard COCO practice and dedup hurts every model. But the
   reason it hurts is a finding (§5), and P4/P5 measure the same output at an operating threshold.
2. **Bootstrap CI as the noise floor.** Clean mAP SD ≈ 2 points from page sampling alone; any single
   condition difference smaller than that is not interpretable without pairing. → P3 uses a **paired**
   bootstrap with one shared resample index across all configurations and models.
3. **Fast bootstrap engine** (`rdld/fastcoco.py`): COCO matching once per prediction set, accumulation
   re-done per resample. Verified to reproduce pycocotools' mAP exactly (YOLO 0.68797, heron 0.64231),
   ~10–70 ms per resample instead of seconds.
4. **Drop, never relabel, unmapped classes** (as in the proposal) - and record the counts.

## 5. Problems found
- **mAP rewards label hedging.** Transformers emit the same box with two labels; under COCO mAP the wrong
  copy is a cheap low-score false positive while the right copy earns credit, so hedging *raises* mAP.
  A downstream system taking boxes above a threshold receives both copies. This is direct evidence for
  RQ2 (mAP vs. usable output) before any degradation is applied.
- Published heron mAP is 0.69 on DocLayNet; we measure 64.2 on our 300-page stratified subset, within
  the subset's CI band of class/category composition effects but lower - reported, not tuned.
- The original determinism check compared prediction lists position by position and reported false
  mismatches for the transformers (top-k queries with near-tied scores can reorder). Replaced by a
  set comparison; verified identical.
- First baseline run's timings were inflated by concurrent cache generation (167–591 ms/page); the
  reported timings come from a re-run on an idle machine.

## 6. Main takeaway
O3 achieved. Clean-condition prediction instability characterised (≈27% cross-label duplicates
for transformer heads at 0.3), separating detector noise from degradation effects. The page-bootstrap
SD (~2 points) motivates the paired design used for RQ1.

## 7. Hand-off
- P2/P3: `rdld/detectors.py` and the prediction-cache format are the interface; clean predictions are
  reused by the degraded-condition runs.
- RQ2 (P5): the hedging finding - compare mAP with operating-threshold structural errors (Alishba's P4).

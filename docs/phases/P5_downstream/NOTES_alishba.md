# P5 - Downstream Link - Alishba (Workstream A)

**Dates:** 2026-11-04

## 1. Goal
Build the downstream pipeline that RQ2 needs: page → layout regions → text in reading order, and measure
how much of the text each detector's layout delivers correctly - with the OCR engine held exactly
constant so that every difference is caused by layout alone.

## 2. What was done

| Step | Command | Output |
|---|---|---|
| OCR + region pipeline + scoring | `WORKERS=12 OMP_THREAD_LIMIT=1 python scripts/p5_ocr_downstream.py` | `RDLD_data/ocr/<config>/<page>.json` (cached words), `RDLD_data/results/p5/downstream/{downstream_pages.csv, downstream_configs.csv}` |

Log: `RDLD_data/results/p5/logs/p5_ocr_downstream.log`.

**Pipeline.** Tesseract 5.5.3 (`--psm 3`, English) reads each page **once** per condition; its words are
assigned to regions by word centre. Two pipelines then read the *same words from the same image*:

- **reference:** ground-truth regions → text
- **detector:** predicted regions with score ≥ 0.5 → text

Regions are read in recursive XY-cut order (horizontal cut first), words within a region line by line;
Picture regions are not read; a word inside two regions is read twice (as cropping each box would).

**Metrics** per page and detector, aggregated per condition weighted by words:
- **WER** - word edit distance between detector text and reference text / reference words (content *and* order)
- **text recall** - share of reference words (multiset) the detector pipeline reads at all
- **text precision** - share of detector-pipeline words that exist in the reference (penalises duplicates)
- **table recall** - share of words inside GT Tables that are read by a predicted Table region

Scope: all 300 pages × 37 conditions (clean + every severity-2 single and pair) = 11,100 OCR passes,
× 3 detectors.

## 3. Results (evidence)

### 3.1 Clean condition (`p5/downstream/downstream_configs.csv`; 77,732 reference words)

| Detector | mAP (P1) | WER | text recall | text precision | table recall |
|---|---|---|---|---|---|
| YOLO | 68.8 | **0.022** | 0.987 | **0.994** | 0.882 |
| heron | 64.2 | 0.112 | 0.954 | 0.938 | 0.763 |
| egret | 61.5 | 0.102 | 0.955 | 0.946 | 0.715 |

At the operating threshold all three label ≈ 88% of regions correctly (P4), and their mAPs differ by at
most 7 points - but **YOLO's WER is 5× lower** than the transformers'. Their lower text precision is the
P1 hedging effect arriving downstream: duplicate boxes on the same region make the pipeline read the same
text twice.

### 3.2 Severity-2 means (`downstream_configs.csv`)

| Detector | kind | WER | text recall | text precision | table recall |
|---|---|---|---|---|---|
| YOLO | single | 0.103 | 0.907 | 0.991 | 0.716 |
| YOLO | pair | 0.234 | **0.775** | 0.990 | 0.594 |
| heron | single | 0.144 | 0.954 | 0.911 | 0.674 |
| heron | pair | 0.198 | 0.938 | 0.880 | 0.515 |
| egret | single | 0.155 | 0.945 | 0.910 | 0.598 |
| egret | pair | 0.229 | 0.895 | 0.882 | 0.442 |

The P4 prediction holds: **YOLO, which fails by *missing* regions, loses text recall** (0.987 → 0.775 on
pairs); **the transformers, which fail by *merging*, keep recall** (≥ 0.89) and lose order and precision
instead. On compound degradation the three end with similar WER (0.20–0.23) for different reasons.

### 3.3 OCR coverage
Pages where Tesseract returns no words at all: clean 7 / 300 (white text on photographs, picture-only
pages), texture@2 29 / 300, speckle@2 6 / 300. These pages carry zero weight in the word-weighted
aggregates; both pipelines see the same (empty) word set, so they cannot bias the comparison.

## 4. Decisions and why
1. **OCR once per page, words assigned to regions** instead of OCR per crop. Cropping would make OCR
   quality depend on crop size and padding, mixing an OCR effect into the layout effect. With one OCR pass
   the text available is identical for both pipelines; only the layout differs.
2. **Reference = GT regions on the *same degraded* image**, not clean text: the question is how much the
   detector's layout costs given the image, not how much the degradation costs the OCR.
3. **Word-weighted aggregation**, so a dense page counts more than a near-empty one and empty pages count
   zero.
4. **Severity 2 only** (37 conditions): the downstream link is the "droppable" phase in the plan; severity 2
   is where interactions are largest. Extending to s1/s3 is a re-run with `SEV` changed.

## 5. Problems found
- **First OCR run silently produced no text** for all 11,100 pages. Cause: Tesseract could not find its
  language data (`TESSDATA_PREFIX` is set by conda activation, not when the env's python is called
  directly), and the `tsv` config file is absent in the conda-forge package; the script swallowed the error.
  Fixed: set `TESSDATA_PREFIX` from the env path, use `-c tessedit_create_tsv=1`, and **fail loudly** on a
  non-zero exit or an empty first page. Empty cache deleted and re-run.
- **Thread oversubscription.** 12 Tesseract processes each spawning OpenMP threads gave ≈ 0.5 pages/s;
  `OMP_THREAD_LIMIT=1` gave ≈ 3.6 pages/s (7×).
- Tesseract cannot read light text on photographs; affected pages are excluded by weighting (§3.3).

## 6. Main takeaway
Downstream outcome for RQ2. The failure *type* found in P4 predicts the downstream *symptom*:
missed regions → lost text (YOLO recall 0.78 on pairs); merged regions → wrong order and duplicated text
(transformers' precision 0.88). Similar WERs can hide different failures.

## 7. Hand-off
- P5 (Khizar): `downstream_configs.csv` → correlation of each layout metric with WER / recall / precision /
  table recall; ranking agreement; threshold fit.

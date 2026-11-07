# P5 - Downstream Link - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to all eight detectors on 2026-11-08)

## 1. Goal
Build the downstream pipeline that RQ2 needs: page → layout regions → text in reading order, and measure
how much of the text each detector's layout delivers correctly - with the OCR engine held exactly
constant so that every difference is caused by layout alone.

## 2. What was done

| Step | Command | Output |
|---|---|---|
| OCR + region pipeline + scoring | `WORKERS=12 OMP_THREAD_LIMIT=1 python scripts/p5_ocr_downstream.py` | `RDLD_data/ocr/<config>/<page>.json` (cached words), `RDLD_data/results/p5/downstream/{downstream_pages.csv, downstream_configs.csv}` |

Logs: `RDLD_data/results/p5/logs/p5_ocr_downstream.log` (first three detectors) and
`p5_ocr_downstream_8det.log` (all eight).

**Detectors:** YOLO, heron, egret, heron101, egretxl, DETR, Deformable DETR and RF-DETR. The numbers for
the first three are the same as in the first run.

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
× 8 detectors.

## 3. Results (evidence)

### 3.1 Clean condition (`p5/downstream/downstream_configs.csv`; 77,732 reference words)

| Detector | mAP (P1) | WER | text recall | text precision | table recall |
|---|---|---|---|---|---|
| YOLO | 68.8 | **0.022** | 0.987 | 0.994 | 0.882 |
| heron | 64.2 | 0.112 | 0.954 | 0.938 | 0.763 |
| egret | 61.5 | 0.102 | 0.955 | 0.946 | 0.715 |
| heron101 | 65.9 | 0.098 | 0.952 | 0.952 | 0.702 |
| egretxl | 62.3 | 0.105 | 0.955 | 0.942 | 0.711 |
| DETR | 53.0 | 0.185 | 0.987 | 0.858 | **0.891** |
| Deformable DETR | 66.2 | 0.058 | 0.948 | **0.999** | 0.779 |
| RF-DETR | 48.9 | 0.084 | 0.947 | 0.975 | 0.795 |

**YOLO's WER is 4–5× lower** than the four Docling models' (0.098–0.112), even though their mAPs differ by
at most 7 points. Low text precision is the P1 hedging effect arriving downstream: duplicate boxes on the
same region make the pipeline read the same text twice. The detectors with the fewest duplicate boxes (P4)
have the highest text precision - Deformable DETR (no duplicates, precision 0.999) and YOLO - and DETR, with
the most duplicates, has the lowest precision (0.858) and the highest WER (0.185) although it reads almost
every word (recall 0.987). WER does not follow mAP: RF-DETR has the lowest mAP (48.9) but a lower WER than
any Docling model.

### 3.2 Severity-2 means (`downstream_configs.csv`)

| Detector | kind | WER | text recall | text precision | table recall |
|---|---|---|---|---|---|
| YOLO | single | 0.103 | 0.907 | 0.991 | 0.716 |
| YOLO | pair | 0.234 | **0.775** | 0.990 | 0.594 |
| heron | single | 0.144 | 0.954 | 0.911 | 0.674 |
| heron | pair | 0.198 | 0.938 | 0.880 | 0.515 |
| egret | single | 0.155 | 0.945 | 0.910 | 0.598 |
| egret | pair | 0.229 | 0.895 | 0.882 | 0.442 |
| heron101 | single | 0.133 | 0.953 | 0.921 | 0.618 |
| heron101 | pair | 0.181 | 0.935 | 0.893 | 0.477 |
| egretxl | single | 0.153 | 0.941 | 0.912 | 0.615 |
| egretxl | pair | 0.220 | 0.902 | 0.883 | 0.467 |
| DETR | single | 0.262 | 0.951 | 0.827 | 0.781 |
| DETR | pair | 0.367 | 0.876 | 0.793 | 0.635 |
| Deformable DETR | single | 0.257 | 0.748 | 0.997 | 0.535 |
| Deformable DETR | pair | 0.445 | **0.560** | 0.995 | 0.345 |
| RF-DETR | single | 0.151 | 0.882 | 0.968 | 0.703 |
| RF-DETR | pair | 0.232 | **0.803** | 0.959 | 0.585 |

The P4 prediction holds for all eight detectors. **The detectors that fail by *missing* regions lose text
recall** on pairs: YOLO 0.987 → 0.775, Deformable DETR 0.948 → 0.560, RF-DETR 0.947 → 0.803. **The four
Docling models, which fail by *merging*, keep recall** (≥ 0.89) and lose order and precision instead.
DETR, which does both, loses some of each and has the lowest precision (0.793). On compound degradation
YOLO, RF-DETR and the Docling models end with similar WER (0.18–0.23) for different reasons; DETR (0.367)
and Deformable DETR (0.445) are clearly worse. Deformable DETR has the second-lowest WER on clean pages and
the highest on pairs.

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
Downstream outcome for RQ2. The failure *type* found in P4 predicts the downstream *symptom* for all eight
detectors: missed regions → lost text (recall on pairs 0.56–0.80 for YOLO, Deformable DETR and RF-DETR);
merged regions → wrong order and duplicated text (Docling models' precision 0.88–0.89). Similar WERs can
hide different failures, and neither clean mAP nor clean WER predicts which detector holds up downstream
under compound degradation.

## 7. Hand-off
- P5 (Khizar): `downstream_configs.csv` → correlation of each layout metric with WER / recall / precision /
  table recall; ranking agreement; threshold fit.

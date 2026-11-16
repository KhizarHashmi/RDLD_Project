# P5 - Downstream Link - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to eight detectors on 2026-11-08, and to eleven detectors, the test pages and the
score cut-off check on 2026-11-10)

## 1. Goal
Build the downstream pipeline that RQ2 needs: page → layout regions → text in reading order, and measure
how much of the text each detector's layout delivers correctly - with the OCR engine held exactly
constant so that every difference is caused by layout alone.

## 2. What was done

| Step | Command | Output |
|---|---|---|
| OCR + region pipeline + scoring | `WORKERS=12 OMP_THREAD_LIMIT=1 python scripts/p5_ocr_downstream.py` | `RDLD_data/ocr/<config>/<page>.json` (cached words), `RDLD_data/results/p5/downstream/{downstream_pages.csv, downstream_configs.csv}` |
| Score cut-off check, τ = 0.3 and 0.7 (validation pages) | `TAU=0.3 WORKERS=12 python scripts/p5_ocr_downstream.py` | `RDLD_data/results/p5/downstream_tau0.3/`, `downstream_tau0.7/` |
| The same on the test pages | `RDLD_DATA=<test root> ...` | `RDLD_data_test/results/p5/downstream/` |

Logs: `RDLD_data/results/p5/logs/` and `RDLD_data_test/results/logs/p5_ocr_downstream.log`. The OCR cache is reused across
detectors and cut-offs; only the region step reruns.

**Detectors (11):** YOLO26-m (`yolo`), YOLO26-l, YOLOv11-m, YOLOv11-l, heron, heron101, egret, egretxl, DETR,
Deformable DETR and RF-DETR. The numbers for earlier detectors are the same as in the earlier runs.

**Page sets:** the 300 validation pages, and 300 pages from the official DocLayNet test split (P7).

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

Scope per page set: 300 pages × 37 conditions (clean + every severity-2 single and pair) = 11,100 OCR passes, × 11 detectors.

## 3. Results (evidence)

### 3.1 Clean condition (`p5/downstream/downstream_configs.csv`; 77,732 reference words on validation, 111,274 on test)

| Detector | val mAP | val WER | val recall | val precision | val table recall | test mAP | test WER | test recall | test precision | test table recall |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO26-m | 68.8 | **0.022** | 0.987 | 0.994 | 0.882 | 71.4 | **0.029** | 0.986 | 0.989 | **0.937** |
| YOLO26-l | 68.2 | 0.035 | 0.975 | 0.992 | 0.818 | 73.1 | 0.040 | 0.975 | 0.993 | 0.934 |
| YOLOv11-m | 68.9 | 0.063 | 0.966 | 0.978 | 0.851 | 71.3 | 0.060 | 0.968 | 0.982 | 0.929 |
| YOLOv11-l | 69.4 | 0.041 | 0.972 | 0.991 | 0.885 | 72.5 | 0.052 | 0.973 | 0.991 | 0.927 |
| heron | 64.2 | 0.112 | 0.954 | 0.938 | 0.763 | 65.7 | 0.187 | 0.988 | 0.862 | 0.905 |
| heron101 | 65.9 | 0.098 | 0.952 | 0.952 | 0.702 | 66.1 | 0.175 | 0.980 | 0.870 | 0.841 |
| egret | 61.5 | 0.102 | 0.955 | 0.946 | 0.715 | 64.4 | 0.171 | 0.987 | 0.874 | 0.895 |
| egretxl | 62.3 | 0.105 | 0.955 | 0.942 | 0.711 | 63.9 | 0.164 | 0.986 | 0.875 | 0.883 |
| DETR | 53.0 | 0.185 | 0.987 | 0.858 | **0.891** | 41.3 | 0.237 | 0.984 | 0.830 | 0.917 |
| Def. DETR | 66.2 | 0.058 | 0.948 | **0.999** | 0.779 | 70.4 | 0.072 | 0.942 | **0.997** | 0.898 |
| RF-DETR | 48.9 | 0.084 | 0.947 | 0.975 | 0.795 | 49.7 | 0.111 | 0.928 | 0.972 | 0.783 |

**YOLO26-m has the lowest clean WER on both page sets**, and all four YOLO models (0.022–0.063) are 1.6–6× lower than the four
Docling models (0.098–0.112 on validation, 0.164–0.187 on test), with at most 7 mAP points between the groups. Low text precision is the P1 hedging effect arriving
downstream: duplicate boxes on the same region make the pipeline read the same text twice. The detectors with the fewest
duplicate boxes (P4) have the highest text precision - Deformable DETR (precision 0.997–0.999) and the YOLO models - and
DETR and the Docling models, with the most duplicates, have the lowest. On the test pages the Docling models duplicate
more (P4 §3.3), so their precision falls to 0.86–0.88 and their WER rises by 0.06–0.08 while their recall stays at 0.98–0.99.
WER does not follow mAP: RF-DETR has a low mAP (48.9 and 49.7) but a lower WER than any Docling model on both page sets.

### 3.2 Severity-2 means (`downstream_configs.csv`)

| Detector | kind | val WER | val recall | val precision | val table recall | test WER | test recall | test precision | test table recall |
|---|---|---|---|---|---|---|---|---|---|
| YOLO26-m | single | 0.103 | 0.907 | 0.991 | 0.716 | 0.107 | 0.909 | 0.990 | 0.814 |
| YOLO26-m | pair | 0.234 | **0.775** | 0.990 | 0.594 | 0.236 | **0.779** | 0.990 | 0.625 |
| YOLO26-l | single | 0.131 | 0.877 | 0.993 | 0.729 | 0.123 | 0.890 | 0.993 | 0.789 |
| YOLO26-l | pair | 0.256 | **0.751** | 0.993 | 0.564 | 0.245 | **0.767** | 0.993 | 0.628 |
| YOLOv11-m | single | 0.197 | 0.823 | 0.982 | 0.670 | 0.190 | 0.833 | 0.985 | 0.707 |
| YOLOv11-m | pair | 0.364 | **0.651** | 0.979 | 0.492 | 0.366 | **0.653** | 0.986 | 0.488 |
| YOLOv11-l | single | 0.180 | 0.830 | 0.991 | 0.673 | 0.186 | 0.834 | 0.990 | 0.715 |
| YOLOv11-l | pair | 0.339 | **0.668** | 0.987 | 0.503 | 0.352 | **0.665** | 0.990 | 0.507 |
| heron | single | 0.144 | 0.954 | 0.911 | 0.674 | 0.194 | 0.975 | 0.862 | 0.760 |
| heron | pair | 0.198 | 0.938 | 0.880 | 0.515 | 0.238 | 0.943 | 0.848 | 0.564 |
| heron101 | single | 0.133 | 0.953 | 0.921 | 0.618 | 0.186 | 0.971 | 0.871 | 0.681 |
| heron101 | pair | 0.181 | 0.935 | 0.893 | 0.477 | 0.230 | 0.934 | 0.859 | 0.485 |
| egret | single | 0.155 | 0.945 | 0.910 | 0.597 | 0.200 | 0.969 | 0.863 | 0.751 |
| egret | pair | 0.229 | 0.895 | 0.882 | 0.441 | 0.273 | 0.898 | 0.850 | 0.531 |
| egretxl | single | 0.153 | 0.941 | 0.912 | 0.615 | 0.207 | 0.966 | 0.855 | 0.723 |
| egretxl | pair | 0.220 | 0.902 | 0.883 | 0.467 | 0.291 | 0.884 | 0.844 | 0.502 |
| DETR | single | 0.262 | 0.951 | 0.827 | 0.781 | 0.332 | 0.954 | 0.788 | 0.828 |
| DETR | pair | 0.367 | 0.876 | 0.793 | 0.635 | 0.421 | 0.898 | 0.756 | 0.690 |
| Def. DETR | single | 0.257 | 0.748 | 0.997 | 0.535 | 0.248 | 0.764 | 0.997 | 0.591 |
| Def. DETR | pair | 0.445 | **0.560** | 0.995 | 0.345 | 0.423 | **0.586** | 0.996 | 0.378 |
| RF-DETR | single | 0.151 | 0.882 | 0.968 | 0.703 | 0.183 | 0.861 | 0.966 | 0.670 |
| RF-DETR | pair | 0.232 | **0.803** | 0.959 | 0.585 | 0.253 | **0.796** | 0.957 | 0.555 |

The P4 prediction holds for all eleven detectors on both page sets. **The detectors that fail by *missing* regions lose text
recall** on pairs: the YOLO models 0.65–0.78, Deformable DETR 0.56–0.59, RF-DETR 0.80. **The four Docling models, which fail
by *merging*, keep recall** (≥ 0.88) and lose order and precision instead. DETR, which does both, loses some of each and has
the lowest precision. YOLOv11 loses about 10 points more recall on pairs than YOLO26 and has a WER of 0.34–0.37, close to DETR
and Deformable DETR. On compound degradation YOLO26, RF-DETR and the Docling models end with similar WER (0.18–0.29) for different
reasons. Deformable DETR has one of the lowest WERs on clean pages and the highest on pairs.

### 3.3 Score cut-off (validation pages, severity-2 pairs)

| Detector | WER τ 0.3 / 0.5 / 0.7 | text recall τ 0.3 / 0.5 / 0.7 | text precision τ 0.3 / 0.5 / 0.7 |
|---|---|---|---|
| YOLO26-m | 0.192 / 0.234 / 0.324 | 0.848 / 0.775 / 0.680 | 0.957 / 0.990 / 0.995 |
| YOLO26-l | 0.221 / 0.256 / 0.331 | 0.808 / 0.751 / 0.673 | 0.969 / 0.993 / 0.997 |
| YOLOv11-m | 0.301 / 0.364 / 0.476 | 0.751 / 0.651 / 0.529 | 0.938 / 0.979 / 0.994 |
| YOLOv11-l | 0.276 / 0.339 / 0.436 | 0.756 / 0.668 / 0.567 | 0.961 / 0.987 / 0.995 |
| heron | **0.698** / 0.198 / 0.178 | 0.963 / 0.938 / 0.831 | 0.638 / 0.880 / 0.992 |
| heron101 | **0.622** / 0.181 / 0.178 | 0.963 / 0.935 / 0.831 | 0.652 / 0.893 / 0.992 |
| egret | **0.667** / 0.229 / 0.235 | 0.949 / 0.895 / 0.774 | 0.621 / 0.882 / 0.992 |
| egretxl | **0.667** / 0.220 / 0.218 | 0.951 / 0.902 / 0.789 | 0.632 / 0.883 / 0.993 |
| DETR | **0.664** / 0.367 / 0.314 | 0.931 / 0.876 / 0.774 | 0.643 / 0.793 / 0.899 |
| Def. DETR | 0.329 / 0.445 / 0.600 | 0.748 / 0.560 / 0.403 | 0.906 / 0.995 / 0.998 |
| RF-DETR | 0.325 / 0.232 / 0.366 | 0.961 / 0.803 / 0.643 | 0.789 / 0.959 / 0.987 |

- The detectors that miss gain recall and lower WER at 0.3 and lose both at 0.7; their precision barely moves.
- The Docling models and DETR read duplicated text at 0.3: precision falls to 0.62–0.65 and the Docling models' WER more than triples. At 0.7
  the Docling models' duplicates disappear and their WER is the lowest of all detectors (0.18–0.24), but recall drops below 0.84.
- So the best cut-off differs by detector family, and the merge/miss symptom split holds in relative terms at every cut-off:
  at each τ the Docling models have higher recall and lower precision than the YOLO models and Deformable DETR.

### 3.4 OCR coverage
Pages where Tesseract returns no words at all (validation): clean 7 / 300 (white text on photographs, picture-only pages),
texture@2 29 / 300, speckle@2 6 / 300. Pages with no reference words (no OCR word inside any GT region): validation clean 8,
texture@2 33, speckle@2 7; test clean 3, texture@2 45, speckle@2 4; the most is texture@2 + vibration@2 (71 validation,
76 test). These pages carry zero weight in the word-weighted aggregates; both pipelines see the same (empty) word set, so
they cannot bias the comparison.

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
5. **No deduplication of overlapping regions** in the pipeline. A real pipeline that takes boxes as-is reads duplicates
   twice; this is the effect being measured. §3.3 shows how strongly it depends on the cut-off.

## 5. Problems found
- **First OCR run silently produced no text** for all 11,100 pages. Cause: Tesseract could not find its
  language data (`TESSDATA_PREFIX` is set by conda activation, not when the env's python is called
  directly), and the `tsv` config file is absent in the conda-forge package; the script swallowed the error.
  Fixed: set `TESSDATA_PREFIX` from the env path, use `-c tessedit_create_tsv=1`, and **fail loudly** on a
  non-zero exit or an empty first page. Empty cache deleted and re-run.
- **Thread oversubscription.** 12 Tesseract processes each spawning OpenMP threads gave ≈ 0.5 pages/s;
  `OMP_THREAD_LIMIT=1` gave ≈ 3.6 pages/s (7×).
- Tesseract cannot read light text on photographs; affected pages are excluded by weighting (§3.4).

## 6. Main takeaway
Downstream outcome for RQ2. The failure *type* found in P4 predicts the downstream *symptom* for all eleven detectors on
both page sets: missed regions → lost text (recall on pairs 0.56–0.80 for the YOLO models, Deformable DETR and RF-DETR);
merged and duplicated regions → wrong order and duplicated text (Docling models' precision 0.84–0.89). Similar WERs can hide
different failures, the score cut-off moves the families in opposite directions, and neither clean mAP nor clean WER predicts
which detector holds up downstream under compound degradation.

## 7. Hand-off
- P5 (Khizar): `downstream_configs.csv` → correlation of each layout metric with WER / recall / precision /
  table recall; ranking agreement; threshold fit. The same for the 0.3 and 0.7 folders.

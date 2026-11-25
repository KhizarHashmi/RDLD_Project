# P8 - Review Checks - Alishba (Workstream A)

**Date:** 2026-11-26
**Status:** code ready. The runs are on the shared laptop and the results are added to these notes when they finish.

## 1. Why
The review of the paper raised two points that sit in Workstream A.

1. **OCR and layout are mixed in RQ2.** On a degraded page Tesseract makes more mistakes, so part of the change in
   word error rate comes from the OCR, not from the layout. We want the share that comes from the layout alone.
2. **Pairs are only tested at the same severity.** A light degradation can meet a strong one on a real page, and
   the main design never builds that case.

## 2. What was added

### 2.1 Clean-text reading (`scripts/p5_ocr_downstream.py`, new option `TEXT=clean`)
The ground-truth regions and the predicted regions are read from the words of the clean page instead of the words
of the degraded page. None of our eight degradations moves anything on the page, so every clean word sits where the
degraded word would be. The OCR errors caused by the degradation then drop out, and whatever change in word error
rate is left comes from the layout alone.

- Same pages, same 37 conditions at severity 2, same score cut-off 0.5, same reading order and region rules.
- Results go to `results/p5/downstream_cleantext/`. The main results in `results/p5/downstream/` are not touched.
- Without the option the script behaves exactly as before.

```
TEXT=clean python scripts/p5_ocr_downstream.py
RDLD_DATA=../RDLD_data_test TEXT=clean python scripts/p5_ocr_downstream.py
```

### 2.2 Mixed-severity pairs (`scripts/p7_cross_severity.py`, new)
Five pairs at every mixed severity, (1,2) (2,1) (1,3) (3,1) (2,3) (3,2), which gives 30 conditions.

- **How the pairs were chosen:** the five with the largest mean multiplicative interaction over the eleven detectors
  at severity 2 on the validation pages: speckle + vibration, texture + vibration, speckle + defocus,
  ink holdout + vibration, ink bleeding + speckle.
- **Where they are tested:** on the test pages, so the pages that picked the pairs are not the pages that test them.
- **How each image is built:** degradation A is read from the cached single A at its severity, then B is applied
  with the same seed as the single B at its severity. The common random numbers hold, so the interaction can be read
  against the two cached singles, exactly as in the main experiment.
- `--check` compares the cached route with applying both steps to the clean page. On 3 pages and 3 conditions all
  9 images were identical.

```
RDLD_DATA=../RDLD_data_test python scripts/p7_cross_severity.py --check
RDLD_DATA=../RDLD_data_test python scripts/p7_cross_severity.py
```

Writes `degraded/configs_xsev.csv` and one folder of 300 pages per condition. Detection and the bootstrap tables are
in Workstream B, as for the order and seed checks.

### 2.3 Second seed on the test pages
`scripts/p4_seed_replicate.py` runs unchanged with `RDLD_DATA=../RDLD_data_test`, so the seed check is no longer
limited to the validation pages. The reversed-order cache (`scripts/p3_order_check.py`) is also rebuilt on the test
pages, unchanged.

## 3. Results
Added after the runs finish.

# P8 - Review Checks - Alishba (Workstream A)

**Date:** 2026-11-26, updated 2026-11-27
**Status:** clean-text reading, mixed-severity pairs and the test-page replicates done (Section 3). Geometric pilot
code ready; its results are added when the runs finish.

## 1. Why
The review of the paper raised three points that sit in Workstream A.

1. **OCR and layout are mixed in RQ2.** On a degraded page Tesseract makes more mistakes, so part of the change in
   word error rate comes from the OCR, not from the layout. We want the share that comes from the layout alone.
2. **Pairs are only tested at the same severity.** A light degradation can meet a strong one on a real page, and
   the main design never builds that case.
3. **Geometric degradations are left out.** Rotation and keystoning move the regions, and the paper only argued why
   they were excluded. A pilot gives a number instead.

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

### 2.4 Geometric pilot (`scripts/p8_geometric_pairs.py`, new)
The paper left geometric degradations out because they move the regions. The review asked for a number instead of
an argument, so this pilot builds them on the test pages at severity 2.

- **Conditions:** rotation@2 and keystoning@2 as singles, and each of them paired with every one of the eight image
  degradations: 2 singles and 16 pairs, 18 conditions.
- **Transforms:** both follow RoDLA's `spatial.py` with the same parameters. RoDLA's code needs detectron2 and imgaug,
  which do not run in our environment, so the two transforms are written with OpenCV. Rotation turns the page by 5 to
  10 degrees either way and grows the canvas so nothing is cut off. Keystoning pulls the four corners in by a random
  amount and maps the page back to a rectangle of the same size. Both fill the uncovered area with black, as RoDLA does.
- **Boxes:** every box is moved with its page. Its four corners go through the same matrix as the image and the new box
  is the smallest upright box around them, as in RoDLA. Each condition folder carries its own `annotations.json`, and
  `transforms.json` keeps the matrix of every page.
- **Order:** paper, printing and overlays come before the geometry; lighting and blur come after it. So
  `texture@2+rotation@2` turns the cached texture page, and `rotation@2+defocus@2` blurs the cached rotated page.
- **Common random numbers:** the geometric draw uses the same seed in the single and in the pair, so a pair has
  exactly the same boxes as its geometric single.
- `--check` on 3 pages: the cached route and the route from the clean page gave identical images for every pair it
  tried, the draw repeats exactly, all 4,351 moved boxes have a positive area, and the pictures with the boxes drawn on
  them show the boxes sitting on the moved regions.

```
RDLD_DATA=../RDLD_data_test python scripts/p8_geometric_pairs.py --check
RDLD_DATA=../RDLD_data_test python scripts/p8_geometric_pairs.py
```

Writes `degraded/configs_geom.csv` and one folder per condition. Detection, the bootstrap tables against each
condition's own boxes, and the interaction are in Workstream B.

## 3. Results

### 3.1 Clean-text reading (both page sets, 407 points, 37 conditions)
Within-condition Spearman correlations, degraded reading vs clean-text reading:

| | Validation, degraded | Validation, clean text | Test, degraded | Test, clean text |
|---|---|---|---|---|
| mAP vs WER | -0.731 | -0.731 | -0.730 | -0.744 |
| duplicate share vs text precision | -0.909 | -0.922 | -0.894 | -0.904 |

The detector order by WER agrees between the two readings at a median Kendall tau of 0.93 (validation) and 0.96
(test). The RQ2 answer does not come from OCR noise.

### 3.2 Mixed-severity pairs (test pages, 308 noncontrol cells)
235 cells are super-multiplicative and 17 sub-multiplicative; 86 are super-additive under both nulls; the median
I_mult is positive for all eleven detectors. The mean interaction, 5.0 points, lies between that of the same pairs at
the lower and the higher severity (1.8 and 7.7). The 22 cells with defocus at severity 1 are exact zeros and are left
out as controls.

### 3.3 Second seed and reversed order on the test pages
The second seed moves the interaction by 0.21 to 0.58 times the page-bootstrap noise (0.28 to 0.41 on the validation
pages), and no significant call changes sign. In reversed order, 170 of 308 cells are significant on the test pages
(164 on the validation pages), with a median of 0.57 (0.52).

### 3.4 Geometric pilot
Added after the runs finish.

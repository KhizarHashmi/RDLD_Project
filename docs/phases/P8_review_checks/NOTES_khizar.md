# P8 - Review Checks - Khizar (Workstream B)

**Dates:** 2026-11-26 – 2026-11-27
**Status:** done. Detector exclusion sets, clean-text reading comparison, mixed-severity interaction, seed and order
replicates on the test pages, and the geometric pilot interaction. Every result is in the paper (v5 and v6).

## 1. Goal
The review of the paper asked for evidence on five points that sit in Workstream B.

1. **Do the conclusions rest on detectors we could not control?** RF-DETR was fine-tuned by us, DETR may have seen
   validation-split pages, and the two YOLO generations were trained for different lengths.
2. **Is the RQ2 answer an OCR effect?** On a degraded page Tesseract itself makes more mistakes.
3. **Do pairs compound only at matched severities?**
4. **Are the seed and order checks limited to the validation pages?**
5. **What happens with geometric degradations?** The paper only argued why they were left out.

## 2. What was done

| Point | Script | Run | Output |
|---|---|---|---|
| 1 | `scripts/p8_detector_exclusion.py` | `python scripts/p8_detector_exclusion.py`, then with `RDLD_DATA=../RDLD_data_test` | `results/p8/detector_exclusion/exclusion_summary.csv` |
| 2 | `scripts/p8_cleantext_reading.py` (needs Alishba's `TEXT=clean` reading first) | as above, both page sets | `results/p8/cleantext_reading/{within_condition,agreement,summary_by_kind}.csv` |
| 3 | `scripts/p8_cross_severity_interaction.py` (after inference on `configs_xsev.csv`) | `RDLD_DATA=../RDLD_data_test` | `RDLD_data_test/results/p8/cross_severity/xsev_{interactions,summary,vs_same}.csv` |
| 4 | `scripts/p4_seed_replicate.py`, `scripts/p3_order_check.py`, `scripts/p4_rq3_robustness.py`, unchanged | `RDLD_DATA=../RDLD_data_test` | `RDLD_data_test/results/p4/...`, `RDLD_data_test/results/p8/seed_noise_floor/` |
| 5 | `scripts/p8_geometric_interaction.py` (after Alishba's `p8_geometric_pairs.py` and inference) | see 2.1 | `RDLD_data_test/results/p8/geometric/` |

### 2.1 Geometric pilot run
```
RDLD_DATA=../RDLD_data_test WORKERS=12 python scripts/p8_geometric_pairs.py          # Alishba's generator, 1 min
CONFIGS=../RDLD_data_test/degraded/configs_geom.csv RDLD_DATA=../RDLD_data_test python scripts/p2_run_inference.py
RDLD_DATA=../RDLD_data_test WORKERS=10 python scripts/p8_geometric_interaction.py
```
18 conditions × 300 pages × 11 detectors. Inference took 2 h 31 min (the GPU throttled at 88 °C), the bootstrap and
interaction 14 min. Logs: `RDLD_data_test/results/p8/logs/geom*.log`.

The interaction script scores every geometric condition against its own moved ground truth, with the same bootstrap
resample index as all other tables. It also scores an **equivariant reference**: the clean predictions moved by the
same matrix as the page. This is what a detector that followed the page perfectly would score.

## 3. Results (evidence)

### 3.1 Detector exclusion (`results/p8/detector_exclusion/exclusion_summary.csv`, both page sets)
Fifteen sets: all 11, without RF-DETR, without DETR, without both, one YOLO generation only, and each other detector
left out in turn.

| | Validation | Test |
|---|---|---|
| Super- to sub-multiplicative ratio | 2.06 to 2.36 | 2.09 to 2.44 |
| Detectors with positive median I_mult | all, in every set | all, in every set |
| τ gap | 0.216 to 0.305 | 0.207 to 0.301 |
| τ gap page interval excludes zero | every set | every set |
| τ, mRD vs single retention | −0.28 to 0.11 | −0.33 to 0.00 |

Leaving out DETR raises the τ gap (0.294 and 0.301), so the unknown DETR training split does not create the effect.

### 3.2 Clean-text reading (`results/p8/cleantext_reading/`)

| Within-condition Spearman (407 points) | Validation, degraded text | Validation, clean text | Test, degraded | Test, clean |
|---|---|---|---|---|
| mAP vs WER | −0.731 | −0.731 | −0.730 | −0.744 |
| Duplicate share vs text precision | −0.909 | −0.922 | −0.894 | −0.904 |

Median τ between the detector order by WER under the two readings: 0.93 (validation) and 0.96 (test). The RQ2
answer is a layout effect, not OCR noise.

### 3.3 Mixed severities (`RDLD_data_test/results/p8/cross_severity/xsev_summary.csv`)
308 noncontrol cells (11 detectors × 28 mixed pairs; the two with defocus at 1 are controls): 235 super- and 17
sub-multiplicative, 86 super-additive under both nulls, median I_mult positive for 11 of 11. The mean I_mult, 5.01,
lies between the same pairs at the lower (1.78) and the higher (7.67) equal severity. The pairs were chosen for strong
interaction, so the counts are not comparable with the main table.

### 3.4 Seed and order on the test pages
- Seed: the interaction moves by 0.21 to 0.58 times the page-bootstrap noise (validation 0.28 to 0.41), no significant
  sign changes. Detector order under a second seed: τ 0.97 singles and 0.94 pairs
  (`RDLD_data_test/results/p8/seed_noise_floor/seed_noise_floor.csv`).
- Order: reversing the composition order changes mAP significantly in 170 of 308 tests (validation 164) but by a
  median of 0.57 points (0.52); texture is the exception (median 4.0 and 3.2).

### 3.5 Geometric pilot (`RDLD_data_test/results/p8/geometric/`)

Singles (`geom_singles.csv`):

| | mAP drop D | Equivariant reference vs clean |
|---|---|---|
| rotation@2 | 33.2 to 66.9 points | +6.4 to +12.1 points |
| keystoning@2 | 6.3 to 27.8 points | +2.2 to +3.9 points |

The equivariant reference scores **above** clean. The upright box around a turned region is larger than the region,
so IoU is easier to pass (Heron: clean 65.7, equivariant 76.5 under rotation). The real loss of the detector is
therefore larger than D shows.

Interactions (`geom_summary.csv`, page bootstrap, FDR 0.05 within each detector):

| Set | Cells | Super-mult. | Sub-mult. | Median I_mult | Detectors with median > 0 |
|---|---|---|---|---|---|
| All geometric pairs | 176 | 53 | 16 | 0.57 | 8/11 |
| With rotation | 88 | 41 | 3 | 1.26 | 10/11 |
| With keystoning | 88 | 12 | 13 | −0.17 | 4/11 |
| Image pairs at severity 2 (reference) | 308 | 117 | 51 | 1.08 | 10/11 |

The frame-corrected interaction I_frame (retention measured against the equivariant reference) gives the same class
as I_mult in all 176 cells (53 super, 16 sub, 107 n.s.). The strongest geometric pair is rotation + vibration (mean
I_mult 5.12, 10 super cells), and rotation + texture (3.16) and rotation + defocus (2.22) follow (`geom_by_degradation.csv`).

## 4. Decisions and why
- **Geometric pilot on the test pages at severity 2 only.** It answers the review with a number inside the GPU budget
  (about 2.5 h). Warping was left for later work.
- **Equivariant reference instead of trusting D.** Without it, a reviewer could say the geometric interaction comes
  from the looser boxes. I_frame removes that effect and changes no class.
- **Exclusion sets cover each detector in turn,** not just the three with known confounds, so no single model carries
  a conclusion.
- **Mixed-severity pairs tested on the test pages,** so the pages that chose the pairs do not test them.

## 5. Problems found
- The laptop GPU throttled to about 300 MHz at 88 °C, so the geometric inference took 2.5 h instead of about 1 h.
- The bootstrap tables of the new conditions have no jackknife rows, so they are sliced to the first 1,001 rows when
  read (`1 + n_boot`).
- Absolute mAP under rotation is not comparable with the image degradations, because the boxes change shape. Only
  the interaction, which compares like with like, is reported.

## 6. Main takeaway
None of the review points changes an answer. The results hold without any single detector, are a layout effect and
not OCR noise, hold at mixed severities, and repeat on the test pages. Geometric damage compounds for rotation in the
same way as image damage, and does not compound for keystoning.

## 7. Hand-off
- Paper: geometric paragraph, Method sentence, Limitations line and image count (118,200) are in v5 and v6.
- Alishba: her notes Section 3.4 can take the numbers in 3.5 above.
- Next: merge `khizar` and `alishba` into `main` once both notes are final.

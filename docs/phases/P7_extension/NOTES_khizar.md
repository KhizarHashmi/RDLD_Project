# P7 - Detector Extension - Khizar (Workstream B)

**Dates:** 2026-11-04 – 2026-11-07
**Status:** done. All 8 detectors went through inference, bootstrap and every analysis. Some methodology checks remain open (Section 7).

## 1. Goal
Widen the comparison from 3 to 8 detectors, with every added detector going through the same pipeline
as the original three.

## 2. What was done

| Key | Model | Source |
|---|---|---|
| `heron101` | RT-DETR v2, ResNet-101 | `docling-project/docling-layout-heron-101` |
| `egretxl` | D-FINE x-large | `docling-project/docling-layout-egret-xlarge` |
| `detr` | DETR, ResNet-50 | `cmarkea/detr-layout-detection` |
| `ddetr` | Deformable DETR | `Aryn/deformable-detr-DocLayNet` |
| `rfdetr` | RF-DETR medium, DINOv2 backbone | fine-tuned by us from `Roboflow/rf-detr-medium` |

Code: `rdld/detectors.py` (wrappers, label mapping, colours), `scripts/p2_run_inference.py` (per-detector batch size),
`scripts/p7_finetune_rfdetr.py` (training). Analysis scripts were changed to handle any number of detectors.

New scripts:

| Script | Purpose | Output in `RDLD_data/results/` |
|---|---|---|
| `scripts/p4_rq3_robustness.py` | Is the ranking change caused by pairing or by the amount of damage? | `p4/rq3_robustness/` |
| `scripts/p7_select_vs_test.py` | All detectors on the 600 selection pages and the 300 test pages | `p7/select_vs_test/` |
| `scripts/p7_title_sensitivity.py` | Headline numbers with and without the Title class | `p7/title_sensitivity/` |

## 3. Results

Clean test pages (300). `p1/baseline/summary.csv`, `p1/baseline_ext/summary.csv`, `p7/select_vs_test/select_vs_test.csv`:

| Detector | mAP | mAP@.5 |
|---|---|---|
| `yolo` | 68.8 | 87.6 |
| `ddetr` | 66.2 | 87.6 |
| `heron101` | 65.9 | 88.3 |
| `heron` | 64.2 | 89.1 |
| `egretxl` | 62.3 | 87.1 |
| `egret` | 61.5 | 86.1 |
| `detr` | 53.0 | 77.9 |
| `rfdetr` | 48.9 | 69.8 |

RF-DETR on the 600 selection pages (`RDLD_data/finetune/rfdetr/train_log.csv`). 11,526 training pages, about one sixth
of the DocLayNet training split, 12.4 training hours.

| Epoch | Train loss | Select mAP | Select mAP@.5 |
|---|---|---|---|
| 1 | 15.89 | 32.4 | 49.1 |
| 2 | 6.54 | 43.6 | 60.9 |
| 3 | 5.54 | 46.5 | 64.4 |
| 4 | 5.06 | 48.1 | 66.0 |
| 5 | 4.73 | 52.1 | 70.9 |
| 6 | 4.44 | 53.9 | 72.5 |
| 7 | 4.21 | 54.3 | 73.4 |
| 8 | 4.01 | 55.9 | 74.8 |
| 9 | 3.84 | 55.7 | 74.4 |
| 10 | 3.72 | 57.2 | 75.8 |
| 11 (best) | 3.65 | 57.4 | 76.2 |
| 12 | 3.62 | 57.3 | 75.7 |

Through the inference wrapper the best checkpoint scores 57.55 on the selection pages, against 57.38 during training,
so the wrapper is correct and the low clean score is real.

### 3.1 Original three unchanged
`rq1_interactions.csv`, `single_degradation_drops.csv`, `metrics_long.csv`, `uw_map.csv` and `structural_rates.csv`
match `results/_backup_3det` row by row for YOLO, heron and egret. Adding detectors changed nothing for them.

### 3.2 Selection pages vs test pages (`p7/select_vs_test/`)

| Detector | Select mAP | Test mAP | Gap [95% CI] | Gap without Title |
|---|---|---|---|---|
| `egretxl` | 60.02 | 62.27 | -2.25 [-7.46, 2.54] | +1.32 |
| `egret` | 61.07 | 61.52 | -0.45 [-5.65, 4.27] | +1.45 |
| `heron101` | 63.67 | 65.87 | -2.20 [-7.44, 2.48] | +1.64 |
| `heron` | 63.17 | 64.23 | -1.05 [-6.15, 3.49] | +3.38 |
| `detr` | 56.20 | 53.01 | +3.19 [-2.14, 8.50] | +6.09 |
| `ddetr` | 71.23 | 66.20 | +5.04 [0.22, 9.61] | +7.07 |
| `yolo` | 74.47 | 68.82 | +5.65 [0.42, 11.54] | +7.45 |
| `rfdetr` | 57.55 | 48.88 | +8.67 [3.12, 13.16] | +8.51 |

Without Title, every detector scores lower on the test pages, so those pages are harder. No detector does unusually
well on them, which is what overlap with its training data would look like. This is a screening check and not proof.
The two groups fit the much broader training data of the heron and egret families, but we did not test that explanation.

### 3.3 Title class (`p7/title_sensitivity/`)
The test pages have 6 Title boxes (37 on the selection pages), and Title counts for 1/11 of mAP. Heron's Title AP goes
from 15.3 on the selection pages to 60.7 on the test pages, which is noise from so few boxes.

Without Title: median interactions stay similar; significant pairs rise for every detector and null (YOLO super-multiplicative
15 to 28); three swaps between neighbours in the clean order; one swap in the retention order on pairs; tau 0.618 on singles
and 0.452 on pairs (0.671 and 0.464 with Title, controls excluded). The findings hold. Report both in the paper.

## 4. Decisions and why
1. **Clean training only, no flips.** The study measures robustness a detector did not train for, and
   mirrored text is not a real document.
2. **DETR uses its detection sub-model only.** The checkpoint's segmentation head needed about 25 GB of VRAM.
3. **Batch size 1 for DETR and Deformable DETR.** Batching was slower for these two.
4. **Training can resume.** Weights, optimiser, schedule and random state are saved after every epoch, and
   each epoch has its own seeded data order.
5. **Tau is reported with controls excluded.** `defocus@1` is a pixel no-op, so those configurations copy other ones.
   This gives 0.67 and 0.46. The main log shows 0.69 and 0.49 with controls included.
6. **Weight sensitivity uses the ratio of mean drops.** See Section 5.

## 5. Problems found
- The data loader could not pickle a local collate function under Windows spawn. Fixed with a top-level
  class before the real run. A stop-and-resume test on a 32-page subset then passed.
- The laptop GPU reached 95 °C and throttled, which slowed the Deformable DETR runs.
- Analysis scripts assumed 3 detectors (3×3 heatmap grid, pairs cut to 3, fixed colours). Fixed and dry-run on 7 detectors.
- DETR's config looks up `resnet50` on the Hub, so it cannot load under `HF_HUB_OFFLINE=1`. Two jobs crashed on this and were rerun online.
- `p5_rq2_metric_validity.py` weight sensitivity gave -5,158,634. It averaged per-configuration ratios, and Deformable DETR scores
  above clean on `defocus@3` and `illumination@3+defocus@3`, so the denominator was clamped to 1e-9. Now the ratio of mean drops:
  1.06 [1.02, 1.10]. Under the new estimator the original three give 1.08. The 1.09 from the old estimator is not comparable.
- `fig_interaction_heatmaps.png` has an empty band above the first row with 8 detectors. Values are correct. To fix for the paper.

## 6. Limitations this adds
- RF-DETR had about one sixth of the training pages, so it is not like for like with the released detectors.
- Its selection pages are from the same split as the test pages. There is no page or document overlap with its training data, but selection
  pages may share documents with test pages. This only affects which epoch was picked.
- The 300 test pages come from the DocLayNet validation split. The official test split is not downloaded. The Docling report
  (arXiv:2509.11720) says heron and egret trained on a corpus with canonical DocLayNet plus other data, and evaluated on test splits,
  so our pages probably were not training data but may have been used to pick checkpoints. The DETR, Deformable DETR and YOLO cards name no split.

## 7. Open methodology checks
1. Results with and without RF-DETR.
2. Repeat the main results on 300 pages from the official DocLayNet test split (decided 2026-11-07). About 14–18 hours with the added detectors.
3. BCa bootstrap intervals (decided 2026-11-07). Some percentile intervals are lopsided around the estimate (tau gap 0.207 [0.131, 0.243]).
4. Sensitivity of the structural and OCR analyses to the fixed 0.5 score threshold.
5. Damage as a confound in RQ1 and RQ2, as tested for RQ3.
6. Deformable DETR clean mAP is 66.20 from the stored predictions and 66.17 in the baseline rerun. Small, but find out why.
7. Reproducibility: pinned versions, one-command rerun, per-condition results, and release of the RF-DETR weights after a licence check.
8. Add YOLO26 large, YOLOv11 medium and YOLOv11 large from `hantian/yolo-doclaynet` (decided 2026-11-07). With the current YOLO26 medium this
   gives size × generation. Check the author's training settings per version first; the weights are AGPL-3.0.

Order: add the YOLO models, then the official test split, then items 1, 3, 4 and 5 once the detectors and pages are final.

# P7 - Detector Extension - Khizar (Workstream B)

**Dates:** 2026-11-04 – in progress
**Status:** four added detectors finished through inference and bootstrap; RF-DETR training paused at epoch 6 of 12.

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

Code: `rdld/detectors.py` (wrappers, label mapping), `scripts/p2_run_inference.py` (per-detector batch size),
`scripts/p7_finetune_rfdetr.py` (training).

## 3. Results so far

Clean test pages (300), from `RDLD_data/predictions/<model>/clean.json`:

| Detector | mAP | mAP@.5 |
|---|---|---|
| `ddetr` | 66.2 | 87.6 |
| `heron101` | 65.9 | 88.3 |
| `egretxl` | 62.3 | 87.1 |
| `detr` | 53.0 | 77.9 |

Each of the four has 173 prediction sets (main, order and seed replicates) and 173 bootstrap tables.

RF-DETR on the 600 selection pages (`RDLD_data/finetune/rfdetr/train_log.csv`). These numbers are not
comparable with the table above because the pages differ.

| Epoch | Train loss | Select mAP | Select mAP@.5 |
|---|---|---|---|
| 1 | 15.89 | 32.4 | 49.1 |
| 2 | 6.54 | 43.6 | 60.9 |
| 3 | 5.54 | 46.5 | 64.4 |
| 4 | 5.06 | 48.1 | 66.0 |
| 5 | 4.73 | 52.1 | 70.9 |
| 6 | 4.44 | 53.9 | 72.5 |

## 4. Decisions and why
1. **Clean training only, no flips.** The study measures robustness a detector did not train for, and
   mirrored text is not a real document.
2. **DETR uses its detection sub-model only.** The checkpoint's segmentation head needed about 25 GB of VRAM.
3. **Batch size 1 for DETR and Deformable DETR.** Batching was slower for these two.
4. **Training can resume.** Weights, optimiser, schedule and random state are saved after every epoch, and
   each epoch has its own seeded data order.

## 5. Problems found
- The data loader could not pickle a local collate function under Windows spawn. Fixed with a top-level
  class before the real run. A stop-and-resume test on a 32-page subset then passed.
- The laptop GPU reached 95 °C and throttled, which slowed the Deformable DETR runs.

## 6. Remaining
Finish RF-DETR training, run its inference and bootstrap, then rerun all analyses for 8 detectors.

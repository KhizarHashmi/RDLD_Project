# P0 - Feasibility & Detector Selection - Khizar (Workstream B)

**Dates:** 2026-09-06 – 2026-09-27 (re-verified 2026-11-03 in the RDLD_Project layout)

## 1. Goal
Establish that an independent, released layout detector runs on the project machine without RoDLA's
deprecated stack (torch 1.10 + mmdetection + compiled DCNv3), select the detector set for RQ3, and
confirm the research gaps are still open.

## 2. What was done

| Script | Command | Output |
|---|---|---|
| `scripts/p0_detector_gate.py` | `python scripts/p0_detector_gate.py` | `RDLD_data/results/p0/detections/`, log `results/p0/logs/p0_detector_gate.log` |
| `scripts/p0_make_figures.py` | `python scripts/p0_make_figures.py` | `RDLD_data/results/p0/figures/` (proposal / defence figures from real data) |

Competitor check (Sept 12): literature search for compound/composite degradation in layout analysis and
the RoDLA authors' 2025–26 output - recorded in our local gate results.

## 3. Results (evidence)

**Gate C - detector runs** (`p0_detector_gate.log`, RTX 3080 Laptop 16 GB)

| Item | Result |
|---|---|
| Model | `ds4sd/docling-layout-heron` (RT-DETR v2) |
| Dependencies | `torch` + `transformers` only - no mmdetection / mmcv / CUDA compilation |
| VRAM peak | **0.30 GB** |
| Latency | 594 ms/page steady state (Sept); 1,649 ms in the 2026-11-03 re-run, which is a single cold call incl. CUDA warm-up - real throughput is measured in P1 |
| Output | 9 regions @ conf > 0.3; classes `text`, `section_header`, `picture`; boxes land on real regions |

**Detector set selected** (verified loadable Sept 15; our metric plan)

| ID | Model | Family | Label space |
|---|---|---|---|
| M1 | `hantian/yolo-doclaynet` (yolo26m) | CNN one-stage (YOLO) | 11 - identical to DocLayNet |
| M2 | `ds4sd/docling-layout-heron` | RT-DETR v2 transformer | 12+ - needs mapping |
| M3 | `ds4sd/docling-layout-egret-medium` | D-FINE transformer | 17 - needs mapping |

**Gap check:** no compound-degradation study for layout analysis found; RoDLA's released code applies
one perturbation per dataset (`for key in key_list: for degree in degree_list`). Closest work (SpaceDG,
2605.22536) is indoor spatial reasoning, single degradations. mAP inadequacy is asserted in the
literature (DLER; LED benchmark) but never quantified against downstream quality.

## 4. Decisions and why
1. **Three detectors from three families** (CNN, RT-DETR, D-FINE): a ranking change inside one family
   cannot separate architecture sensitivity from one model's quirks - required for RQ3.
2. **Do not run RoDLA's own model or HybriDLA (83.5% SOTA).** Both need InternImage + mmdetection +
   compiled DCNv3 on torch 1.10. Cited as motivation; listed as a limitation / future work.
3. **Inference only, no training** - ~7 GPU-hours total at proposal-time estimates.

## 5. Problems found
- **Label instability on clean input:** heron emitted the *same box* twice with different labels
  (`section_header` 0.529 and `text` 0.493 at `[86,88,430,158]`). If uncharacterised, this noise would be
  wrongly attributed to degradation. → P1 must measure the clean error floor and fix a dedup policy.
- **Label spaces differ** (11 / 12+ / 17). Unmapped classes would count as false positives and corrupt
  the ranking comparison. → P1 builds an explicit mapping to the 11 DocLayNet classes.

## 6. Main takeaway
Feasibility pre-test on the target hardware de-risks the method; the detector set is the
minimum that lets RQ3 attribute ranking changes to architecture.

## 7. Hand-off
- To P1 (Khizar): wrappers for M1–M3 behind one interface; class mapping; clean mAP; instability floor.

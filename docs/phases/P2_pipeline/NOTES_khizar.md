# P2 - Pipeline - Khizar (Workstream B)

**Dates:** 2026-11-03 – 2026-11-04

## 1. Goal
Turn Workstream A's degraded-image cache into scored detections: run the three detectors over every
cached configuration with the exact clean-baseline settings, and score each prediction set with one
evaluator that also yields the drop from clean and RoDLA's mRD.

## 2. What was done

| Item | File / command |
|---|---|
| Resumable batch inference over the cache | `python scripts/p2_run_inference.py` (env `MODELS`, `CONFIGS`, `N_PAGES`) |
| Scoring of every prediction set | `WORKERS=4 python scripts/p2_evaluate.py` |
| Outputs | `RDLD_data/predictions/<model>/<config>.json`, `RDLD_data/results/p2/evaluation/metrics_long.csv` |
| Logs | `RDLD_data/results/p2/logs/p2_run_inference*.log`, `p2_evaluate.log` |

Scope run: 3 detectors × 109 conditions (clean + 24 singles + 84 same-severity pairs) = **327 prediction
sets**, plus the 28 reversed-order (`rev-*`) and 36 second-seed (`r1-*`) configurations for P3/P4
= **519 prediction sets**, 3.2 GB of compact JSON. Same settings as P1: score floor 0.01, batch 8,
YOLO imgsz 1024, ≤ 300 detections/page. A two-batch thread-pool prefetch overlaps PNG decode with GPU work.

`metrics_long.csv` holds one row per detector × configuration: mAP, mAP50, mAP75, AP by size,
per-class AP, the configuration descriptors (kind, severity, a, b), `D_pts` (clean − degraded, points)
and `D_rel` (relative drop). `mrd.csv` / `rd_per_degradation.csv` are written once the image-quality
table (Workstream A, `iq_per_config.csv`) exists.

## 3. Results (evidence)

### 3.1 Mean mAP drop from clean, points (`p2/evaluation/metrics_long.csv`)

| Detector | single s1 | single s2 | single s3 | pair s1 | pair s2 | pair s3 |
|---|---|---|---|---|---|---|
| YOLO | 8.80 | 14.60 | 19.10 | 17.29 | 27.48 | 33.76 |
| heron | 5.27 | 10.71 | 16.54 | 10.27 | 21.54 | 31.21 |
| egret | 6.81 | 12.62 | 18.69 | 13.24 | 24.52 | 34.68 |

A pair costs roughly twice a single at the same severity for all three detectors - the first, crude
sign that compounding is close to additive on average (tested properly in P3).

### 3.2 Single-degradation drops (`p2/logs/p2_evaluate.log`)

| Degradation | YOLO s1/s2/s3 | heron s1/s2/s3 | egret s1/s2/s3 |
|---|---|---|---|
| speckle | 24.7 / 50.3 / 59.9 | 17.6 / 30.6 / 40.1 | 17.9 / 32.1 / 42.4 |
| texture | 42.0 / 44.8 / 43.8 | 18.8 / 20.9 / 20.0 | 27.5 / 25.9 / 24.9 |
| vibration | 0.0 / 17.5 / 35.9 | 0.8 / 24.7 / 44.6 | 2.5 / 25.8 / 46.7 |
| defocus | 0.0 / −0.1 / 2.0 | 0.0 / 1.3 / 12.4 | 0.0 / 4.3 / 16.5 |
| watermark | 4.4 / 4.4 / 6.2 | 2.4 / 3.6 / 3.9 | 2.0 / 4.4 / 4.3 |
| ink_bleeding | −0.8 / −0.5 / 2.6 | 0.9 / 2.9 / 7.6 | 1.1 / 3.2 / 7.7 |
| ink_holdout | −0.7 / −0.6 / 0.9 | 0.1 / 0.6 / 2.6 | 0.8 / 2.4 / 4.4 |
| illumination | 0.9 / 1.0 / 1.4 | 1.6 / 1.0 / 1.2 | 2.6 / 2.7 / 2.6 |

- **Speckle and vibration dominate**; ink and illumination effects are small.
- **Texture is severity-flat and hits YOLO twice as hard** (≈ 43 points at every level). RoDLA's texture
  operation re-renders the page and drops colour; the re-render itself, not the texture strength, is what
  YOLO fails on. This is an architecture-specific sensitivity, not a severity effect.
- **YOLO is robust to blur** (defocus s3: 2.0 points vs 12–17 for the transformers) but **most fragile
  to speckle**. The detectors have different weak spots - the precondition for RQ3 ranking reversals.

### 3.3 Correctness checks
- **Byte-identity control.** `defocus@1` is a pixel no-op (Gaussian kernel 1): its images are identical
  to clean, and after the fix in §5 its predictions are identical to the clean predictions for all
  three detectors (0 differing boxes). The same holds for every `X+defocus@1` vs `X` pair.
- **Corruption check** after the interrupted first run: all saved prediction files parse.

## 4. Decisions and why
1. **One prediction file per detector × configuration, cached.** Analyses (bootstrap, structural errors,
   OCR) re-read predictions many times; inference runs once.
2. **Resumable, file-level checkpointing.** A configuration is skipped if its file exists. This let the
   run survive a session crash with no recomputation.
3. **Compact JSON** (bbox to 0.01 px, score to 1e-5). Smaller files with no measurable effect on mAP - but it
   must be applied to *every* prediction set, including clean (see §5).
4. **mRD on the IQA-only mPE**, model-independent (Workstream A, P2), and configurations with
   mPE < 1 are excluded from mRD - `defocus@1` (mPE = 0) would otherwise divide by zero.

## 5. Problems found
- **Precision mismatch between clean and degraded predictions (found by the negative control, fixed).**
  P1 saved clean predictions at full float precision; P2 saved degraded ones rounded. The `defocus@1`
  pairs, whose interaction must be exactly 0, came out at up to 0.026 points. Fix: the three `clean.json`
  files were rounded with the same `compact()` used by P2 (originals kept in
  `RDLD_data/predictions_p1_fullprecision/`) and the clean bootstrap tables were rebuilt. Control now
  exactly 0. Effect on P1 numbers: YOLO 68.80 → 68.82, heron 64.23 → 64.22 - unchanged at one decimal.
- **GPU sharing slowed both jobs ~20×.** Running inference and the image-quality pass on the GPU at the
  same time took heron to ≈ 500 s per configuration (vs ≈ 40 s alone). GPU jobs are now run one at a time.
- **Background jobs died with the session.** Long jobs are now launched as detached Windows processes.
- **Timings in `p2/inference/inference_timing.csv` are not clean throughput numbers** (median 134–179
  ms/page end-to-end) because other jobs shared the machine; the reference throughput remains P1's
  idle-machine figures (50–75 ms/page).

## 6. Main takeaway
O4 achieved. The measurement apparatus is validated by an exact negative control (21 pairs,
interaction = 0.00). Single-degradation profiles differ sharply by architecture (texture: YOLO 43 vs
heron 20 points; defocus s3: YOLO 2 vs egret 17), motivating RQ3.

## 7. Hand-off
- P3 (Khizar): predictions → paired bootstrap tables → RQ1 interaction analysis.
- P3 (Alishba): `metrics_long.csv` for image-quality vs accuracy; `rev-*` predictions for the order check.
- P4: `r1-*` predictions for seed stability; all predictions for structural errors.
- After Workstream A's `iq_per_config.csv` lands: `python scripts/p2_evaluate.py --mrd-only` adds
  `rd_per_degradation.csv` and `mrd.csv` from the existing scores (no re-scoring).

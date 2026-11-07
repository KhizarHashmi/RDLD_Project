# P2 - Pipeline - Alishba (Workstream A)

**Dates:** 2026-11-03 – 2026-11-04

## 1. Goal
Build the experimental stimulus: a seeded, reproducible composition engine over RoDLA's image
degradations, a cache of every degraded page the study needs, and a detector-independent measure of
how much each configuration changes the image.

## 2. What was done

| Item | File / command | Output |
|---|---|---|
| Composition engine | `rdld/degradations.py` | `apply_one`, `apply_config`, `experiment_configs()` (109 rows) |
| Cache generator | `python scripts/p2_generate_cache.py` (`--check` for equivalence) | `RDLD_data/degraded/<config>/*.png`, `configs.csv` |
| Image-quality metrics | `rdld/image_quality.py` | PSNR, MS-SSIM, CW-SSIM |
| Image-quality pass | `python scripts/p2_image_quality.py` (`--sanity` for checks only) | `RDLD_data/results/p2/image_quality/` |

Logs: `RDLD_data/results/p2/logs/p2_generate_cache.log`, `p2_image_quality.log`.

**Design.** 8 image-only RoDLA degradations (texture, ink_bleeding, ink_holdout, speckle, watermark,
illumination, defocus, vibration) × 3 severities → 24 singles, plus all 28 unordered pairs at the same
severity × 3 = 84 pairs; with clean, **109 conditions × 300 pages = 32,400 images (13.6 GB)**.

**Composition order.** Pairs are applied in a fixed physical order, matching how a document is produced
and then captured: texture → ink_bleeding → ink_holdout → speckle → watermark → illumination → defocus →
vibration (paper → printing → page damage → lighting → camera). The alternative order is tested in P3.

**Seeding - common random numbers.** The seed for each step is derived from md5(name | severity | page |
rep). A pair A+B therefore uses *exactly* the same random draw for A as the single A does, so the
single and the pair differ only by the added degradation - the image-level analogue of a paired design.
Pairs are generated from the cached single A, then B is applied.

**Image-quality measures.** Against the clean page: PSNR, MS-SSIM and CW-SSIM (own implementation:
Fourier-domain complex steerable bank, 4 scales × 8 orientations, 7 × 7 window, K = 0.01). Per
configuration:

  mPE_IQA = 100 × mean(1 − MS-SSIM, 1 − CW-SSIM)

## 3. Results (evidence)

### 3.1 Reproducibility
`p2_generate_cache.py --check`: regenerating a sample of singles and pairs from scratch gives
byte-identical PNGs; a pair built from the cached single equals the pair built in one pass.

### 3.2 Metric sanity (`p2/image_quality/sanity.csv`)
For defocus, texture and ink_bleeding: identity gives 1.0000 on both SSIMs, and scores fall
monotonically with severity - all PASS. **`defocus@1` is an exact identity** (RoDLA's Gaussian kernel at
severity 1 has size 1): mPE = 0, pixels identical to clean.

### 3.3 Image change per single degradation (`p2/image_quality/iq_per_config.csv`, mPE_IQA)

| Degradation | s1 | s2 | s3 |
|---|---|---|---|
| texture | 28.52 | 39.92 | 44.20 |
| speckle | 10.50 | 21.52 | 28.64 |
| vibration | 1.16 | 10.12 | 16.16 |
| ink_bleeding | 0.43 | 4.35 | 12.25 |
| defocus | 0.00 | 2.29 | 8.01 |
| illumination | 5.78 | 7.54 | 7.33 |
| ink_holdout | 0.21 | 1.96 | 4.88 |
| watermark | 0.63 | 2.08 | 3.05 |

Mean mPE_IQA: singles 5.9 / 11.2 / 15.6, pairs 11.4 / 20.5 / 27.0 (s1 / s2 / s3). Figure:
`fig_iq_by_degradation.png`.

### 3.4 Cost (`p2/generation/generation_timing.csv`)
Per image under 14-way parallel load: texture 3.4–6.7 s (it re-renders the page), ink_holdout 1.8–2.8 s,
ink_bleeding ≈ 1.6 s, defocus ≈ 0.15–0.22 s. Pairs from cached singles: 1,080 s for all 25,200.
Image-quality pass: ≈ 100–140 s per configuration on the GPU.

## 4. Decisions and why
1. **Fixed physical order** rather than random order per page: makes every pair a single well-defined
   stimulus; whether order matters is measured separately (P3) instead of being mixed into noise.
2. **Common random numbers** across singles and pairs: removes "different random draw" as an explanation
   for any interaction.
3. **IQA-only mPE.** RoDLA's mPE adds the mAP drop of a reference Faster R-CNN. Using one of our three
   detectors as the reference would make mRD partly divide a model's loss by itself; an IQA-only mPE is
   model-independent (RoDLA's own first requirement). This is a documented deviation.
4. **CW-SSIM implemented in-house** rather than via `pyiqa`, keeping the detector environment's
   dependency set unchanged; validated by the sanity checks.
5. **PNG, compression level 1:** lossless (JPEG would add its own degradation) and fast to write.

## 5. Problems found
- **`defocus@1` is a no-op.** Not a bug in our code - RoDLA's severity-1 kernel is size 1. Turned into an
  asset: 7 pairs per detector become an exact negative control for RQ1 (interaction must be 0). Excluded
  from mRD (mPE = 0).
- **Texture re-renders the page** (normalises intensities, drops colour), so even texture@1 changes the
  image a lot (CW-SSIM 0.44 on the sanity page). This is why texture's effect is severity-flat for the
  detectors and why it dominates the order check (P3).
- **Illumination randomly chooses glare or shadow** per page; its mPE is therefore flat across severity
  (5.8 / 7.5 / 7.3).
- **The first image-quality run was lost** when the session ended (results were only written at the end)
  and it was 20× slower because it shared the GPU with inference. The script now writes one file per
  configuration (`per_config/`) and resumes; it was re-run alone.

## 6. Main takeaway
O2 achieved - full 8 × 8 × 3 same-severity factorial with common random numbers; the
perturbation measure is model-independent by construction. Documented deviation from RoDLA's mPE.

## 7. Hand-off
- Workstream B: `RDLD_data/degraded/` + `configs.csv` for inference; `iq_per_config.csv` for mRD
  (`p2_evaluate.py --mrd-only`).
- P3 (Alishba): reversed-order cache for the order check; image-level interaction (I_img) vs detector
  interaction.

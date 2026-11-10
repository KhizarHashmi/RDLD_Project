# P3 - Main Experiment - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to eight detectors on 2026-11-08, and to eleven detectors and the test pages on 2026-11-10)

## 1. Goal
Two checks on the RQ1 design. (1) The main experiment fixes one composition order; does reversing it
change the result? (2) When a pair compounds for a detector, is that because the pixels themselves are
damaged super-additively, or because the detector reacts super-additively to an additively damaged image?

## 2. What was done

| Step | Command | Output |
|---|---|---|
| Reversed-order cache, all 28 severity-2 pairs (B applied before A) | `python scripts/p3_order_check.py` | `RDLD_data/degraded/rev-<config>/`, `configs_order.csv` (8,400 images) |
| Order analysis, image and detection level | `python scripts/p3_order_check.py --analyse` | `RDLD_data/results/p3/order/order_effect.csv` |
| Image damage vs detection damage | `python scripts/p3_iq_vs_accuracy.py` | `RDLD_data/results/p3/iq_vs_accuracy/` |
| The same on the test pages | `RDLD_DATA=<test root> python scripts/p3_iq_vs_accuracy.py` | `RDLD_data_test/results/p3/iq_vs_accuracy/` |

Inference and paired bootstrap tables for the `rev-*` configurations were produced by Workstream B with
the same settings and the same resample index as the main experiment.

**Detectors (11):** YOLO26-m (`yolo`), YOLO26-l, YOLOv11-m, YOLOv11-l, heron, heron101, egret, egretxl, DETR,
Deformable DETR and RF-DETR. The analyses were first run on three detectors, then on eight, then on eleven, with the
scripts unchanged; the numbers for earlier detectors stay the same.

**Page sets:** the 300 validation pages, and 300 pages from the official DocLayNet test split (P7). The order check
exists for the validation pages only.

**Order effect** = mAP(reversed) − mAP(forward), with a 95% paired page-bootstrap CI; "significant" if the
CI excludes 0. Image level: MS-SSIM between the forward and reversed image (60 pages), and the share of
pixel-identical pages.

**Image-level interaction**, same additive test as RQ1 but on image damage d = 1 − MS-SSIM:
I_img = d(A+B) − d(A) − d(B).

## 3. Results (evidence)

### 3.1 Composition order (`p3/order/order_effect.csv`, validation pages)

- **164 of 308** detector × pair cells have a significant order effect; median |Δ| = 0.52 points.
  Per detector (of 28): YOLO26-m 16, YOLO26-l 19, YOLOv11-m 21, YOLOv11-l 16, heron 11, heron101 14, egret 19,
  egretxl 12, DETR 12, Deformable DETR 15, RF-DETR 9.
- **Texture accounts for most of it.** Pairs containing texture: 63 of 77 cells significant, median
  |Δ| = **3.24** points. Pairs without texture: 101 of 231 significant, median |Δ| = **0.34** points.

| Detector | texture + defocus | texture + ink_bleeding | texture + vibration | texture + ink_holdout | ink_bleeding + speckle | ink_holdout + speckle |
|---|---|---|---|---|---|---|
| YOLO26-m | **−23.43** | −15.89 | −11.87 | −8.00 | +1.16 | −3.77 |
| YOLO26-l | −19.43 | −12.08 | −17.19 | −6.34 | +2.20 | −2.83 |
| YOLOv11-m | −18.19 | −10.36 | −5.26 | −7.34 | −1.27 | −4.09 |
| YOLOv11-l | −22.75 | −16.06 | +0.73 | −7.40 | +1.03 | −2.75 |
| heron | −7.15 | −9.28 | −10.56 | +0.33 | +5.60 | −4.15 |
| heron101 | −9.15 | −7.97 | −13.39 | +0.54 | +5.37 | −3.57 |
| egret | −4.73 | −7.13 | −6.28 | +0.53 | +6.41 | −5.05 |
| egretxl | −7.32 | −9.00 | −3.33 | +0.53 | +6.08 | −4.90 |
| DETR | −8.95 | −13.69 | −2.59 | +0.79 | +3.10 | −0.90 |
| Def. DETR | −12.32 | −8.21 | −5.56 | −0.51 | +2.29 | −1.23 |
| RF-DETR | −5.11 | −6.63 | −11.25 | +1.66 | +1.50 | −0.07 |
| MS-SSIM fwd vs rev | 0.978 | 0.974 | 0.918 | 0.991 | 0.991 | 0.995 |

All pairs at severity 2. Values are Δ in mAP points.

Reading: in the reversed order texture is applied *last*, re-rendering a page that is already blurred or
inked (RoDLA's texture step normalises intensities and drops colour). For texture + defocus and texture + ink_bleeding
this is clearly worse for **all eleven** detectors (every cell significant), and for texture + vibration for ten of them
(YOLOv11-l +0.73, n.s., is the exception). The four YOLO models also lose 6–8 points on texture + ink_holdout, where the other detectors barely move. YOLO
is the most texture-sensitive family: median |Δ| on texture pairs 8.0 (YOLO26-m), 6.3 (YOLO26-l) and 5.3 (YOLOv11-m)
points, against 1.7–4.7 for the others (YOLOv11-l 1.9). Outside texture, order changes mAP by well under a point for most
pairs; the speckle × ink pairs are the main exception (up to ≈ 6 points for the four Docling models).

### 3.2 Does image damage predict detector damage? (`p3/iq_vs_accuracy/iq_vs_drop_correlation.csv`, Spearman ρ over 108 configurations)

| Detector | val mPE_IQA | val PSNR | test mPE_IQA | test PSNR |
|---|---|---|---|---|
| YOLO26-m | 0.880 | −0.467 | 0.877 | −0.514 |
| YOLO26-l | 0.864 | −0.441 | 0.862 | −0.513 |
| YOLOv11-m | 0.852 | −0.528 | 0.828 | −0.612 |
| YOLOv11-l | 0.853 | −0.549 | 0.829 | −0.628 |
| heron | 0.835 | −0.629 | 0.807 | −0.638 |
| heron101 | 0.843 | −0.630 | 0.815 | −0.674 |
| egret | 0.858 | −0.650 | 0.853 | −0.679 |
| egretxl | 0.907 | −0.658 | 0.878 | −0.681 |
| DETR | 0.779 | −0.540 | 0.802 | −0.587 |
| Def. DETR | 0.823 | −0.458 | 0.865 | −0.588 |
| RF-DETR | 0.841 | −0.527 | 0.880 | −0.625 |

For every detector on both page sets, the structural measure (mPE from MS-SSIM + CW-SSIM) tracks detector loss far
better than PSNR. Singles-only and pairs-only correlations are in the same file and tell the same story.

### 3.3 Pixel interaction vs detector interaction (`p3/iq_vs_accuracy/interaction_correlation.csv`, 84 pairs)

| Detector | val ρ(I_img, I_mult) | p | test ρ | p |
|---|---|---|---|---|
| YOLO26-m | +0.187 | 0.089 | +0.129 | 0.243 |
| YOLO26-l | +0.301 | 0.005 | +0.190 | 0.083 |
| YOLOv11-m | +0.101 | 0.362 | +0.075 | 0.499 |
| YOLOv11-l | +0.101 | 0.362 | +0.023 | 0.838 |
| heron | −0.261 | 0.017 | −0.205 | 0.061 |
| heron101 | −0.174 | 0.113 | −0.080 | 0.471 |
| egret | −0.325 | 0.003 | −0.100 | 0.364 |
| egretxl | −0.251 | 0.021 | −0.280 | 0.010 |
| DETR | +0.064 | 0.560 | −0.099 | 0.371 |
| Def. DETR | +0.077 | 0.486 | +0.217 | 0.047 |
| RF-DETR | −0.181 | 0.100 | −0.220 | 0.045 |

Share of pairs with I_img > 0: **0.179** on validation and **0.167** on test (the same images for every detector).

Only 17–18% of pairs damage the *pixels* super-additively, yet every detector shows a positive median
interaction (RQ1). For the Docling models the correlation between pixel and detector interaction is negative (significant
for heron, egret and egretxl on validation, and for egretxl and RF-DETR on test): the pairs whose pixel damage compounds least are
the ones where detection compounds most. Two positive correlations reach p < 0.05 (YOLO26-l on validation, Deformable DETR on
test), each weak and not repeated on the other page set. **Compounding is created by the detector, not inherited from the image.**

## 4. Decisions and why
1. **Order check at severity 2 only, all 28 pairs** - the middle severity, where interactions are largest
   and saturation is not yet a factor; 8,400 extra images instead of 25,200.
2. **Paired bootstrap on the same resample index** as the main experiment, so the order effect is a
   paired difference, comparable in precision to the interaction terms.
3. **Image damage measured with MS-SSIM** for I_img: bounded, structural, and on the same scale for
   every degradation (PSNR is unbounded for near-identity cases such as defocus@1).
4. **No order check on the test pages.** The order effect is a property of the degradation pipeline, which is the same
   for both page sets; the GPU time went to the eleven detectors instead.

## 5. Problems found
- **Order is not a nuisance for texture.** Any result that involves texture depends on where texture sits
  in the pipeline. The main experiment keeps the physical order (texture first = paper texture exists
  before printing); the write-up must state this and report texture pairs with that caveat.
- PSNR is unbounded for identical images (defocus@1), so mean PSNR over singles at s1 is not meaningful;
  only rank correlations are used.

## 6. Main takeaway
Threat to validity addressed: composition order is a real factor for texture and is controlled by
fixing a physically motivated order; this holds for all eleven detectors, and most strongly for the YOLO family. Mechanism
for RQ1: compounding is a *model* property - only 17–18% of pairs compound at pixel level on either page set, no detector's
compounding consistently follows the pixel compounding, and for the Docling models the two are negatively correlated.

## 7. Hand-off
- RQ1 write-up (Khizar): cite §3.3 as the mechanism and §3.1 as the order caveat for texture pairs.
- P5: the order caveat applies to downstream results on texture pairs too.

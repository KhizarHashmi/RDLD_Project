# P3 - Main Experiment - Alishba (Workstream A)

**Dates:** 2026-11-04 (results updated to all eight detectors on 2026-11-08)

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

Inference and paired bootstrap tables for the `rev-*` configurations were produced by Workstream B with
the same settings and the same resample index as the main experiment.

**Detectors:** YOLO, heron, egret, heron101, egretxl, DETR, Deformable DETR and RF-DETR. The analyses were
first run on the first three and re-run unchanged on all eight; the numbers for the first three are the same.

**Order effect** = mAP(reversed) − mAP(forward), with a 95% paired page-bootstrap CI; "significant" if the
CI excludes 0. Image level: MS-SSIM between the forward and reversed image (60 pages), and the share of
pixel-identical pages.

**Image-level interaction**, same additive test as RQ1 but on image damage d = 1 − MS-SSIM:
I_img = d(A+B) − d(A) − d(B).

## 3. Results (evidence)

### 3.1 Composition order (`p3/order/order_effect.csv`, `p3/logs/p3_order_8det.log`)

- **108 of 224** detector × pair cells have a significant order effect; median |Δ| = 0.47 points.
  Per detector (of 28): YOLO 16, heron 11, egret 19, heron101 14, egretxl 12, DETR 12, Deformable DETR 15,
  RF-DETR 9.
- **Texture accounts for most of it.** Pairs containing texture: 43 of 56 cells significant, median
  |Δ| = **3.06** points. Pairs without texture: 65 of 168 significant, median |Δ| = **0.32** points.

| Pair (forward order) | YOLO | heron | egret | heron101 | egretxl | DETR | Def. DETR | RF-DETR | MS-SSIM fwd vs rev |
|---|---|---|---|---|---|---|---|---|---|
| texture@2 + defocus@2 | **−23.43** | −7.15 | −4.73 | −9.15 | −7.32 | −8.95 | −12.32 | −5.11 | 0.978 |
| texture@2 + ink_bleeding@2 | −15.89 | −9.28 | −7.13 | −7.97 | −9.00 | −13.69 | −8.21 | −6.63 | 0.974 |
| texture@2 + vibration@2 | −11.87 | −10.56 | −6.28 | −13.39 | −3.33 | −2.59 | −5.56 | −11.25 | 0.918 |
| texture@2 + ink_holdout@2 | −8.00 | +0.33 | +0.53 | +0.54 | +0.53 | +0.79 | −0.51 | +1.66 | 0.991 |
| ink_bleeding@2 + speckle@2 | +1.16 | +5.60 | +6.41 | +5.37 | +6.08 | +3.10 | +2.29 | +1.50 | 0.991 |
| ink_holdout@2 + speckle@2 | −3.77 | −4.15 | −5.05 | −3.57 | −4.90 | −0.90 | −1.23 | −0.07 | 0.995 |

Values are Δ in mAP points.

Reading: in the reversed order texture is applied *last*, re-rendering a page that is already blurred or
inked (RoDLA's texture step normalises intensities and drops colour). For texture + defocus / ink_bleeding
/ vibration this is clearly worse for **all eight** detectors (Δ < 0, every cell significant), most of all
for YOLO, which is the most texture-sensitive detector (P2; median |Δ| on texture pairs 8.0 points, against
1.7–4.7 for the others). Outside texture, order changes mAP by well under a point for most pairs; the
speckle × ink pairs are the main exception (up to ≈ 6 points for the four Docling models, smaller for DETR,
Deformable DETR and RF-DETR).

### 3.2 Does image damage predict detector damage? (`p3/iq_vs_accuracy/iq_vs_drop_correlation.csv`, Spearman ρ over configurations)

| Configurations | image metric | YOLO | heron | egret | heron101 | egretxl | DETR | Def. DETR | RF-DETR |
|---|---|---|---|---|---|---|---|---|---|
| all 108 | mPE_IQA | 0.880 | 0.835 | 0.858 | 0.843 | 0.907 | 0.779 | 0.823 | 0.841 |
| all 108 | PSNR | −0.467 | −0.629 | −0.650 | −0.630 | −0.658 | −0.540 | −0.458 | −0.527 |
| singles | mPE_IQA | 0.823 | 0.856 | 0.893 | 0.845 | 0.913 | 0.742 | 0.712 | 0.761 |
| pairs | mPE_IQA | 0.866 | 0.804 | 0.828 | 0.809 | 0.883 | 0.745 | 0.812 | 0.826 |

For every detector, the structural measure (mPE from MS-SSIM + CW-SSIM) tracks detector loss far better
than PSNR.

### 3.3 Pixel interaction vs detector interaction (`p3/iq_vs_accuracy/interaction_correlation.csv`)

| Detector | n pairs | ρ(I_img, I_mult) | p |
|---|---|---|---|
| YOLO | 84 | +0.187 | 0.089 |
| heron | 84 | −0.261 | 0.017 |
| egret | 84 | −0.325 | 0.003 |
| heron101 | 84 | −0.174 | 0.113 |
| egretxl | 84 | −0.251 | 0.021 |
| DETR | 84 | +0.064 | 0.560 |
| Deformable DETR | 84 | +0.077 | 0.486 |
| RF-DETR | 84 | −0.181 | 0.100 |

Share of pairs with I_img > 0: **0.179** (the same images for every detector).

Only 18% of pairs damage the *pixels* super-additively, yet the detectors show positive median
interactions (RQ1). No detector shows a significant positive correlation between pixel interaction and
detector interaction; for heron, egret and egretxl it is significantly **negative**: the pairs whose pixel
damage compounds least are the ones where detection compounds most. **Compounding is created by the
detector, not inherited from the image.**

## 4. Decisions and why
1. **Order check at severity 2 only, all 28 pairs** - the middle severity, where interactions are largest
   and saturation is not yet a factor; 8,400 extra images instead of 25,200.
2. **Paired bootstrap on the same resample index** as the main experiment, so the order effect is a
   paired difference, comparable in precision to the interaction terms.
3. **Image damage measured with MS-SSIM** for I_img: bounded, structural, and on the same scale for
   every degradation (PSNR is unbounded for near-identity cases such as defocus@1).

## 5. Problems found
- **Order is not a nuisance for texture.** Any result that involves texture depends on where texture sits
  in the pipeline. The main experiment keeps the physical order (texture first = paper texture exists
  before printing); the write-up must state this and report texture pairs with that caveat.
- PSNR is unbounded for identical images (defocus@1), so mean PSNR over singles at s1 is not meaningful;
  only rank correlations are used.

## 6. Main takeaway
Threat to validity addressed: composition order is a real factor for texture and is controlled by
fixing a physically motivated order; this holds for all eight detectors. Mechanism for RQ1: compounding is
a *model* property - only 18% of pairs compound at pixel level, no detector's compounding follows the pixel
compounding, and for three detectors the two are negatively correlated (ρ = −0.25 to −0.33).

## 7. Hand-off
- RQ1 write-up (Khizar): cite §3.3 as the mechanism and §3.1 as the order caveat for texture pairs.
- P5: the order caveat applies to downstream results on texture pairs too.

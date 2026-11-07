# P3 - Main Experiment - Alishba (Workstream A)

**Dates:** 2026-11-04

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

**Order effect** = mAP(reversed) − mAP(forward), with a 95% paired page-bootstrap CI; "significant" if the
CI excludes 0. Image level: MS-SSIM between the forward and reversed image (60 pages), and the share of
pixel-identical pages.

**Image-level interaction**, same additive test as RQ1 but on image damage d = 1 − MS-SSIM:
I_img = d(A+B) − d(A) − d(B).

## 3. Results (evidence)

### 3.1 Composition order (`p3/order/order_effect.csv`, `p3/logs/p3_order_analyse.log`)

- **46 of 84** detector × pair cells have a significant order effect (YOLO 16, heron 11, egret 19 of 28);
  median |Δ| = 0.51 points.
- **Texture accounts for most of it.** Pairs containing texture: 17 of 21 cells significant, median
  |Δ| = **4.73** points. Pairs without texture: 29 of 63 significant, median |Δ| = **0.40** points.

| Pair (forward order) | Δ YOLO | Δ heron | Δ egret | MS-SSIM fwd vs rev |
|---|---|---|---|---|
| texture@2 + defocus@2 | **−23.43** | −7.15 | −4.73 | 0.978 |
| texture@2 + ink_bleeding@2 | −15.89 | −9.28 | −7.13 | 0.974 |
| texture@2 + vibration@2 | −11.87 | −10.56 | −6.28 | 0.918 |
| texture@2 + ink_holdout@2 | −8.00 | +0.33 | +0.53 | 0.991 |
| ink_bleeding@2 + speckle@2 | +1.16 | +5.60 | +6.41 | 0.991 |
| ink_holdout@2 + speckle@2 | −3.77 | −4.15 | −5.05 | 0.995 |

Reading: in the reversed order texture is applied *last*, re-rendering a page that is already blurred or
inked (RoDLA's texture step normalises intensities and drops colour). For texture + defocus / ink_bleeding
/ vibration this is clearly worse for every detector (Δ < 0), most of all for YOLO, which is already the
most texture-sensitive detector (P2). Outside texture, order changes mAP by well under a point for most
pairs; the speckle × ink pairs are the main exception (≈ 4–6 points).

### 3.2 Does image damage predict detector damage? (`p3/iq_vs_accuracy/iq_vs_drop_correlation.csv`, Spearman ρ over configurations)

| Configurations | image metric | YOLO | heron | egret |
|---|---|---|---|---|
| all 108 | mPE_IQA | 0.880 | 0.835 | 0.858 |
| all 108 | PSNR | −0.467 | −0.629 | −0.650 |
| singles | mPE_IQA | 0.823 | 0.856 | 0.893 |
| pairs | mPE_IQA | 0.866 | 0.804 | 0.828 |

The structural measure (mPE from MS-SSIM + CW-SSIM) tracks detector loss far better than PSNR.

### 3.3 Pixel interaction vs detector interaction (`p3/iq_vs_accuracy/interaction_correlation.csv`)

| Detector | n pairs | ρ(I_img, I_mult) | p | share of pairs with I_img > 0 |
|---|---|---|---|---|
| YOLO | 84 | +0.187 | 0.089 | 0.179 |
| heron | 84 | −0.261 | 0.017 | 0.179 |
| egret | 84 | −0.325 | 0.003 | 0.179 |

Only 18% of pairs damage the *pixels* super-additively, yet the detectors show positive median
interactions (RQ1). And for the transformers the correlation is **negative**: the pairs whose pixel damage
compounds least are the ones where detection compounds most. **Compounding is created by the detector,
not inherited from the image.**

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
fixing a physically motivated order. Mechanism for RQ1: compounding is a *model* property - only 18% of
pairs compound at pixel level, and pixel and detector interactions are negatively correlated for both
transformer detectors (ρ = −0.26, −0.33).

## 7. Hand-off
- RQ1 write-up (Khizar): cite §3.3 as the mechanism and §3.1 as the order caveat for texture pairs.
- P5: the order caveat applies to downstream results on texture pairs too.

# P1 - Corpus Validation & EDA - Alishba (Workstream A)

**Dates:** 2026-11-03

## 1. Goal
Confirm the ground truth is sound before any metric is computed on it, and characterise the clean
corpus so later degradation effects can be interpreted (which classes, sizes and document types are
being tested).

## 2. What was done

| Script | Command | Output |
|---|---|---|
| `scripts/p1_validate_corpus.py` | `python scripts/p1_validate_corpus.py` | `RDLD_data/results/p1/validation/{checks.csv, issues.csv}` |
| `scripts/p1_eda.py` | `python scripts/p1_eda.py` | `RDLD_data/results/p1/eda/` - 5 CSVs, 4 figures |

Logs: `RDLD_data/results/p1/logs/p1_validate_corpus.log`, `p1_eda.log`.

## 3. Results (evidence)

### 3.1 Validation - 13/14 checks pass (`validation/checks.csv`)

| Check | Result |
|---|---|
| Images open, sizes match COCO records | 300/300, all 1025×1025 RGB |
| Boxes positive size / valid class ids | 3,530/3,530 |
| Stored `area` = box w×h | ratio exactly 1.000 for every box |
| Exact duplicate GT boxes | 0 |
| Same region annotated with two classes (IoU ≥ 0.9) | **0** - GT has no label ambiguity of the kind the detectors show |
| Manifest agrees with annotations; 50 pages per category | yes |
| Boxes inside page (±1 px) | **WARN - 1/3,530**: page `0022.png`, a full-width box overshooting by 1.3 px left / 0.5 px right. IoU effect < 0.2%; kept unmodified and documented |

### 3.2 EDA (`eda/*.csv`, figures `eda/fig_*.png`)

**Class composition depends strongly on document category** (`class_by_category.csv`, `fig_class_by_category.png`):

| Finding | Evidence |
|---|---|
| Formula is almost exclusively scientific | 48 of 50 Formula boxes in scientific articles |
| Footnote is almost exclusively legal | 40 of 41 in laws & regulations |
| Tables concentrate in two categories | financial 40 + tenders 37 = 77 of 102 |
| Pictures concentrate in financial reports | 63 of 119; manuals have 0 |
| Structural gaps | laws have 0 Page-headers; tenders have 0 Page-footers and 0 Captions |
| Title | 6 boxes total, at most 2 per category |

**Region size** (`size_buckets.csv`, `fig_box_size_by_class.png`) - COCO buckets: small 370 · medium 1,456 · large 1,704.

| Class | small / medium / large | Note |
|---|---|---|
| Page-footer | **150** / 65 / 0 | 70% small - the class most exposed to blur |
| Section-header | 98 / 436 / 44 | thin single-line boxes |
| Page-header | 64 / 243 / 5 | |
| Table | 0 / 2 / 100 | large; failure is about structure, not visibility |
| Picture | 0 / 13 / 106 | large |

**Per document category** (`summary_by_category.csv`, `fig_regions_per_page.png`, `fig_page_stats.png`):

| Category | regions/page (mean ± sd) | coverage | ink ratio | colour pages |
|---|---|---|---|---|
| financial_reports | 12.3 ± **14.8** | 0.44 | 0.063 | **17 / 50** |
| scientific_articles | 9.3 ± 4.0 | 0.41 | 0.020 | 0 |
| government_tenders | 10.5 ± 4.1 | 0.32 | 0.044 | 1 |
| laws_and_regulations | 8.3 ± 3.3 | 0.42 | 0.051 | 4 |
| manuals | **19.6** ± 7.2 | 0.30 | 0.030 | 1 |
| patents | 10.6 ± 4.3 | 0.47 | 0.072 | 0 |

## 4. Decisions and why
1. **Do not modify the ground truth.** The single overshooting box is a sub-pixel annotation artefact;
   clipping it would make our GT differ from the published dataset.
2. **Per-category comparisons must control for class mix.** Because categories contain different
   classes (formulae only in science, footnotes only in law, tables mostly in financial/tenders), a raw
   per-category mAP difference is partly a class-composition effect. Later phases report per-class
   results first, and per-category results with the class mix stated.
3. **Feature tables saved now** (`box_features.csv`, `page_features.csv`) so P3/P4 can join
   degradation sensitivity against page/box properties (size bucket, ink ratio, colour, density)
   without recomputing.
4. **Colour is not uniform across categories** (17 colour pages in financial reports, ~0 elsewhere).
   Degradations that act on colour/illumination may hit financial reports differently; recorded as a
   covariate, not corrected.

## 5. Problems found
- **Title (n = 6)** cannot support per-class conclusions - confirmed at the category level too.
  Recommendation for P4: report Title with an explicit caveat and exclude it from per-class rankings.
- **Formula (50) and Footnote (41) are each confined to one category**, so their per-class AP is
  effectively a single-category measurement. State this when reporting.
- Financial reports have very high layout variance (sd 14.8 regions/page) - expect wider CIs there.

## 6. Main takeaway
Validity: ground-truth integrity verified before measurement (no duplicate or ambiguous
annotations), so measured degradation effects are not artefacts of label noise in the reference.
Stratification by category is justified by the class-mix differences found in the EDA.

## 7. Hand-off
- To P2 (Alishba): page ids and file names are stable; the perturbation cache should mirror
  `RDLD_data/doclaynet/images/` file names so ground truth applies unchanged.
- To P3/P4 (both): join on `image_id` / `ann_id` with `eda/page_features.csv` and `eda/box_features.csv`.

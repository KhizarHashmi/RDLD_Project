# P0 - Feasibility & Data Acquisition - Alishba (Workstream A)

**Dates:** 2026-09-06 – 2026-09-27 (re-verified 2026-11-03 in the RDLD_Project layout)

## 1. Goal
Establish that (a) the RoDLA degradation functions run outside RoDLA's heavy stack, (b) they can be
**composed** on one image - the property the whole study depends on - and (c) a stratified,
annotated DocLayNet corpus can be built without manual labelling.

## 2. What was done

| Script | Command | Output |
|---|---|---|
| `scripts/p0_verify_perturbations.py` | `python scripts/p0_verify_perturbations.py` | `RDLD_data/results/p0/perturbed/` (57 PNGs), log `results/p0/logs/p0_verify_perturbations.log` |
| `scripts/p0_prepare_doclaynet.py` | `python scripts/p0_prepare_doclaynet.py` | `RDLD_data/doclaynet/{images/, annotations.json, manifest.csv}`, log `results/p0/logs/p0_prepare_doclaynet.log` |

Input data: DocLayNet v1.2 validation split (`ds4sd/DocLayNet-v1.2`, 7 parquet shards, 6,489 pages,
CDLA-Permissive-1.0), cached in `~/.cache/huggingface`.

## 3. Results (evidence)

**Degradation functions** (`p0_verify_perturbations.log`)

| Check | Result |
|---|---|
| Image-only degradations that run | **8 / 8** at all 3 severities: defocus, vibration, speckle, texture, watermark, ink_holdout, ink_bleeding, illumination |
| Spatial degradations (rotation, keystoning, warping) | Not loadable - need detectron2; excluded by design (see §4) |
| Pairs composable at severity 2 | **28 / 28**, output shape preserved |
| Sampled triples composable | **5 / 5** |
| Slowest function | `apply_texture` ≈ 1.2 s/image (Sept timing) - dominates generation cost |

**Corpus** (`p0_prepare_doclaynet.log`)

| Item | Value |
|---|---|
| Pages | **300** - 50 per category × 6 (financial reports, scientific articles, government tenders, laws & regulations, manuals, patents) |
| Annotated regions | **3,530** (11.8 / page) |
| Image size | 1025 × 1025 RGB PNG, 96 MB total |
| Class counts | Text 1,661 · Section-header 578 · List-item 380 · Page-header 312 · Page-footer 215 · Picture 119 · **Table 102** · Caption 66 · Formula 50 · Footnote 41 · **Title 6** |
| Reproducibility | Re-run on 2026-11-03 produced **byte-identical** `annotations.json` and `manifest.csv` (MD5 `d91ff95f…` / `5d9d1b06…`, see `results/p0/logs/sept_hashes.txt`) |

## 4. Decisions and why

1. **Image-only degradations only.** The 8 retained functions change pixels, not geometry, so the
   original boxes stay valid for every degraded variant - zero annotation cost. The 3 spatial ones
   move boxes; composing them means composing coordinate transforms, a separate problem. Reported as a
   scope boundary, not a limitation hidden.
2. **Degradation functions kept unchanged** (vendored in `third_party/rodla_perturbation/`), so
   single-degradation numbers stay comparable with RoDLA. Composition is the extension.
3. **Stratified sampling across shards.** DocLayNet's validation shards are **ordered by document
   category, not shuffled** (shard 00000 is 100% financial reports). Taking the first N rows would give
   a single-category corpus. The script reads across shards until each category quota of 50 is met.
4. **300 pages, not more.** Cost multiplies by 109 configurations (≈ 32,700 images). Ceiling for a
   balanced sample is 334 × 6 = 2,004 (government tenders is the binding category) if per-class
   results prove too noisy.

## 5. Problems found
- **Title has only 6 instances** - too rare for per-class conclusions. Options for P4: pool with
  Section-header, report with a caveat, or extract more pages.
- **Table is 2.9% of boxes (102)** but is the most downstream-critical class - relevant to RQ2.
- The editable `ocrodeg` install pointed at the old project folder path and broke when the folder was
  moved; fixed by vendoring `ocrodeg` into `third_party/` (no environment change).

## 6. Main takeaway
Secondary data; no human participants. The feasibility pre-test shows the independent
variable (degradation combination) can be manipulated under control: 28/28 pairs compose with
dimensions preserved, so the factorial design is implementable as proposed.

## 7. Hand-off
- To P1 (Alishba): validate the corpus (box bounds, areas, duplicates) and run the EDA.
- To P1 (Khizar): `RDLD_data/doclaynet/` is the clean evaluation set; class ids 1–11 as in `rdld/doclaynet.py`.

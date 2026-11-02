# RDLD: Robustness of Document Layout Detection Under Compound Degradation

**Team:** Khizar Mohammad Hashmi (26I-8009), Alishba Abbas (26I-8032). MS Data Science, FAST-NUCES, Fall 2026.

**Base paper:** RoDLA, CVPR 2024. [DOI](https://doi.org/10.1109/CVPR52733.2024.01473). Details in `docs/base_paper/README.md`.

## Research questions

| RQ | Question |
|---|---|
| RQ1 | Do document-image degradations combine additively, or compound super-additively? `I = ΔmAP(A+B) − [ΔmAP(A) + ΔmAP(B)]` |
| RQ2 | Does mAP predict downstream document-reading quality under degradation? |
| RQ3 | Are clean-condition detector rankings preserved under compound degradation? |

Design: 300 DocLayNet pages (50 from each of 6 categories), with 1 clean, 24 single and 84 pair configurations.
Degradations are image-only, so ground-truth boxes stay valid. The study started with 3 detectors and is being extended to 8.

## Branches

| Branch | Holds | Pushed by |
|---|---|---|
| `main` | Shared scaffold. Both branches are merged here at the end. | Khizar |
| `khizar` | Detection, evaluation, statistics | Khizar |
| `alishba` | Data, perturbation, image quality, EDA, OCR | Alishba |

## Layout

```
rdld/             shared package: paths, DocLayNet schema, third-party loader
scripts/          numbered phase scripts (on the person branches)
third_party/      RoDLA perturbation code (unchanged) and ocrodeg
literature_search/ search scripts, log, records and references
docs/phases/      one folder per phase, one notes file per person
docs/base_paper/  base paper details and links
figures/          figures from the feasibility stage
```

Data and results are not in Git. They live in `../RDLD_data/` (override with `RDLD_DATA`):

```
RDLD_data/doclaynet/     300 clean pages, COCO annotations, manifest
RDLD_data/degraded/      perturbation cache
RDLD_data/predictions/   detector output cache
RDLD_data/results/       metric tables and figures, by phase
```

## Environment

Conda env `rodla_layout` (Python 3.10, torch 2.5.1+cu121). `pip install -r requirements.txt`.
Run scripts from the repo root: `python scripts/<name>.py`.

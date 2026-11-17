# RDLD: Robustness of Document Layout Analysis Under Compound Degradation

**Team:** Khizar Mohammad Hashmi (26I-8009), Alishba Abbas (26I-8032). MS Data Science, FAST-NUCES, Fall 2026.

**Base paper:** RoDLA, CVPR 2024. [DOI](https://doi.org/10.1109/CVPR52733.2024.01473). Details in `docs/base_paper/README.md`.

## What the study does

Layout robustness benchmarks apply one image degradation at a time. Real scans and photos usually carry several at
once. We apply eight image-only RoDLA degradations alone and in all same-severity pairs, at three severities, to two
sets of 300 DocLayNet pages (one from the validation split, one from the official test split, 50 pages from each of 6
categories). That gives 109 conditions per page set: 1 clean, 24 single and 84 pair. Eleven layout detectors from six
architecture families are evaluated on every condition. Degradations are image-only, so the ground-truth boxes stay
valid.

| RQ | Question |
|---|---|
| RQ1 | Do degradations combine additively, or do they compound? For a pair A+B, the interaction compares the pair's mAP loss with an additive null, `I_add = D(A+B) − D(A) − D(B)`, and a multiplicative null, `I_mult = 100 [R(A) R(B) − R(A+B)]` (D = drop, R = retention). |
| RQ2 | Does mAP predict the quality of the text a downstream reader extracts under degradation? |
| RQ3 | Are clean-condition detector rankings preserved under single and compound degradation? |

## Where to start

| If you want to | Read |
|---|---|
| See the results, phase by phase, without running anything | `explanation_guide/` (notebook with stored outputs, and the same as a PDF) |
| Reproduce the study or run it locally | **`docs/PIPELINE.md`**, the full run order, with every command, its outputs and its run time |
| Understand a specific number | `docs/phases/P*/NOTES_khizar.md` and `NOTES_alishba.md`, and `docs/phases/P6_writeup/RESULTS_SYNTHESIS.md` |

## Reproducing the study

Everything needed to rebuild the results is in this repository, except the data and model weights, which the first
step downloads. The steps, in short:

1. Create the environment (below) and install `requirements.txt`.
2. `python scripts/p0_download_data.py --models` downloads the DocLayNet files and every checkpoint, at the exact
   revisions we used.
3. Follow `docs/PIPELINE.md` from step 1. It covers the data, the degraded images, image quality, detection, scoring,
   the three research questions and the checks, for both page sets.

The analysis steps run on a CPU. Detection, image quality and the RF-DETR fine-tuning need a CUDA GPU. On one RTX 3080
Laptop GPU, detection took about 22 hours for the 109 conditions of one page set (the validation replicates added more),
and the RF-DETR fine-tuning about 12 hours. On 18 Nov 2026 the repository was cloned fresh and rebuilt this way, and
the page sets, a sample of the degraded images, the bootstrap resamples, the rescoring of every prediction file and all
analysis tables matched our results byte for byte (`docs/PIPELINE.md`, "Reproduction check from a fresh clone").

## Repository layout

```
rdld/                 shared package: paths, DocLayNet schema, degradations, detectors, evaluation, intervals
scripts/              numbered phase scripts (p0_ ... p7_), run in the order given in docs/PIPELINE.md
third_party/          RoDLA perturbation code (unchanged) and ocrodeg
docs/PIPELINE.md      how to run everything, step by step, and where each paper number comes from
docs/RUNBOOK.md       the short version: download, then build the page sets
docs/phases/          one folder per phase, one notes file per person
docs/base_paper/      base paper details and links
explanation_guide/    results walkthrough: executed notebook and PDF
literature_search/    search scripts, log, records and references
figures/              figures from the feasibility stage
requirements.txt      exact package versions
```

## Data folders

Data and results are not in Git. They sit next to the repository, and every script reads its data folder from the
`RDLD_DATA` environment variable (default `../RDLD_data`):

```
RDLD_data/               validation page set
  doclaynet/             300 clean pages, COCO annotations, manifest
  degraded/              degraded-image cache and the condition lists
  predictions/           detector outputs, one file per detector and condition
  results/               metric tables and figures, by phase
RDLD_data_test/          official test-split page set, same layout
RDLD_data_docboot/       document-bootstrap results for the validation pages
RDLD_data_test_docboot/  document-bootstrap results for the test pages
```

## Environment

Conda env `rodla_layout` (Python 3.10.21, torch 2.5.1+cu121). `requirements.txt` pins the exact versions every result
was produced with; its header gives the CUDA install line for torch and the conda line for Tesseract 5.5.3, which the
OCR step needs. Run scripts from the repository root: `python scripts/<name>.py`.

## Branches

| Branch | Holds | Pushed by |
|---|---|---|
| `main` | The full code: both person branches merged | Khizar |
| `khizar` | Detection, evaluation, statistics | Khizar |
| `alishba` | Data, degradation, image quality, EDA, OCR | Alishba |

Changes are made on the person's branch first and then merged into `main`.

## Data, models and licences

Nothing below is redistributed in this repository. `python scripts/p0_download_data.py --models` downloads the
dataset shards and every checkpoint at the exact revision used for the paper (`docs/PIPELINE.md`, step 0).

| Item | Source | Licence |
|---|---|---|
| DocLayNet v1.2 pages and annotations | `docling-project/DocLayNet-v1.2` (Hugging Face) | CDLA-Permissive-1.0 |
| YOLO26-m, YOLO26-l, YOLOv11-m, YOLOv11-l | `hantian/yolo-doclaynet` | AGPL-3.0 |
| Heron, Heron-101, Egret-m, Egret-xl | `docling-project/docling-layout-*` | Apache-2.0 |
| DETR | `cmarkea/detr-layout-detection` | Apache-2.0 |
| Deformable DETR | `Aryn/deformable-detr-DocLayNet` | Apache-2.0 |
| RF-DETR medium (base for our fine-tuning) | `Roboflow/rf-detr-medium` | Apache-2.0 |
| RoDLA perturbation code (vendored, unchanged) | `third_party/rodla_perturbation` | Apache-2.0, see its `LICENSE` and `NOTICE.md` |
| ocrodeg, needed by RoDLA's `noise.py` (vendored) | `third_party/ocrodeg` | BSD-style (NVIDIA), see its `NOTICE.md` |

**Fine-tuned RF-DETR weights.** Our RF-DETR checkpoint is not in the repository, because of its size. It is rebuilt
with `p7_prepare_train_data.py` and `p7_finetune_rfdetr.py` (`docs/PIPELINE.md`, step 4). The base model is Apache-2.0,
so the checkpoint can be shared, and it is available from the authors on request.

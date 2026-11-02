# Phase Plan

Each phase has a Khizar part and an Alishba part of similar size. The two sides connect through the shared
data folder (`RDLD_data/`), not through code imports: Alishba's side produces images and image-level
measurements, and Khizar's side uses those images and produces detections and detection-level measurements.

| Phase | Khizar (branch `khizar`) | Alishba (branch `alishba`) |
|---|---|---|
| **P0** Feasibility and data | Detector feasibility; 3-detector verification; figures | Perturbation and composition feasibility; stratified DocLayNet extraction |
| **P1** Clean baseline | Detector wrappers, class mapping, clean mAP, label-instability floor | Corpus validation and EDA |
| **P2** Pipeline | Batch inference over the degraded cache; mAP, mPE, mRD evaluator | Composition engine, degraded-image cache, image-quality metrics |
| **P3** Main experiment | Bootstrap, interaction term, RQ1 analysis | Image quality versus accuracy; composition-order check |
| **P4** Rankings and per-region | Ranking stability, seeds, significance tests | Per-class AP; structural error rates (LED taxonomy) |
| **P5** Downstream link | Weighted mAP and weight sensitivity; mAP versus downstream | OCR pipeline; WER and text measures |
| **P6** Results | Results synthesis and research paper | Results synthesis and research paper |
| **P7** Detector extension | Added detector wrappers; RF-DETR fine-tuning | RF-DETR training data with leakage checks |

## Where notes go

`docs/phases/P<n>_<name>/NOTES_khizar.md` on branch `khizar`, and
`docs/phases/P<n>_<name>/NOTES_alishba.md` on branch `alishba`. Use `NOTE_TEMPLATE.md`.
Numbers in notes must come from a script output file under `RDLD_data/results/`, with the path cited.

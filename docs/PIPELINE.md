# RDLD pipeline: how to run it, in order

What each script does, the order to run them in, what hardware each step needs, and where each result ends up.

---

## 1. The pipeline in one picture

```
 DocLayNet parquet shards (Hugging Face cache)
        │
        ▼
 1. DATA          p0_prepare_doclaynet  ─►  300 clean pages + COCO boxes
        │         p1_validate_corpus, p1_eda
        ▼
 2. DEGRADE       p2_generate_cache     ─►  109 conditions × 300 pages (the main design)
        │         p3_order_check        ─►  28 pairs in reversed order
        │         p4_seed_replicate     ─►  36 severity-2 conditions with a second seed
        ▼
 3. IMAGE DAMAGE  p2_image_quality      ─►  PSNR / MS-SSIM / CW-SSIM per condition (mPE)
        │
        ▼
 4. DETECT        p1_clean_baseline     ─►  clean mAP of each detector (own inference pass)
        │         p2_run_inference      ─►  11 detectors × every condition → prediction files
        ▼
 5. SCORE         p2_evaluate           ─►  mAP per detector × condition, RoDLA mRD
        │         p3_bootstrap          ─►  1,000 paired page resamples per detector × condition
        │         p3_jackknife          ─►  leave-one-page-out tables (for BCa intervals)
        ▼
 6. ANSWER THE RQs
        RQ1  compounding        p3_rq1_interaction, p4_seed_stability,
                                p3_order_check --analyse, p3_iq_vs_accuracy, p4_per_class
        RQ3  rankings           p4_rq3_rankings, p4_rq3_robustness
        RQ2  downstream         p4_structural_errors → p5_ocr_downstream → p5_rq2_metric_validity
        ▼
 7. CHECKS        p7_select_vs_test, p7_title_sensitivity, p7_test_split_leakage,
                  official test split repeat, document bootstrap, score cut-off 0.3 / 0.7
```

RF-DETR has no released layout checkpoint, so it is fine-tuned once before step 4: `p7_prepare_train_data`, then
`p7_finetune_rfdetr`. Only the RF-DETR checkpoint comes from that side branch.

**The design:**
- **109 conditions:** clean, 24 singles (8 degradations × 3 severities) and 84 pairs (28 pairs × 3 severities).
- **2 extra condition sets** for robustness checks:
  - `configs_order.csv`: the 28 pairs in reversed order.
  - `configs_rep.csv`: 36 severity-2 conditions with a second seed.

All three lists are written to `degraded/`.

---

## 2. Setup

### 2.1 Environment
```
conda create -n rodla_layout python=3.10
conda activate rodla_layout
pip install -r requirements.txt      # torch 2.5.1 + CUDA 12.1 build
```
Tesseract has to be installed and on PATH for the OCR step (`p5_ocr_downstream`).

### 2.2 Folders
Data and results are not in Git. They sit next to the repository:

```
RDLD_Project/            this repository
RDLD_data/               validation page set: doclaynet/, degraded/, predictions/, results/
RDLD_data_test/          official test-split page set, same layout
RDLD_data_docboot/       document-bootstrap results for the validation pages
RDLD_data_test_docboot/  same for the test pages
```

Every script reads its data folder from `RDLD_DATA`. When it is not set, the default is `../RDLD_data`. Switching page
sets only means changing that one variable.

### 2.3 How to run any script
Run each script from the repository root:

```
python scripts/<name>.py
```

Options are set with environment variables. In Bash:

```
RDLD_DATA=../RDLD_data_test WORKERS=12 python scripts/p2_evaluate.py
```

In PowerShell:

```
$env:RDLD_DATA = "..\RDLD_data_test"; $env:WORKERS = "12"; python scripts\p2_evaluate.py
```

PowerShell keeps `$env:` variables for the rest of the session. Clear them afterwards (`Remove-Item Env:RDLD_DATA`) so
a later script does not pick up the wrong data folder.

| Variable | Used by | Meaning |
|---|---|---|
| `RDLD_DATA` | all | data folder (page set) |
| `SPLIT` | p0_prepare_doclaynet | `validation` (default) or `test` |
| `MODELS` | inference, baseline, bootstrap, jackknife | comma list, e.g. `heron,yolo11m`. Default: all 11 |
| `CONFIGS` | inference, bootstrap, jackknife | which condition list; default `degraded/configs.csv` |
| `WORKERS` | CPU-heavy scripts | parallel processes |
| `N_BOOT` | bootstrap, baseline | resamples (default 1000) |
| `BOOT_SCHEME` | p3_bootstrap | `page` (default) or `doc` (run in its own data folder) |
| `TAU` | structural, OCR | score cut-off (default 0.5) |
| `N_PAGES` | cache, inference | pilot on the first N pages only (writes to `predictions_pilot/`) |

The 11 detector names are `yolo` (YOLO26-m), `yolo26l`, `yolo11m`, `yolo11l`, `heron`, `heron101`, `egret`, `egretxl`,
`detr`, `ddetr` and `rfdetr`.

---

## 3. Full run from scratch, in order

- **Resumable steps:** inference, the degraded cache and image quality skip work that already exists.
- **Overwriting steps:** the analysis scripts rewrite their own output folder.

Times were measured on one RTX 3080 Laptop GPU with 16 logical CPU cores.

### Step 1: Data (CPU)
| # | Command | Writes | Time |
|---|---|---|---|
| 1.1 | `python scripts/p0_verify_perturbations.py` | `results/p0/perturbed/` (57 sample images): checks that the degradations compose | 20 s |
| 1.2 | `python scripts/p0_prepare_doclaynet.py` | `doclaynet/images/`, `annotations.json`, `manifest.csv` (300 pages, 50 per category) | a few min |
| 1.3 | `python scripts/p1_validate_corpus.py` | `results/p1/validation/` (box and image checks) | < 1 min |
| 1.4 | `python scripts/p1_eda.py` | `results/p1/eda/` (box and page feature tables) | < 1 min |

For the test page set, run 1.2 with `RDLD_DATA=../RDLD_data_test` and `SPLIT=test`, then 1.3 and 1.4 with the same
`RDLD_DATA`. Both page sets rebuild byte for byte.

### Step 2: Degraded images (CPU)
| # | Command | Writes | Time |
|---|---|---|---|
| 2.1 | `python scripts/p2_generate_cache.py --check` | nothing; checks that the cache matches direct generation and is repeatable | 1 min |
| 2.2 | `python scripts/p2_generate_cache.py` | `degraded/configs.csv` + 109 folders × 300 PNG | 20–25 min with 14 workers |
| 2.3 | `python scripts/p3_order_check.py` | `degraded/configs_order.csv` + 28 reversed pairs | about a quarter of 2.2 |
| 2.4 | `python scripts/p4_seed_replicate.py` | `degraded/configs_rep.csv` + 36 second-seed conditions | about a third of 2.2 |

The test page set needs only 2.2.

### Step 3: Image damage (GPU)
| # | Command | Writes | Time |
|---|---|---|---|
| 3.1 | `python scripts/p2_image_quality.py --sanity` | metric sanity checks | 1 min |
| 3.2 | `python scripts/p2_image_quality.py` | `results/p2/image_quality/iq_per_config.csv` (mPE per condition) | several hours; resumes per condition |

Step 3 does not depend on any detector, so it can run alongside step 4, although both then share the GPU.

### Step 4: Detection (GPU)
| # | Command | Writes | Time |
|---|---|---|---|
| 4.1 | `python scripts/p0_detector_gate.py` | `results/p0/detections/`: one page, checks that the detectors load | 20 s |
| 4.2 | `python scripts/p1_clean_baseline.py` | `results/p1/baseline*/` (clean mAP with CI, from its own inference pass) | about 20 min |
| 4.3 | `python scripts/p2_run_inference.py` | `predictions/<model>/<condition>.json` for every condition in `configs.csv` | about 22 h for all 11 |
| 4.4 | `CONFIGS=../RDLD_data/degraded/configs_order.csv`, then 4.3 | reversed-order predictions | about 1 h per 3 models |
| 4.5 | `CONFIGS=../RDLD_data/degraded/configs_rep.csv`, then 4.3 | second-seed predictions | about 1.3 h per 3 models |

RF-DETR fine-tuning, before 4.3:
1. `python scripts/p7_prepare_train_data.py` builds 11,526 training pages and 600 selection pages from the DocLayNet
   train split and checks for leakage. About 1 h.
2. `python scripts/p7_finetune_rfdetr.py` trains for 12 epochs (about 12.4 h) and writes `finetune/rfdetr/best/`.

Inference is close to deterministic: repeating it gives identical predictions on a 48-page check, and the clean pass in
4.2 agrees with 4.3 within 0.04 mAP points for every detector. All reported numbers come from the 4.3 predictions.

### Step 5: Scoring (CPU)
| # | Command | Writes | Time |
|---|---|---|---|
| 5.1 | `python scripts/p2_evaluate.py` | `results/p2/evaluation/metrics_long.csv`, `mrd.csv` | about 20 min |
| 5.2 | `python scripts/p3_bootstrap.py`, once per condition list (`CONFIGS=` main, order, rep) | `results/p3/boot/<model>/<condition>.npy` | about 1.5 h for all 11, main list |
| 5.3 | `python scripts/p3_jackknife.py` (main and rep lists) | `results/p3/jack/` (only for BCa intervals) | about 30 min |

If image quality (step 3) finished after 5.1, run `python scripts/p2_evaluate.py --mrd-only` to add the mRD column.

Each bootstrap file has shape (1001, 11): row 0 is the full sample and rows 1–1000 are resamples; the columns are
per-class AP, and mAP is their mean.

### Step 6: The research questions
Every script here reads the tables from step 5 and runs in seconds to minutes. Order matters only where an arrow is
shown.

| RQ | Command | Main output |
|---|---|---|
| RQ1 | `p3_rq1_interaction.py` | `results/p3/rq1/rq1_interactions.csv`, `rq1_summary.csv`, heatmaps |
| RQ1 | `p4_seed_stability.py` | `results/p4/seeds/` (are interactions stable across degradation seeds) |
| RQ1 | `p3_order_check.py --analyse` | `results/p3/order/order_effect.csv` (about 1 h) |
| RQ1 | `p3_iq_vs_accuracy.py` | `results/p3/iq_vs_accuracy/` (does pixel damage explain detection damage) |
| RQ1 | `p4_per_class.py` | `results/p4/per_class/` (which region types break) |
| RQ3 | `p4_rq3_rankings.py` | `results/p4/rq3/rank_agreement.csv`, `benchmark_rankings.csv` |
| RQ3 | `p4_rq3_robustness.py` | `results/p4/rq3_robustness/tau_gap.csv`, `pair_effect_at_equal_damage.csv`, `benchmark_agreement.csv` |
| RQ2 | `p4_structural_errors.py` → | `results/p4/structural/structural_rates.csv` (merge, split, miss) |
| RQ2 | `p5_ocr_downstream.py` → | `results/p5/downstream/` (WER, text recall, table recall; 30–60 min) |
| RQ2 | `p5_rq2_metric_validity.py` | `results/p5/rq2/` (does mAP predict reading quality) |

### Step 7: Checks
| Check | How |
|---|---|
| Released detectors not tuned on our pages | `p7_select_vs_test.py` |
| Title has only 6 boxes on the validation pages | `p7_title_sensitivity.py` |
| Test pages share no document with training or validation | `RDLD_DATA=../RDLD_data_test`, then `p7_test_split_leakage.py` |
| Whole study on the official test split | `RDLD_DATA=../RDLD_data_test`, then repeat 1.2–1.4, 2.2, 3.2, 4.3, 5.1–5.3 and step 6 (no order or seed sets) |
| Document bootstrap | In a separate data folder whose `doclaynet`, `degraded` and `predictions` link to the real ones: `BOOT_SCHEME=doc` on 5.2, then step 6. The page-bootstrap results stay untouched |
| RQ1 at equal predicted damage | `p7_rq1_damage_bands.py`, once per data folder (both page sets, page and document bootstrap) |
| Score cut-off | `TAU=0.3` and `TAU=0.7` on `p4_structural_errors`, `p5_ocr_downstream`, `p5_rq2_metric_validity`. These overwrite the 0.5 folders, so copy those first; the results go to `*_tau0.3` and `*_tau0.7` |

---

## 4. Where each paper number comes from

Validation page set: `RDLD_data/results`; test page set: `RDLD_data_test/results`.

| Paper claim | File |
|---|---|
| Clean mAP per detector, with 95% CI | `p2/evaluation/metrics_long.csv` (clean rows); intervals from `p3/boot/<model>/clean.npy` |
| Pairs compound; multiplicative rule predicts best | `p3/rq1/rq1_summary.csv`, `rq1_prediction_from_singles.csv` |
| Rankings break down more under pairs (τ gap ≈ 0.27) | `p4/rq3_robustness/tau_gap.csv` (page and BCa CIs); document CIs in the `*_docboot` folders |
| No consistent pair effect at equal damage | `p4/rq3_robustness/pair_effect_at_equal_damage.csv` |
| mRD and retention disagree | `p2/evaluation/mrd.csv` vs `p4/rq3/benchmark_rankings.csv` |
| YOLO size vs generation | `p4/rq3/benchmark_rankings.csv`, `p4/structural/structural_rates.csv` |
| mAP vs reading quality | `p5/rq2/metric_vs_downstream_correlation.csv`, `ranking_agreement.csv`, `within_condition_correlation.csv` |
| Score cut-off sensitivity | `p4/structural_tau0.3`, `_tau0.7`; `p5/rq2_tau0.3`, `_tau0.7` |

The written account of every number is in the phase notes, `docs/phases/P*/NOTES_khizar.md` and `NOTES_alishba.md`.
The cross-RQ summary is `docs/phases/P6_writeup/RESULTS_SYNTHESIS.md`.

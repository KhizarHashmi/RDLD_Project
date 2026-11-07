# P7 - Detector Extension - Alishba (Workstream A)

**Dates:** 2026-11-04
**Status:** training data complete. The fine-tuning itself is in Workstream B.

## 1. Goal
RF-DETR has no released layout checkpoint, so it has to be fine-tuned. This phase builds its training
and checkpoint-selection data, and proves that none of the 300 test pages leak into either set.

## 2. What was done
`python scripts/p7_prepare_train_data.py`, log at `RDLD_data/results/p7/logs/p7_prepare_train_data.log`.

- **Train:** all annotated pages from 12 DocLayNet train shards (every 6th of 72). The shards are sorted by
  document category, so taking every 6th covers all six categories.
- **Select:** 100 validation pages per category, leaving out the 300 test pages.
- Membership and leakage are checked from metadata first, before any image is written.

## 3. Results

| Leakage check | Count |
|---|---|
| Test pages in train | 0 |
| Test pages in select | 0 |
| Test documents in train | 0 |
| Pages shared by train and select | 0 |

| Category | Train | Select |
|---|---:|---:|
| financial_reports | 3,838 | 100 |
| manuals | 2,882 | 100 |
| scientific_articles | 1,928 | 100 |
| government_tenders | 964 | 100 |
| patents | 963 | 100 |
| laws_and_regulations | 951 | 100 |
| **Total pages** | **11,526** | **600** |
| **Boxes** | **161,012** | **8,409** |

## 4. Decisions and why
1. **Documents are checked as well as pages.** Pages from one document look alike, so a test document
   appearing in training would also be leakage.
2. **Selection pages come from validation.** The test pages stay unseen, including during checkpoint selection.

## 5. Hand-off
Workstream B trains on `RDLD_data/finetune/train` and selects on `RDLD_data/finetune/select`.

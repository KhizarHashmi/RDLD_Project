# P7 - Detector Extension - Alishba (Workstream A)

**Dates:** 2026-11-04 (test-split pages added 2026-11-10)
**Status:** training data and test-split pages complete. The fine-tuning and the test-split runs are in Workstream B.

## 1. Goal
1. RF-DETR has no released layout checkpoint, so it has to be fine-tuned. Build its training and checkpoint-selection
   data, and prove that none of the 300 validation pages used in the study leak into either set.
2. Build a second page set from the official DocLayNet test split, so the study can be repeated on pages no released
   detector could have used to pick a checkpoint.

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

### 3.1 Test-split pages
`scripts/p0_prepare_doclaynet.py` gained a `SPLIT` option (default `validation`, so the original pages are unchanged):

    RDLD_DATA=<test data root> SPLIT=test python scripts/p0_prepare_doclaynet.py

It reads the 6 test shards (4,999 pages) with the same stratified rule, 50 pages per category. Result: 300 pages,
4,351 boxes, from 44 documents (the validation pages come from 23), with 19 Title boxes (validation: 6). Rerunning the
script rebuilds the pages, `annotations.json` and `manifest.csv` byte for byte, and with no `SPLIT` it still rebuilds
the validation pages byte for byte.

Workstream B's leakage check (`scripts/p7_test_split_leakage.py`) finds no test document in the validation split or in
RF-DETR's training shards.

## 4. Decisions and why
1. **Documents are checked as well as pages.** Pages from one document look alike, so a test document
   appearing in training would also be leakage.
2. **Selection pages come from validation.** The test pages stay unseen, including during checkpoint selection.
3. **Same extraction rule for the test split.** Only the split changes, so differences between the page sets come from the
   pages, not from how they were sampled.

## 5. Hand-off
Workstream B trains on `RDLD_data/finetune/train` and selects on `RDLD_data/finetune/select`, and runs all eleven
detectors on the test pages in `RDLD_data_test`.

# Literature search

Scripts and records for the documented search run on 6 November 2026.

| File | What it is |
|---|---|
| `run_search.py` | Runs the four Boolean queries against OpenAlex, Semantic Scholar, arXiv and DBLP. Writes the two files below. |
| `search_log.csv` | Date, source, exact query, filters, records reported and records saved. Failed sources are recorded as failed. |
| `raw_records.csv` | Every record returned (709). |
| `dedupe_records.py` | Removes exact title duplicates. |
| `dedup_records.csv` | Unique records after duplicate removal (323). |
| `verify.py` | Looks up known papers on Crossref and OpenAlex by title. |
| `verification.csv` | Results of that lookup. Low-similarity rows are wrong matches and were not used. |
| `references.bib` | The 34 selected papers, each labelled Published or Preprint. |

DBLP blocked automated access and Semantic Scholar rate-limited Q2, so those cells have no records.
This was a targeted search of open indexes. It is not a full systematic review.

```
python run_search.py
python dedupe_records.py
python verify.py
```

Re-running will give different counts because the indexes change over time.

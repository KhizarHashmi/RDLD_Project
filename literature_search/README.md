# Literature search

Scripts and records for the documented searches run on 6 and 7 November 2026.

## First round, 6 November

| File | What it is |
|---|---|
| `run_search.py` | Runs the four Boolean queries against OpenAlex, Semantic Scholar, arXiv and DBLP. Writes the two files below. |
| `search_log.csv` | Date, source, exact query, filters, records reported and records saved. Failed sources are recorded as failed. |
| `raw_records.csv` | Every record returned (709). |
| `dedupe_records.py` | Removes exact title duplicates. |
| `dedup_records.csv` | Unique records after duplicate removal (323). |
| `verify.py` | Looks up known papers on Crossref and OpenAlex by title. |
| `verification.csv` | Results of that lookup. Low-similarity rows are wrong matches and were not used. |

DBLP blocked automated access and Semantic Scholar rate-limited Q2, so those cells have no records.
This was a targeted search of open indexes. It is not a full systematic review.

```
python run_search.py
python dedupe_records.py
python verify.py
```

Re-running will give different counts because the indexes change over time.

## Second round, 7 November

Adds databases and a new query, Q5, aimed at the compound-degradation gap.

| File | What it is |
|---|---|
| `run_browser_search.py` | Runs the queries in a visible browser through normal search pages, with long pauses. It completed IEEE Xplore. On SpringerLink and ScienceDirect it stopped at a CAPTCHA and did not try to get past it; we then exported SpringerLink by hand, and ScienceDirect asked for an institutional login we do not have. |
| `run_publisher_search.py` | ACM and Elsevier records from Crossref and OpenAlex using publisher filters, plus Q5 on OpenAlex, Semantic Scholar and arXiv. These are not searches of the ACM Digital Library or ScienceDirect. |
| `search_log_browser.csv`, `search_log_publisher.csv` | Date, source, exact string, filters, counts and notes. |
| `source_counts.csv` | Results reported and records saved per source and query, including the exports done by hand. |
| `exports/` | Raw exports. `ieee_q*.csv` from the script, `publisher_records.csv` from Crossref and OpenAlex, `SearchResults*.csv` from SpringerLink by hand. |
| `merge_all_sources.py` | Merges everything with the first round and removes exact title duplicates. Needs the Google Scholar pages, which stay local. |
| `enrich_and_scan.py` | Adds abstracts from OpenAlex by DOI, then scans for overlapping work. |
| `all_sources_dedup.csv` | 1,750 unique records, 648 with abstracts. |
| `overlap_candidates.csv` | Keyword hits to read by hand. A hit is not a finding. |

SpringerLink files by query: Q1 `SearchResults (1).csv`, Q2 `SearchResults.csv`, Q3 `SearchResults (2).csv`, Q4 `SearchResults (3).csv`, Q5 `SearchResults (4).csv`. Exports had fewer rows than the pages reported.

Google Scholar was searched by hand for discovery only: the first 10 results for Q1 to Q4 and all 35 for Q5. Its saved pages are not in this repository. Its records are in `all_sources_dedup.csv`.

ScienceDirect, the ACM Digital Library, Scopus and Web of Science were not searched directly because we do not have access.

`references.bib` holds the 37 selected papers, each labelled Published or Preprint.

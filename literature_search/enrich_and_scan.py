"""Adds abstracts from OpenAlex by DOI to records that have none, then rescans for overlapping work.

Outputs: all_sources_dedup.csv (updated with abstracts), overlap_candidates.csv
"""
import csv
import json
import re
import time
import urllib.parse as U
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
csv.field_size_limit(10 ** 7)
UA = {"User-Agent": "generic-literature-search-client/1.0"}
F = HERE / "all_sources_dedup.csv"

rows = list(csv.DictReader(open(F, encoding="utf-8", newline="")))
need = [r for r in rows if not r["abstract"].strip() and r["doi"].strip()]
print("records", len(rows), "no abstract with DOI", len(need), flush=True)
found = {}
for i in range(0, len(need), 50):
    chunk = [r["doi"].strip().lower() for r in need[i:i + 50]]
    u = "https://api.openalex.org/works?per-page=50&select=doi,abstract_inverted_index&filter=" + U.quote("doi:" + "|".join(chunk))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
                d = json.loads(r.read().decode("utf-8", "ignore"))
            break
        except Exception as e:
            print("retry", i, str(e)[:80], flush=True)
            d = {"results": []}
            time.sleep(5)
    for w in d["results"]:
        ab = w.get("abstract_inverted_index") or {}
        if ab:
            found[(w.get("doi") or "").lower().replace("https://doi.org/", "")] = " ".join(k for _, k in sorted((p, k) for k, v in ab.items() for p in v))
    time.sleep(0.2)
for r in rows:
    if not r["abstract"].strip() and r["doi"].strip().lower() in found:
        r["abstract"] = found[r["doi"].strip().lower()][:1500]
print("abstracts added", len(found), "now with abstract", sum(bool(r["abstract"].strip()) for r in rows), "of", len(rows), flush=True)
with open(F, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

DEG = r"(degrad\w*|corrupt\w*|distort\w*|perturb\w*|noise|artifact\w*|artefact\w*)"
A = re.compile(rf"(compound\w*|combin\w*|multiple|mixed|simultaneous\w*|composite|concurrent\w*|co-occurr\w*|interact\w*)(\s+\w+){{0,3}}\s+{DEG}|{DEG}(\s+\w+){{0,3}}\s+(combin\w*|compound\w*|interact\w*|co-occurr\w*)", re.I)
D = re.compile(r"document layout|layout analysis|layout detection|page segmentation|layout parsing|document parsing", re.I)
R3 = re.compile(r"kendall|rank\w*\s+(\w+\s+){0,2}(stab|preserv|chang|revers|flip|consisten)|robustness rank", re.I)
R2 = re.compile(r"(\bmAP\b|average precision|geometric metric|overlap metric|\bIoU\b).{0,80}(OCR|downstream|reading|text extraction|character error|word error)|(OCR|downstream|text extraction|character error|word error).{0,80}(\bmAP\b|average precision|\bIoU\b)", re.I | re.S)
ROB = re.compile(r"robust|degrad|corrupt|perturb|benchmark", re.I)
cand = []
for r in rows:
    t = f"{r['title']} {r['abstract']}"
    if not D.search(t):
        continue
    for label, pat in (("RQ1", A), ("RQ2", R2), ("RQ3", R3)):
        m = pat.search(t)
        if m and (label != "RQ2" or ROB.search(t)):
            i = m.start()
            cand.append({"rq": label, "title": r["title"], "year": r["year"], "venue": r["venue"][:80], "doi": r["doi"], "sources": r["sources"], "context": t[max(0, i - 110):i + 150]})
with open(HERE / "overlap_candidates.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["rq", "title", "year", "venue", "doi", "sources", "context"])
    w.writeheader()
    w.writerows(cand)
print("candidates", Counter(c["rq"] for c in cand))
for c in sorted(cand, key=lambda c: c["rq"]):
    print(f"[{c['rq']}] {c['year']} {c['title'][:105]} | {c['venue'][:28]} | {c['doi']}\n    ...{c['context']}")
print("DONE")

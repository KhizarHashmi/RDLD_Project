"""Merges every search export with the earlier open-index records and scans for overlapping work.

Outputs: all_sources_dedup.csv, source_counts.csv, overlap_candidates.csv
"""
import csv
import html
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EX = HERE / "exports"
csv.field_size_limit(10 ** 7)

SPRINGER_TOTALS = {"q1": 1044, "q2": 266, "q3": 626, "q4": 304, "q5": 7}
SCHOLAR_FILES = {"q1": "robustness OR degradation", "q2": "metric OR _average precision_", "q3": "_common corruptions_",
                 "q4": "transformer OR DETR", "q5": "_compound degradation_"}


def norm(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def clean(t):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", t or "")).split())


def read_csv(p):
    with open(p, encoding="utf-8", errors="ignore", newline="") as f:
        return list(csv.DictReader(f))


def springer_files():
    """Each export is matched to the query whose reported total equals its row count."""
    out, used = {}, set()
    files = sorted(EX.glob("SearchResults*.csv"))
    sized = {p: len(read_csv(p)) for p in files}
    for q, total in SPRINGER_TOTALS.items():
        target = min(total, 1000)
        best = min((p for p in files if p not in used), key=lambda p: abs(sized[p] - target))
        out[q] = best
        used.add(best)
        print("springer", q, "reported", total, "file", best.name, "rows", sized[best])
    return out


def scholar(path):
    h = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"(?:About )?([\d,]+) results?", clean(h[h.find('id="gs_ab_md"'):][:2000]))
    total = int(m.group(1).replace(",", "")) if m else None
    rows = []
    for block in h.split('class="gs_ri"')[1:]:
        t = re.search(r'class="gs_rt"[^>]*>(.*?)</h3>', block, re.S)
        a = re.search(r'class="gs_a"[^>]*>(.*?)</div>', block, re.S)
        s = re.search(r'class="gs_rs"[^>]*>(.*?)</div>', block, re.S)
        title = re.sub(r"^\[[A-Z]+\]\s*", "", clean(t.group(1))) if t else ""
        info = clean(a.group(1)) if a else ""
        y = re.search(r"\b(19|20)\d{2}\b", info)
        rows.append({"title": title, "year": y.group(0) if y else "", "venue": info, "doi": "", "abstract": clean(s.group(1)) if s else ""})
    return total, rows


def main():
    recs, counts = [], []

    def add(source, q, rows):
        for r in rows:
            if r.get("title"):
                recs.append({"source": source, "query": q, **{k: r.get(k, "") for k in ("title", "year", "venue", "doi", "abstract")}})

    for r in read_csv(HERE / "dedup_records.csv"):
        recs.append({"source": r["src"], "query": r["q"], "title": r["title"], "year": r["year"], "venue": r["venue"], "doi": r["doi"], "abstract": r["abstract"]})
    n_old = len(recs)

    for q in ("q1", "q2", "q3", "q4"):
        rows = read_csv(EX / f"ieee_{q}.csv")
        add("IEEE Xplore", q, [{"title": clean(r["title"]), "year": (re.search(r"Year: (\d{4})", r["info"]) or [None, ""])[1],
                                "venue": r["info"], "abstract": r["abstract"]} for r in rows])
        counts.append({"source": "IEEE Xplore", "query": q, "reported": len(rows), "saved": len(rows)})
    counts.append({"source": "IEEE Xplore", "query": "q5", "reported": 0, "saved": 0})

    for q, p in springer_files().items():
        rows = read_csv(p)
        add("SpringerLink", q, [{"title": clean(r["Item Title"]), "year": r["Publication Year"], "venue": clean(r["Publication Title"]), "doi": r["Item DOI"].lower()} for r in rows])
        counts.append({"source": "SpringerLink", "query": q, "reported": SPRINGER_TOTALS[q], "saved": len(rows)})

    pages = list(EX.glob("*Google Scholar.html"))
    for q, key in SCHOLAR_FILES.items():
        p = [x for x in pages if key in x.name]
        if len(p) != (4 if q == "q5" else 1):   # q5 has all 4 result pages saved, the others page 1 only
            raise SystemExit(f"Scholar file for {q}: {[x.name for x in p]}")
        parsed = [scholar(x) for x in p]
        total = max(t for t, _ in parsed if t is not None)
        rows = [r for _, page in parsed for r in page]
        add("Google Scholar", q, rows)
        counts.append({"source": "Google Scholar", "query": q, "reported": total, "saved": len(rows)})

    pub = read_csv(EX / "publisher_records.csv")
    for r in pub:
        recs.append({"source": f"{r['source']} ({r['publisher']})", "query": r["query_id"], "title": r["title"], "year": r["year"], "venue": r["venue"], "doi": r["doi"], "abstract": r["abstract"]})

    merged, sources = {}, defaultdict(set)
    for r in recs:
        k = norm(r["title"])
        if not k:
            continue
        sources[k].add(r["source"])
        if k not in merged or len(r["abstract"]) > len(merged[k]["abstract"]):
            merged[k] = r
    rows = [{**v, "sources": "; ".join(sorted(sources[k]))} for k, v in merged.items()]
    rows_out = [{k: r[k] for k in ("title", "year", "venue", "doi", "sources", "abstract")} for r in rows]
    with open(HERE / "all_sources_dedup.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
        w.writeheader()
        w.writerows(rows_out)
    with open(HERE / "source_counts.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(counts[0]))
        w.writeheader()
        w.writerows(counts)

    print("\nsaved per source:", dict(Counter(r["source"] for r in recs)))
    print("earlier records", n_old, "all before dedup", len(recs), "unique", len(rows))
    print("years 2019+:", sum(str(r["year"])[:4].isdigit() and int(str(r["year"])[:4]) >= 2019 for r in rows))

    DEG = r"(degrad\w*|corrupt\w*|distort\w*|perturb\w*|noise|artifact\w*|artefact\w*)"
    A = re.compile(rf"(compound\w*|combin\w*|multiple|mixed|simultaneous\w*|composite|concurrent\w*|co-occurr\w*|interact\w*)(\s+\w+){{0,3}}\s+{DEG}|{DEG}(\s+\w+){{0,3}}\s+(combin\w*|compound\w*|interact\w*|co-occurr\w*)", re.I)
    D = re.compile(r"document layout|layout analysis|layout detection|page segmentation|layout parsing|document parsing", re.I)
    R3 = re.compile(r"kendall|rank\w*\s+(\w+\s+){0,2}(stab|preserv|chang|revers|flip|consisten)|robustness rank", re.I)
    R2 = re.compile(r"(\bmAP\b|average precision|geometric metric|overlap metric|\bIoU\b).{0,80}(OCR|downstream|reading|text extraction|character error|word error)|(OCR|downstream|text extraction|character error|word error).{0,80}(\bmAP\b|average precision|\bIoU\b)", re.I | re.S)
    cand = []
    for r in rows:
        t = f"{r['title']} {r['abstract']}"
        if not D.search(t):
            continue
        for label, pat in (("RQ1", A), ("RQ2", R2), ("RQ3", R3)):
            m = pat.search(t)
            if m:
                i = m.start()
                cand.append({"rq": label, "title": r["title"], "year": r["year"], "venue": r["venue"][:80], "doi": r["doi"], "sources": r["sources"], "context": t[max(0, i - 90):i + 130]})
    with open(HERE / "overlap_candidates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["rq", "title", "year", "venue", "doi", "sources", "context"])
        w.writeheader()
        w.writerows(cand)
    print("\noverlap candidates", Counter(c["rq"] for c in cand))
    for c in sorted(cand, key=lambda c: c["rq"]):
        print(f"[{c['rq']}] {c['year']} {c['title'][:100]} | {c['sources'][:40]}\n     ...{c['context']}")
    print("\ncounts:")
    for c in counts:
        print(c)


if __name__ == "__main__":
    main()

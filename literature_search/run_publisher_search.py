"""Open-index search for ACM and Elsevier records, plus the new Q5 on the open indexes.

These are not searches of the ACM Digital Library or ScienceDirect. They are Crossref and OpenAlex
records filtered by publisher, and must be reported that way.

Crossref has no Boolean search, so it returns its top 100 by relevance for the key terms and the
concept logic of each query is then applied locally to the title and abstract. Many Crossref records
have no abstract, so a record is kept if the first concept matches; `all_concepts_match` shows whether
every concept matched.

Outputs: exports/publisher_records.csv, exports/q5_open_records.csv, search_log_publisher.csv
"""
import csv
import datetime
import json
import re
import time
import urllib.parse as U
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "exports"
OUT.mkdir(exist_ok=True)
UA = {"User-Agent": "generic-literature-search-client/1.0"}
TODAY = datetime.date.today().isoformat()
FROM = 2019

Q = {
    "q1": '("document layout analysis" OR "document layout detection" OR "layout analysis" OR "page segmentation") AND (robust* OR degrad* OR perturb* OR corrupt* OR noise)',
    "q2": '("document layout analysis" OR "document layout detection") AND (metric* OR "average precision" OR evaluation) AND (OCR OR "reading order" OR downstream OR "error rate" OR RAG OR "document parsing")',
    "q3": '("common corruptions" OR "corruption robustness" OR "corrupted images") AND (compound OR combined OR mixture OR multiple OR interaction) AND ("object detection" OR detector*)',
    "q4": '("document layout analysis" OR "document layout detection") AND (transformer OR DETR OR YOLO OR DocLayNet OR "object detection")',
    "q5": '("document layout" OR "layout analysis") AND ("compound degradation" OR "combined degradation" OR "multiple degradations" OR "mixed degradation" OR "compound corruption")',
}
TERMS = {
    "q1": "document layout analysis page segmentation robustness degradation perturbation corruption noise",
    "q2": "document layout analysis evaluation metric average precision OCR reading order downstream document parsing",
    "q3": "common corruptions corruption robustness object detection compound combined interaction",
    "q4": "document layout analysis detection transformer DETR YOLO DocLayNet object detection",
    "q5": "document layout analysis compound degradation combined degradation multiple degradations mixed degradation",
}
CONCEPTS = {
    "q1": [r"document layout (analysis|detection)|layout analysis|page segmentation", r"robust|degrad|perturb|corrupt|noise"],
    "q2": [r"document layout (analysis|detection)", r"metric|average precision|evaluation", r"\bOCR\b|reading order|downstream|error rate|\bRAG\b|document parsing"],
    "q3": [r"common corruptions|corruption robustness|corrupted images", r"compound|combined|mixture|multiple|interaction", r"object detection|detector"],
    "q4": [r"document layout (analysis|detection)", r"transformer|DETR|YOLO|DocLayNet|object detection"],
    "q5": [r"document layout|layout analysis", r"compound degradation|combined degradation|multiple degradations|mixed degradation|compound corruption"],
}
PUBLISHERS = {"ACM": ("10.1145", "Association for Computing Machinery"), "Elsevier": ("10.1016", "Elsevier")}
LOG, RECS, Q5 = [], [], []


def get(url, tries=4):
    err = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            err = e
            time.sleep(5 * (i + 1))
    raise RuntimeError(f"{url[:90]}: {err}")


def log(source, q, string, filters, total, saved):
    LOG.append({"date": TODAY, "source": source, "query_id": q, "string": string, "filters": filters,
                "total_reported": total, "records_saved": saved})
    print(source, q, "total", total, "saved", saved, flush=True)


def inverted(ab):
    return " ".join(k for _, k in sorted((p, k) for k, v in (ab or {}).items() for p in v))


def crossref(q, prefix, pub):
    s = TERMS[q]
    u = ("https://api.crossref.org/works?rows=100&select=DOI,title,abstract,issued,container-title,type,publisher"
         f"&query.bibliographic={U.quote(s)}&filter=prefix:{prefix},from-pub-date:{FROM}")
    m = json.loads(get(u))["message"]
    kept = 0
    for w in m["items"]:
        title = " ".join(w.get("title") or [])
        ab = re.sub(r"<[^>]+>", " ", w.get("abstract") or "")
        txt = f"{title} {ab}"
        hits = [bool(re.search(c, txt, re.I)) for c in CONCEPTS[q]]
        if not hits[0]:
            continue
        RECS.append({"source": "Crossref", "publisher": pub, "query_id": q, "title": title,
                     "year": ((w.get("issued") or {}).get("date-parts") or [[""]])[0][0],
                     "venue": " ".join(w.get("container-title") or []), "doi": w.get("DOI", "").lower(),
                     "type": w.get("type", ""), "abstract": ab[:1500], "all_concepts_match": all(hits)})
        kept += 1
    log(f"Crossref ({pub})", q, s, f"prefix {prefix}; from {FROM}; top 100 by relevance; concept logic applied locally", m["total-results"], kept)
    time.sleep(2)


def publisher_id(name):
    d = json.loads(get("https://api.openalex.org/publishers?search=" + U.quote(name)))
    return d["results"][0]["id"].rsplit("/", 1)[-1], d["results"][0]["display_name"]


def openalex(q, pub=None, pid=None):
    f = f"from_publication_date:{FROM}-01-01,title_and_abstract.search:{Q[q].replace('*', '')}"
    if pid:
        f += f",primary_location.source.host_organization_lineage:{pid}"
    base = "https://api.openalex.org/works?per-page=200&filter=" + U.quote(f)
    d = json.loads(get(base + "&cursor=*"))
    total, cur, n = d["meta"]["count"], "*", 0
    while cur and n < 400:
        d = json.loads(get(base + "&cursor=" + cur))
        for w in d["results"]:
            pl = w.get("primary_location") or {}
            s = pl.get("source") or {}
            rec = {"source": "OpenAlex", "publisher": pub or "", "query_id": q, "title": w.get("title") or "",
                   "year": w.get("publication_year"), "venue": s.get("display_name") or "",
                   "doi": (w.get("doi") or "").lower().replace("https://doi.org/", ""), "type": w.get("type") or "",
                   "abstract": inverted(w.get("abstract_inverted_index"))[:1500], "all_concepts_match": True}
            (RECS if pub else Q5).append(rec)
        n += len(d["results"])
        cur = d["meta"].get("next_cursor") if d["results"] else None
        time.sleep(0.3)
    log(f"OpenAlex ({pub})" if pub else "OpenAlex", q, Q[q], f"title_and_abstract.search; from {FROM}; truncation removed" + (f"; publisher {pid}" if pid else ""), total, n)


def s2_q5():
    query = Q["q5"].replace(" AND ", " + ").replace(" OR ", " | ")
    u = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + U.urlencode(
        {"query": query, "year": f"{FROM}-", "fields": "title,year,venue,externalIds,abstract"})
    d = json.loads(get(u))
    for w in d.get("data") or []:
        Q5.append({"source": "Semantic Scholar", "publisher": "", "query_id": "q5", "title": w.get("title") or "", "year": w.get("year"),
                   "venue": w.get("venue") or "", "doi": ((w.get("externalIds") or {}).get("DOI") or "").lower(), "type": "",
                   "abstract": (w.get("abstract") or "")[:1500], "all_concepts_match": True})
    log("Semantic Scholar", "q5", query, f"bulk search; year {FROM}-", d.get("total", 0), len(d.get("data") or []))


def arxiv_q5():
    q = ('(all:"document layout" OR all:"layout analysis") AND (all:"compound degradation" OR all:"combined degradation" '
         'OR all:"multiple degradations" OR all:"mixed degradation" OR all:"compound corruption")')
    x = get("http://export.arxiv.org/api/query?" + U.urlencode({"search_query": q, "max_results": 200, "sortBy": "relevance"}))
    r = ET.fromstring(x)
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    n = 0
    for e in r.findall("a:entry", ns):
        y = int(e.find("a:published", ns).text[:4])
        if y < FROM:
            continue
        Q5.append({"source": "arXiv", "publisher": "", "query_id": "q5", "title": re.sub(r"\s+", " ", e.find("a:title", ns).text), "year": y,
                   "venue": "arXiv preprint", "doi": "", "type": "preprint",
                   "abstract": re.sub(r"\s+", " ", e.find("a:summary", ns).text)[:1500], "all_concepts_match": True})
        n += 1
    log("arXiv", "q5", q, f"all fields; year >= {FROM} applied after retrieval", int(r.find("o:totalResults", ns).text), n)


def attempt(label, fn, *a):
    try:
        fn(*a)
    except Exception as e:
        log(label, a[0] if a else "q5", "", "FAILED: " + str(e)[:180], "", 0)


def save():
    for name, rows in (("publisher_records.csv", RECS), ("q5_open_records.csv", Q5)):
        if rows:
            with open(OUT / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(HERE / "search_log_publisher.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LOG[0]))
        w.writeheader()
        w.writerows(LOG)


def main():
    ids = {}
    for pub, (_, name) in PUBLISHERS.items():
        pid, shown = publisher_id(name)
        ids[pub] = pid
        print("OpenAlex publisher", pub, pid, shown, flush=True)
    for q in Q:
        for pub, (prefix, _) in PUBLISHERS.items():
            attempt(f"Crossref ({pub})", crossref, q, prefix, pub)
            attempt(f"OpenAlex ({pub})", openalex, q, pub, ids[pub])
    attempt("OpenAlex", openalex, "q5")
    attempt("Semantic Scholar", s2_q5)
    attempt("arXiv", arxiv_q5)
    save()
    print("records", len(RECS), "q5", len(Q5), "SEARCH_DONE", flush=True)


if __name__ == "__main__":
    main()

"""Literature search: runs fixed queries against open scholarly indexes,
records exact strings and record counts, and saves every returned record."""
import csv, json, re, time, urllib.parse, urllib.request, datetime, pathlib, xml.etree.ElementTree as ET
OUT = pathlib.Path(__file__).parent
UA = {"User-Agent": "generic-literature-search-client/1.0"}
def get(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read().decode("utf-8", "ignore")
        except Exception as e:
            err = e; time.sleep(5 * (i + 1))
    raise RuntimeError(f"{url}: {err}")
Q = {
 "Q1 DLA robustness": '("document layout analysis" OR "document layout detection" OR "layout analysis" OR "page segmentation") AND (robust* OR degrad* OR perturb* OR corrupt* OR noise)',
 "Q2 metrics & downstream": '("document layout analysis" OR "document layout detection") AND (metric* OR "average precision" OR "evaluation") AND (OCR OR "reading order" OR downstream OR "error rate" OR RAG OR "document parsing")',
 "Q3 compound corruption": '("common corruptions" OR "corruption robustness" OR "corrupted images") AND (compound OR combined OR mixture OR multiple OR interaction) AND ("object detection" OR detector*)',
 "Q4 layout detectors": '("document layout analysis" OR "document layout detection") AND (transformer OR DETR OR YOLO OR DocLayNet OR "object detection")',
}
FROM = 2019
log, recs = [], []
def add(src, q, title, year, venue, doi, typ, url, abstract=""):
    recs.append(dict(source=src, query=q, title=(title or "").strip(), year=year, venue=venue or "", doi=(doi or "").lower().replace("https://doi.org/", ""), type=typ or "", url=url or "", abstract=(abstract or "")[:1500]))
def openalex(name, q):
    base = "https://api.openalex.org/works?per-page=200&filter=" + urllib.parse.quote(f"from_publication_date:{FROM}-01-01,title_and_abstract.search:{q.replace('*','')}")
    d = json.loads(get(base + "&cursor=*")); total = d["meta"]["count"]; cur = "*"; n = 0
    while cur and n < 600:
        d = json.loads(get(base + "&cursor=" + cur))
        for w in d["results"]:
            pl = w.get("primary_location") or {}; s = pl.get("source") or {}
            ab = w.get("abstract_inverted_index") or {}
            words = sorted((p, k) for k, v in ab.items() for p in v)
            add("OpenAlex", name, w.get("title"), w.get("publication_year"), s.get("display_name"), w.get("doi"), f"{w.get('type')}/{s.get('type')}", pl.get("landing_page_url"), " ".join(k for _, k in words))
        n += len(d["results"]); cur = d["meta"].get("next_cursor") if d["results"] else None; time.sleep(.3)
    return total, n, f"title_and_abstract.search; from {FROM}; truncation * removed (OpenAlex stems)"
def s2(name, q):
    u = "https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode({"query": q.replace(" AND ", " + ").replace(" OR ", " | "), "year": f"{FROM}-", "fields": "title,year,venue,externalIds,publicationTypes,url,abstract"})
    d = json.loads(get(u)); total = d.get("total", 0); n = 0
    for w in (d.get("data") or [])[:600]:
        e = w.get("externalIds") or {}
        add("Semantic Scholar", name, w.get("title"), w.get("year"), w.get("venue"), e.get("DOI"), ",".join(w.get("publicationTypes") or []), w.get("url"), w.get("abstract")); n += 1
    return total, n, f"bulk search (+ for AND, | for OR); year {FROM}-"
def arxiv(name, q):
    aq = re.sub(r'\b(AND|OR)\b', lambda m: m.group(1), q)
    aq = re.sub(r'"([^"]+)"', lambda m: 'all:"%s"' % m.group(1), aq)
    aq = re.sub(r'(?<![:"\w])(robust|degrad|perturb|corrupt|metric|detector)\*', lambda m: 'all:' + m.group(1) + '*', aq)
    aq = re.sub(r'(?<![:"\w])(noise|OCR|downstream|RAG|transformer|DETR|YOLO|DocLayNet|compound|combined|mixture|multiple|interaction)(?![\w"])', r'all:\1', aq)
    x = get("http://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": aq, "max_results": 300, "sortBy": "relevance"}))
    r = ET.fromstring(x); ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/", "x": "http://arxiv.org/schemas/atom"}
    total = int(r.find("o:totalResults", ns).text); n = 0
    for e in r.findall("a:entry", ns):
        y = int(e.find("a:published", ns).text[:4])
        if y < FROM: continue
        d = e.find("x:doi", ns); j = e.find("x:journal_ref", ns)
        add("arXiv", name, re.sub(r"\s+", " ", e.find("a:title", ns).text), y, j.text if j is not None else "arXiv preprint", d.text if d is not None else "", "preprint", e.find("a:id", ns).text, re.sub(r"\s+", " ", e.find("a:summary", ns).text)); n += 1
    time.sleep(4)
    return total, n, f"all fields; year >= {FROM} applied after retrieval; first 300 by relevance"
def dblp(name, q):
    # DBLP has no Boolean abstract search: use its title keyword search with the concept's core terms
    core = {"Q1 DLA robustness": ["layout robust", "layout degradation", "layout perturbation", "page segmentation robust"],
            "Q2 metrics & downstream": ["layout analysis evaluation metric", "layout analysis OCR evaluation", "document parsing benchmark"],
            "Q3 compound corruption": ["compound corruption", "corruption robustness object detection", "combined corruptions"],
            "Q4 layout detectors": ["document layout detection", "document layout analysis transformer", "document layout YOLO"]}[name]
    total = n = 0
    for c in core:
        d = json.loads(get("https://dblp.org/search/publ/api?" + urllib.parse.urlencode({"q": c, "format": "json", "h": 200})))
        h = d["result"]["hits"]; total += int(h["@total"])
        for w in h.get("hit", []):
            i = w["info"]; y = int(i.get("year", 0))
            if y < FROM: continue
            add("DBLP", name, i.get("title"), y, i.get("venue"), i.get("doi"), i.get("type"), i.get("ee") if isinstance(i.get("ee"), str) else (i.get("ee") or [""])[0]); n += 1
        time.sleep(1)
    return total, n, "title keyword search, phrases: " + "; ".join(core)
for name, q in Q.items():
    for src, fn in [("OpenAlex", openalex), ("Semantic Scholar", s2), ("arXiv", arxiv), ("DBLP", dblp)]:
        try:
            total, n, note = fn(name, q)
            log.append(dict(date=datetime.date.today().isoformat(), source=src, query=name, string=q, filters=note, total_reported=total, records_saved=n)); print(src, name, total, n, flush=True)
        except Exception as e:
            log.append(dict(date=datetime.date.today().isoformat(), source=src, query=name, string=q, filters="FAILED: " + str(e)[:200], total_reported="", records_saved=0)); print(src, name, "FAILED", str(e)[:150], flush=True)
for p, rows in [("search_log.csv", log), ("raw_records.csv", recs)]:
    with open(OUT / p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("records", len(recs))

"""Runs the review queries on IEEE Xplore, SpringerLink and ScienceDirect in a visible browser.

Low volume with long pauses. Uses each site's normal search page. If a site blocks the request,
that is logged and the script moves on; it does not try to get around the block.

Outputs: exports/{ieee,springer,sciencedirect}_q*.csv and search_log_browser.csv
Run: python run_browser_search.py <browser-profile-folder> [ieee springer sciencedirect]
"""
import csv
import datetime
import re
import sys
import time
import urllib.parse as U
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE / "exports"
OUT.mkdir(exist_ok=True)
PROFILE = sys.argv[1]
MAX_IEEE_PAGES = 5   # 100 per page

Q = {
    "q1": '("document layout analysis" OR "document layout detection" OR "layout analysis" OR "page segmentation") AND (robust* OR degrad* OR perturb* OR corrupt* OR noise)',
    "q2": '("document layout analysis" OR "document layout detection") AND (metric* OR "average precision" OR evaluation) AND (OCR OR "reading order" OR downstream OR "error rate" OR RAG OR "document parsing")',
    "q3": '("common corruptions" OR "corruption robustness" OR "corrupted images") AND (compound OR combined OR mixture OR multiple OR interaction) AND ("object detection" OR detector*)',
    "q4": '("document layout analysis" OR "document layout detection") AND (transformer OR DETR OR YOLO OR DocLayNet OR "object detection")',
    "q5": '("document layout" OR "layout analysis") AND ("compound degradation" OR "combined degradation" OR "multiple degradations" OR "mixed degradation" OR "compound corruption")',
}
# ScienceDirect has no wildcards and limits Boolean operators, so these are shortened.
QSD = {
    "q1": '("document layout analysis" OR "layout analysis" OR "page segmentation") AND (robust OR degradation OR perturbation OR corruption OR noise)',
    "q2": '"document layout analysis" AND (metric OR "average precision" OR evaluation OR OCR OR downstream OR "error rate")',
    "q3": '("common corruptions" OR "corruption robustness") AND (compound OR combined OR multiple OR interaction) AND "object detection"',
    "q4": '"document layout analysis" AND (transformer OR DETR OR YOLO OR DocLayNet OR "object detection")',
    "q5": Q["q5"],
}
LOG_PATH = HERE / "search_log_browser.csv"
LOG = list(csv.DictReader(open(LOG_PATH, encoding="utf-8"))) if LOG_PATH.exists() else []


def log(db, q, s, total, saved, note):
    LOG[:] = [r for r in LOG if not (r["database"] == db and r["query_id"] == q)]
    LOG.append({"database": db, "date": datetime.date.today().isoformat(), "query_id": q, "search_string": s,
                "filters": "2019-2026", "records_returned": total, "records_saved": saved, "note": note})
    with open(HERE / "search_log_browser.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(LOG[0]))
        w.writeheader()
        w.writerows(LOG)
    print(db, q, "total", total, "saved", saved, note, flush=True)


def save(name, rows):
    if not rows:
        return
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def dismiss(pg):
    for label in ("Reject optional cookies", "Accept only necessary cookies", "Accept & Close"):
        try:
            b = pg.get_by_role("button", name=label)
            if b.count():
                b.first.click(timeout=3000)
                time.sleep(1)
                return
        except Exception:
            pass


def text(el):
    return " ".join(el.inner_text().split()) if el else ""


def body_text(pg):
    return " ".join(pg.inner_text("body").split())


def ieee(pg):
    for k, s in Q.items():
        rows, total, note = [], None, ""
        for n in range(1, MAX_IEEE_PAGES + 1):
            u = ("https://ieeexplore.ieee.org/search/searchresult.jsp?action=search&matchBoolean=true"
                 f"&queryText={U.quote(s)}&ranges=2019_2026_Year&rowsPerPage=100&pageNumber={n}")
            pg.goto(u, timeout=60000, wait_until="domcontentloaded")
            time.sleep(10)
            dismiss(pg)
            body = body_text(pg)
            if total is None:
                m = re.search(r"Showing \S+ of ([\d,]+) results", body)
                total = int(m.group(1).replace(",", "")) if m else (0 if "No results found" in body else None)
                if total is None:
                    note = "count not found: " + body[:100]
            items = pg.query_selector_all("xpl-results-item")
            for it in items:
                t = it.query_selector("h3 a")
                rows.append({"title": t.inner_text().strip() if t else "",
                             "link": ("https://ieeexplore.ieee.org" + t.get_attribute("href")) if t else "",
                             "info": text(it.query_selector(".publisher-info-container") or it.query_selector(".description")),
                             "abstract": text(it.query_selector(".twist-container"))})
            if not items or total is None or len(rows) >= total:
                break
            time.sleep(8)
        save(f"ieee_{k}.csv", rows)
        log("IEEE Xplore", k, s, total, len(rows), note)
        time.sleep(10)


def springer(pg):
    for k, s in Q.items():
        u = f"https://link.springer.com/search?query={U.quote(s)}&date=custom&dateFrom=2019&dateTo=2026"
        pg.goto(u, timeout=60000, wait_until="domcontentloaded")
        time.sleep(10)
        dismiss(pg)
        body = body_text(pg)
        m = re.search(r"Showing \S+ of ([\d,]+) results", body)
        total = int(m.group(1).replace(",", "")) if m else (0 if "No results" in body else None)
        saved, note = 0, "" if m else "count not found: " + body[:100]
        if total:
            try:
                href = pg.get_by_text("Download results (.csv)").first.evaluate("e => e.closest('a').href")
                r = pg.context.request.get(href, timeout=120000)
                if not r.ok:
                    raise RuntimeError(f"HTTP {r.status}")
                p = OUT / f"springer_{k}.csv"
                p.write_bytes(r.body())
                with open(p, encoding="utf-8", errors="ignore") as f:
                    saved = max(0, sum(1 for _ in csv.reader(f)) - 1)
            except Exception as e:
                note = "download failed: " + str(e)[:80]
        log("SpringerLink", k, s, total, saved, note)
        time.sleep(12)


def sciencedirect(pg):
    retried = False
    for k, s in QSD.items():
        u = f"https://www.sciencedirect.com/search?tak={U.quote(s)}&date=2019-2026&show=100"
        pg.goto(u, timeout=60000, wait_until="domcontentloaded")
        time.sleep(12)
        dismiss(pg)
        body = body_text(pg)
        if "could not be processed" in body and not retried:
            retried = True
            time.sleep(20)
            pg.reload(wait_until="domcontentloaded")
            time.sleep(15)
            body = body_text(pg)
        if "could not be processed" in body:
            log("ScienceDirect", k, s, None, 0, "site refused the request; stopped this site")
            pg.screenshot(path=str(OUT / "sciencedirect_blocked.png"))
            return
        m = re.search(r"([\d,]+) results?", body)
        total = int(m.group(1).replace(",", "")) if m else (0 if "No results" in body else None)
        rows = []
        for it in pg.query_selector_all("li.ResultItem"):
            t = it.query_selector("h2 a")
            rows.append({"title": t.inner_text().strip() if t else "",
                         "link": ("https://www.sciencedirect.com" + t.get_attribute("href")) if t else "",
                         "info": text(it.query_selector(".srctitle-date-fields")),
                         "doi": it.get_attribute("data-doi") or ""})
        save(f"sciencedirect_{k}.csv", rows)
        log("ScienceDirect", k, s, total, len(rows), "" if m else "count not found: " + body[:100])
        time.sleep(15)


def main():
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(PROFILE, channel="chrome", headless=False, accept_downloads=True,
                                                   viewport={"width": 1300, "height": 900})
        pg = ctx.new_page()
        sites = {"ieee": ieee, "springer": springer, "sciencedirect": sciencedirect}
        for fn in [sites[n] for n in (sys.argv[2:] or list(sites))]:
            try:
                fn(pg)
            except Exception as e:
                print("SITE ERROR", fn.__name__, str(e)[:300], flush=True)
        ctx.close()
    print("SEARCH_DONE", flush=True)


if __name__ == "__main__":
    main()

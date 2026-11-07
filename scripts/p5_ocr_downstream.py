"""
p5_ocr_downstream.py -- P5 (Alishba, Workstream A): downstream reading quality.

RQ2 asks whether layout mAP predicts what a document pipeline actually delivers.
This builds that pipeline: page -> layout regions -> text in reading order, and
measures how much of the text a detector's layout gets right.

The OCR engine is held exactly constant: Tesseract reads each (degraded) page ONCE,
and its words are assigned to regions by word centre. Two pipelines then read the
SAME words from the SAME image:

  reference  ground-truth regions  -> text
  detector   predicted regions (score >= TAU) -> text

so every difference between them is caused by layout alone. Region-crop semantics
are kept: a word is read by every region that contains it (duplicate boxes
duplicate text, as cropping each box would), regions are read in XY-cut order,
and Picture regions are not read. Pictures and words outside all regions are
ignored by both pipelines alike.

Per page and detector:
  WER             word-level edit distance (detector text vs reference text) / reference words
  text_recall     share of reference words (multiset) the detector pipeline reads at all
  text_precision  share of detector-pipeline words that exist in the reference (penalises duplicates)
  table_recall    share of words inside GT Tables read by a predicted Table region

Scope: all 300 pages; clean + every severity-2 single and pair (37 conditions).

Outputs: RDLD_data/ocr/<config>/<page>.json        cached Tesseract words
         RDLD_data/results/p5/downstream/{downstream_pages.csv, downstream_configs.csv}

Run:
  python scripts/p5_ocr_downstream.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rdld import config
from rdld.doclaynet import NAME_TO_ID, load_coco

TAU = float(os.getenv("TAU", "0.5"))
SEV = 2
WORKERS = int(os.getenv("WORKERS", "14"))
OCR = config.DATA_ROOT / "ocr"
OUT = config.out_dir("p5", "downstream")
PICTURE, TABLE = NAME_TO_ID["Picture"], NAME_TO_ID["Table"]
TESS = shutil.which("tesseract") or str(Path(sys.executable).parent / "Library" / "bin" / "tesseract.exe")
# conda's activation script sets this; calling the env's python directly does not
os.environ.setdefault("TESSDATA_PREFIX", str(Path(sys.executable).parent / "share" / "tessdata"))


# ------------------------------------------------------------------ OCR (cached)
def ocr_page(args):
    cfg, fname = args
    dst = OCR / cfg / fname.replace(".png", ".json")
    if dst.exists():
        return
    src = (config.IMAGES if cfg == "clean" else config.DEGRADED / cfg) / fname
    r = subprocess.run([TESS, str(src), "stdout", "--psm", "3", "-l", "eng", "-c", "tessedit_create_tsv=1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0 or not r.stdout.startswith("level"):
        raise RuntimeError(f"tesseract failed on {src}: {r.stderr.strip()[:300]}")
    tsv = r.stdout
    words = []
    for line in tsv.splitlines()[1:]:
        f = line.split("\t")
        if len(f) == 12 and f[0] == "5" and f[11].strip() and float(f[10]) >= 0:
            x, y, w, h = map(int, f[6:10])
            words.append([f[11].strip(), x + w / 2, y + h / 2])
    dst.write_text(json.dumps(words), encoding="utf-8")


# ------------------------------------------------------------------ pipeline
def xycut(boxes, idx=None):
    """Recursive XY-cut reading order over xyxy boxes; returns indices in reading order."""
    idx = list(range(len(boxes))) if idx is None else idx
    if len(idx) <= 1:
        return idx
    for axis in (1, 0):                                   # horizontal cut first, then vertical
        lo, hi = boxes[idx, axis], boxes[idx, axis + 2]
        order = np.argsort(lo)
        reach, cut = hi[order[0]], None
        for k in range(1, len(order)):
            if lo[order[k]] > reach:                       # gap: everything before is one block
                cut = k
                break
            reach = max(reach, hi[order[k]])
        if cut is not None:
            first = [idx[i] for i in order[:cut]]
            rest = [idx[i] for i in order[cut:]]
            return xycut(boxes, first) + xycut(boxes, rest)
    return sorted(idx, key=lambda i: (boxes[i, 1], boxes[i, 0]))   # no clean cut: top-left order


def read(words, boxes, classes):
    """Text a region pipeline produces: regions in XY-cut order, words by centre."""
    keep = [i for i, c in enumerate(classes) if c != PICTURE]
    if not keep or not words:
        return []
    b = boxes[keep]
    text = []
    for r in xycut(b):
        x1, y1, x2, y2 = b[r]
        inside = [w for w in words if x1 <= w[1] <= x2 and y1 <= w[2] <= y2]
        inside.sort(key=lambda w: (round(w[2] / 8), w[1]))  # lines, then left to right
        text += [w[0] for w in inside]
    return text


def wer(ref, hyp):
    if not ref:
        return float(len(hyp) > 0)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def table_words(words, boxes, classes):
    out = Counter()
    for (x1, y1, x2, y2), c in zip(boxes, classes):
        if c == TABLE:
            out.update(w[0] for w in words if x1 <= w[1] <= x2 and y1 <= w[2] <= y2)
    return out


_state = None


def score(args):
    global _state
    if _state is None:
        g = load_coco()
        by = defaultdict(list)
        for a in g["annotations"]:
            by[a["image_id"]].append(a)
        gt = {im["id"]: (np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                                   for a in by[im["id"]]]), [a["category_id"] for a in by[im["id"]]], im["file_name"])
              for im in g["images"]}
        _state = gt
    model, cfg = args
    dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    pred = defaultdict(list)
    for d in dets:
        if d["score"] >= TAU:
            x, y, w, h = d["bbox"]
            pred[d["image_id"]].append(([x, y, x + w, y + h], d["category_id"]))
    rows = []
    for iid, (gb, gc, fname) in _state.items():
        words = json.loads((OCR / cfg / fname.replace(".png", ".json")).read_text(encoding="utf-8"))
        ref = read(words, gb, gc)
        pb = np.array([p[0] for p in pred[iid]]).reshape(-1, 4)
        pc = [p[1] for p in pred[iid]]
        hyp = read(words, pb, pc)
        rc, hc = Counter(ref), Counter(hyp)
        common = sum((rc & hc).values())
        tw_ref = table_words(words, gb, gc)
        tw_hyp = table_words(words, pb, pc)
        rows.append({"model": model, "config": cfg, "image_id": iid,
                     "n_ref_words": len(ref), "n_hyp_words": len(hyp),
                     "WER": wer(ref, hyp),
                     "text_recall": common / max(len(ref), 1),
                     "text_precision": common / max(len(hyp), 1),
                     "n_table_words": sum(tw_ref.values()),
                     "table_recall": sum((tw_ref & tw_hyp).values()) / sum(tw_ref.values()) if tw_ref else np.nan})
    return rows


def main():
    if not Path(TESS).exists():
        raise SystemExit("tesseract not found -- install it into the env (conda-forge tesseract)")
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    sel = configs[(configs.kind == "clean") | (configs.severity == SEV)].config.tolist()
    images = load_coco()["images"]
    for c in sel:
        (OCR / c).mkdir(parents=True, exist_ok=True)
    jobs = [(c, im["file_name"]) for c in sel for im in images]
    ocr_page(jobs[0])                                    # fail fast before launching the pool
    if not json.loads((OCR / "clean" / images[0]["file_name"].replace(".png", ".json")).read_text(encoding="utf-8")):
        raise SystemExit("tesseract returned no words on the first clean page -- check TESSDATA_PREFIX")
    print(f"OCR: {len(sel)} conditions x {len(images)} pages ({TESS})")
    with Pool(WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(ocr_page, jobs, chunksize=4), 1):
            if i % 1000 == 0 or i == len(jobs):
                print(f"  ocr {i}/{len(jobs)}")

    models = sorted(p.name for p in config.PREDICTIONS.iterdir() if p.is_dir())
    with Pool(WORKERS) as pool:
        rows = [r for rs in pool.imap_unordered(score, [(m, c) for m in models for c in sel]) for r in rs]
    P = pd.DataFrame(rows)
    P.to_csv(OUT / "downstream_pages.csv", index=False)
    # word-weighted aggregates per detector x condition
    P["err_words"] = P.WER * P.n_ref_words
    C = P.groupby(["model", "config"]).apply(lambda g: pd.Series({
        "WER": g.err_words.sum() / max(g.n_ref_words.sum(), 1),
        "text_recall": (g.text_recall * g.n_ref_words).sum() / max(g.n_ref_words.sum(), 1),
        "text_precision": (g.text_precision * g.n_hyp_words).sum() / max(g.n_hyp_words.sum(), 1),
        "table_recall": (g.table_recall * g.n_table_words).sum() / max(g.n_table_words.sum(), 1),
        "ref_words": g.n_ref_words.sum()}), include_groups=False).reset_index().merge(configs, on="config")
    C.to_csv(OUT / "downstream_configs.csv", index=False)
    print("\nclean condition:")
    print(C[C.config == "clean"].round(4).to_string(index=False))
    print("\nmean over severity-2 conditions:")
    print(C[C.kind != "clean"].groupby(["model", "kind"])[["WER", "text_recall", "text_precision", "table_recall"]].mean().round(4).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

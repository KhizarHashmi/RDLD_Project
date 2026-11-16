"""
p1_validate_corpus.py -- P1 (Alishba, Workstream A): validate the clean corpus.

Every downstream number assumes the ground truth is sound. This checks the
300-page COCO file and images before any metric is computed on them:

  - every image exists, opens, and matches the declared width/height
  - every box is inside the page, has positive size, and a valid class id
  - stored `area` vs. box w*h (DocLayNet stores segmentation area)
  - exact duplicate boxes, and same-region boxes with different classes
    (ground-truth label ambiguity: the GT-side twin of detector instability)
  - manifest agrees with the annotations

Outputs: RDLD_data/results/p1/validation/{checks.csv, issues.csv}

Run:
  python scripts/p1_validate_corpus.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from rdld import config
from rdld.doclaynet import CATEGORIES, DOC_CATEGORIES, load_coco

OUT = config.out_dir("p1", "validation")
TOL = 1.0          # px tolerance for boxes touching the page edge

gt = load_coco()
man = pd.read_csv(config.MANIFEST, dtype={"id": str})
checks, issues = [], []


def check(name, ok, detail):
    checks.append({"check": name, "status": "PASS" if ok else "WARN", "detail": detail})
    print(f"  [{'PASS' if ok else 'WARN'}] {name:44s} {detail}")


print("=" * 70)
print(" P1 CORPUS VALIDATION")
print("=" * 70)

# ---------------------------------------------------------------- images
im_by = {im["id"]: im for im in gt["images"]}
bad_open, bad_size, modes = [], [], Counter()
for im in gt["images"]:
    p = config.IMAGES / im["file_name"]
    try:
        with Image.open(p) as f:
            modes[f.mode] += 1
            if f.size != (im["width"], im["height"]):
                bad_size.append(im["file_name"])
    except Exception as e:                        # noqa: BLE001 - report any failure
        bad_open.append(f"{im['file_name']}: {type(e).__name__}")
check("images open", not bad_open, f"{len(gt['images']) - len(bad_open)}/{len(gt['images'])}")
check("image size matches COCO record", not bad_size, f"{len(bad_size)} mismatches")
check("image ids unique", len(im_by) == len(gt["images"]), f"{len(im_by)} unique")
sizes = Counter((im["width"], im["height"]) for im in gt["images"])
check("single page size", len(sizes) == 1, f"{dict(sizes)}; modes {dict(modes)}")

# ---------------------------------------------------------------- boxes
by_img = defaultdict(list)
n_out, n_nonpos, n_badcls = 0, 0, 0
area_ratio = []
for a in gt["annotations"]:
    x, y, w, h = a["bbox"]
    im = im_by.get(a["image_id"])
    if im is None:
        issues.append({"type": "orphan annotation", "image": a["image_id"], "ann": a["id"], "detail": ""})
        continue
    if w <= 0 or h <= 0:
        n_nonpos += 1
        issues.append({"type": "non-positive size", "image": im["file_name"], "ann": a["id"], "detail": str(a["bbox"])})
    if x < -TOL or y < -TOL or x + w > im["width"] + TOL or y + h > im["height"] + TOL:
        n_out += 1
        issues.append({"type": "box outside page", "image": im["file_name"], "ann": a["id"], "detail": str(a["bbox"])})
    if a["category_id"] not in CATEGORIES:
        n_badcls += 1
        issues.append({"type": "invalid class", "image": im["file_name"], "ann": a["id"], "detail": str(a["category_id"])})
    if w > 0 and h > 0:
        area_ratio.append(a["area"] / (w * h))
    by_img[a["image_id"]].append(a)

n_ann = len(gt["annotations"])
check("boxes inside page (+/-1 px)", n_out == 0, f"{n_out}/{n_ann} outside")
check("boxes have positive size", n_nonpos == 0, f"{n_nonpos}/{n_ann} degenerate")
check("class ids in 1..11", n_badcls == 0, f"{n_badcls} invalid")
ar = np.array(area_ratio)
check("stored area ~= box w*h", bool(np.all(np.abs(ar - 1) < 0.01)),
      f"area/(w*h): median {np.median(ar):.3f}, min {ar.min():.3f}, max {ar.max():.3f}")
empty = [im["file_name"] for im in gt["images"] if not by_img.get(im["id"])]
check("every page has annotations", not empty, f"{len(empty)} empty pages")


# ---------------------------------------------------------------- duplicates
def iou(b1, b2):
    ax1, ay1, aw, ah = b1; bx1, by1, bw, bh = b2
    ix = max(0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1e-9)


exact, cross = 0, 0
for iid, anns in by_img.items():
    for i in range(len(anns)):
        for j in range(i + 1, len(anns)):
            v = iou(anns[i]["bbox"], anns[j]["bbox"])
            if v >= 0.999 and anns[i]["category_id"] == anns[j]["category_id"]:
                exact += 1
                issues.append({"type": "exact duplicate", "image": im_by[iid]["file_name"],
                               "ann": f"{anns[i]['id']},{anns[j]['id']}", "detail": CATEGORIES[anns[i]["category_id"]]})
            elif v >= 0.9 and anns[i]["category_id"] != anns[j]["category_id"]:
                cross += 1
                issues.append({"type": "same region, different class", "image": im_by[iid]["file_name"],
                               "ann": f"{anns[i]['id']},{anns[j]['id']}",
                               "detail": f"{CATEGORIES[anns[i]['category_id']]} vs {CATEGORIES[anns[j]['category_id']]} IoU={v:.2f}"})
check("no exact duplicate GT boxes", exact == 0, f"{exact} pairs")
check("no cross-class GT boxes on one region", cross == 0, f"{cross} pairs (IoU>=0.9)")

# ---------------------------------------------------------------- manifest
mcount = dict(zip(man["id"].astype(int), man["n_regions"]))
mismatch = [i for i, n in mcount.items() if len(by_img.get(i, [])) != n]
check("manifest rows == images", len(man) == len(gt["images"]), f"{len(man)} rows")
check("manifest n_regions == annotations", not mismatch, f"{len(mismatch)} mismatches")
cats = Counter(man["doc_category"])
check("50 pages per document category", all(cats[c] == 50 for c in DOC_CATEGORIES), str(dict(cats)))

pd.DataFrame(checks).to_csv(OUT / "checks.csv", index=False)
pd.DataFrame(issues, columns=["type", "image", "ann", "detail"]).to_csv(OUT / "issues.csv", index=False)
n_warn = sum(c["status"] == "WARN" for c in checks)
print(f"\n  {len(checks) - n_warn}/{len(checks)} checks pass, {len(issues)} issue rows -> {OUT}")

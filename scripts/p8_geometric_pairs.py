"""
p8_geometric_pairs.py -- P8 (Alishba, Workstream A): pairs of an image degradation and a geometric one.

The main experiment uses image-only degradations, so the clean ground truth holds on every page.
Geometric degradations move the regions. This pilot asks whether pairs still compound when one
member is geometric. It builds, on the test pages at severity 2,

  2 geometric singles   rotation@2, keystoning@2
  16 pairs              each of the eight image degradations with each geometric one

Both geometric functions follow RoDLA's spatial.py with the same parameters. RoDLA's own code needs
detectron2 and imgaug, which do not run in our environment, so the two transforms are written here
with OpenCV:

  rotation@2    angle drawn from -10..-5 and 5..10 degrees, canvas grown to hold the whole page,
                black corners (detectron2 RotationTransform, expand=True)
  keystoning@2  corners pulled in by |N(0, s)| of the page size, s drawn from U(0.01, 0.06), the
                quadrilateral mapped to a rectangle, whole page kept, resized back to the page size,
                black fill (imgaug PerspectiveTransform, fit_output=True, keep_size=True)

Every box is moved with its page: its four corners go through the same transform and the new box is
the smallest upright box around them, as in RoDLA. Each condition folder therefore carries its own
annotations.json, and transforms.json keeps the matrix of every page.

Physical order. Paper, printing and overlays come before the page is turned or photographed at an
angle, and the capture lighting and blur come after it:
  texture, ink_bleeding, ink_holdout, speckle, watermark  ->  geometry  ->  illumination, defocus, vibration
So texture@2+rotation@2 turns the cached texture@2 page, and rotation@2+defocus@2 blurs the cached
rotation@2 page. The geometric draw uses the same seed in the single and in the pair (common random
numbers), so a pair has exactly the same boxes as its geometric single.

  <data root>/degraded/configs_geom.csv
  <data root>/degraded/<config>/<page>.png, annotations.json, transforms.json

Run (test pages):
  RDLD_DATA=../RDLD_data_test python scripts/p8_geometric_pairs.py --check   # 3 pages, writes nothing to the cache
  RDLD_DATA=../RDLD_data_test python scripts/p8_geometric_pairs.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from rdld import config
from rdld.doclaynet import load_coco

SEV = 2
GEOM = ["rotation", "keystoning"]
BEFORE = ["texture", "ink_bleeding", "ink_holdout", "speckle", "watermark"]   # applied before the geometry
AFTER = ["illumination", "defocus", "vibration"]                              # applied after it
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 4) - 2)))


# ---------------------------------------------------------------- geometric transforms

def rotation_matrix(h, w, degree):
    """3x3 matrix and output size of RoDLA's rotation (detectron2 RotationTransform, expand=True)."""
    lo, hi = [0, 5, 10][degree - 1], [5, 10, 15][degree - 1]
    if lo == 0:
        angle = np.random.choice(np.arange(-hi, hi + 1))
    else:
        angle = np.random.choice(np.concatenate([np.arange(-hi, -lo + 1), np.arange(lo, hi + 1)]))
    c, s = abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle)))
    bw, bh = int(np.rint(h * s + w * c)), int(np.rint(h * c + w * s))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), float(angle), 1.0)
    M[:, 2] += np.array([bw / 2, bh / 2]) - np.array([w / 2, h / 2])
    return np.vstack([M, [0, 0, 1]]), (bw, bh), {"angle": int(angle)}


def keystoning_matrix(h, w, degree):
    """3x3 matrix and output size of RoDLA's keystoning (imgaug PerspectiveTransform)."""
    sigma = np.random.uniform(0.01, 0.02 * (2 * degree - 1))
    p = np.mod(np.abs(np.random.normal(0, sigma, size=(4, 2))), 1)
    p[1, 0] = 1 - p[1, 0]          # top right
    p[2] = 1 - p[2]                # bottom right
    p[3, 1] = 1 - p[3, 1]          # bottom left
    quad = (p * [w, h]).astype(np.float32)
    mw = int(max(np.hypot(*(quad[2] - quad[3])), np.hypot(*(quad[1] - quad[0]))))
    mh = int(max(np.hypot(*(quad[1] - quad[2])), np.hypot(*(quad[0] - quad[3]))))
    rect = np.float32([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]])
    P = cv2.getPerspectiveTransform(quad, rect)
    corners = cv2.perspectiveTransform(np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), P)[0]
    lo = corners.min(axis=0)                                   # fit_output: keep the whole page
    size = np.ceil(corners.max(axis=0) - lo).astype(int)
    shift = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]])
    scale = np.diag([w / size[0], h / size[1], 1])            # keep_size: back to the page size
    return scale @ shift @ P, (w, h), {"sigma": float(sigma)}


MATRIX = {"rotation": rotation_matrix, "keystoning": keystoning_matrix}


def draw(name, page_id, h, w):
    """The geometric draw for one page: the same in the single and in every pair."""
    from rdld import degradations as D
    np.random.seed(D.seed_for(name, SEV, page_id))
    return MATRIX[name](h, w, SEV)


def warp(img, M, size):
    return cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def move_boxes(anns, M, size):
    out = []
    for a in anns:
        x, y, bw, bh = a["bbox"]
        pts = cv2.perspectiveTransform(np.float32([[[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]]), M)[0]
        x0, y0 = np.clip(pts.min(axis=0), 0, size)
        x1, y1 = np.clip(pts.max(axis=0), 0, size)
        b = copy.deepcopy(a)
        b["bbox"] = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
        b["area"] = float((x1 - x0) * (y1 - y0))
        b["segmentation"] = [pts.reshape(-1).astype(float).tolist()]
        out.append(b)
    return out


# ---------------------------------------------------------------- conditions

def conditions():
    from rdld import degradations as D
    rows = []
    for g in GEOM:
        rows.append({"config": f"{g}@{SEV}", "kind": "geom_single", "severity": SEV, "a": g, "b": "",
                     "geom": g, "image_deg": "", "single_a": "", "single_b": ""})
    for g in GEOM:
        for n in D.DEGRADATIONS:
            a, b = (n, g) if n in BEFORE else (g, n)
            rows.append({"config": f"{a}@{SEV}+{b}@{SEV}", "kind": "geom_pair", "severity": SEV, "a": a, "b": b,
                         "geom": g, "image_deg": n, "single_a": f"{a}@{SEV}", "single_b": f"{b}@{SEV}"})
    return pd.DataFrame(rows)


def build(r, im, source=None):
    """One page of one condition. source=None reads the cached singles; source='clean' builds
    everything from the clean page (used by --check)."""
    from rdld import degradations as D
    fname, pid = im["file_name"], im["id"]
    clean = lambda: cv2.imread(str(config.IMAGES / fname))
    g, n = r["geom"], r["image_deg"]
    h, w = im["height"], im["width"]
    M, size, info = draw(g, pid, h, w)
    if r["kind"] == "geom_single":
        return warp(clean(), M, size), M, size, info
    if n in BEFORE:
        base = (D.apply_one(clean(), n, SEV, pid) if source == "clean"
                else cv2.imread(str(config.DEGRADED / f"{n}@{SEV}" / fname)))
        return warp(base, M, size), M, size, info
    base = warp(clean(), M, size) if source == "clean" else cv2.imread(str(config.DEGRADED / f"{g}@{SEV}" / fname))
    return D.apply_one(base, n, SEV, pid), M, size, info


def _job(args):
    r, im = args
    dst = config.DEGRADED / r["config"] / im["file_name"]
    if dst.exists():
        return None
    img, M, size, info = build(r, im)
    cv2.imwrite(str(dst), img, PNG)
    return None


def annotations(coco, g):
    """Ground truth and per-page transforms for geometric draw g (shared by its single and pairs)."""
    gt = copy.deepcopy(coco)
    by_img = {}
    for a in coco["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    anns, tf = [], {}
    for im in gt["images"]:
        M, size, info = draw(g, im["id"], im["height"], im["width"])
        anns += move_boxes(by_img.get(im["id"], []), M, size)
        tf[im["id"]] = {"M": M.tolist(), "in_size": [im["width"], im["height"]], "out_size": list(size), **info}
        im["width"], im["height"] = int(size[0]), int(size[1])
    gt["annotations"] = anns
    return gt, tf


def check(coco):
    """3 pages: cached route == clean route, boxes inside the page, and a picture of each with its boxes."""
    out = Path(os.getenv("CHECK_OUT", config.RESULTS / "p8" / "geometric_check"))
    out.mkdir(parents=True, exist_ok=True)
    C = conditions()
    pick = C[C.config.isin(["rotation@2", "keystoning@2", "texture@2+rotation@2", "speckle@2+keystoning@2",
                            "rotation@2+vibration@2", "keystoning@2+defocus@2"])]
    images = coco["images"][:3]
    gts = {g: annotations(coco, g)[0] for g in GEOM}
    for r in pick[pick.image_deg.isin(BEFORE)].to_dict("records"):     # these read a cached image single
        for im in images:
            same = np.array_equal(build(r, im, "clean")[0], build(r, im)[0])
            print(f"  {r['config']:28s} page {im['id']:>3}: {'identical' if same else 'DIFFERENT'}")
            if not same:
                raise SystemExit("cached route differs from the clean route")
    for g in GEOM:                                                     # the draw repeats exactly
        im = images[0]
        assert np.array_equal(draw(g, im["id"], im["height"], im["width"])[0],
                              draw(g, im["id"], im["height"], im["width"])[0])
    for g in GEOM:
        gt = gts[g]
        bad = [a for a in gt["annotations"] if a["bbox"][2] <= 0 or a["bbox"][3] <= 0]
        print(f"  {g}: {len(gt['annotations'])} boxes, {len(bad)} with no area")
    for r in pick.to_dict("records"):
        for im in images[:2]:
            img = build(r, im, "clean")[0].copy()
            for a in gts[r["geom"]]["annotations"]:
                if a["image_id"] == im["id"]:
                    x, y, bw, bh = map(int, a["bbox"])
                    cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
            cv2.imwrite(str(out / f"{r['config'].replace('@', '').replace('+', '_')}_{im['id']}.jpg"), img)
    print(f"check passed; pictures -> {out}")


def main():
    coco = load_coco()
    if "--check" in sys.argv:
        return check(coco)
    C = conditions()
    C.to_csv(config.DEGRADED / "configs_geom.csv", index=False)
    for g in GEOM:
        gt, tf = annotations(coco, g)
        for cfg in C[C.geom == g].config:
            d = config.DEGRADED / cfg
            d.mkdir(exist_ok=True)
            (d / "annotations.json").write_text(json.dumps(gt), encoding="utf-8")
            (d / "transforms.json").write_text(json.dumps(tf), encoding="utf-8")
    jobs = []
    for stage in ("geom_single", "geom_pair"):                    # pairs after the geometric singles
        stage_jobs = [(r, im) for r in C[C.kind == stage].to_dict("records") for im in coco["images"]]
        with Pool(WORKERS) as pool:
            for i, _ in enumerate(pool.imap_unordered(_job, stage_jobs, chunksize=4), 1):
                if i % 1000 == 0 or i == len(stage_jobs):
                    print(f"  {stage}: {i}/{len(stage_jobs)}")
        jobs += stage_jobs
    print(f"geometric cache: {len(C)} conditions x {len(coco['images'])} pages -> {config.DEGRADED}")


if __name__ == "__main__":
    main()

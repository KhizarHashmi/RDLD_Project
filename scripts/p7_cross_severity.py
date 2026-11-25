"""
p7_cross_severity.py -- P7 (Alishba, Workstream A): pairs at mixed severities.

The main experiment applies both degradations of a pair at the same severity. Real pages
are rarely that tidy: a light speckle can meet a strong motion blur. This builds the
five strongest pairs at every mixed severity, (1,2) (2,1) (1,3) (3,1) (2,3) (3,2),
which gives 30 new conditions.

The five pairs were chosen on the VALIDATION pages, as the five with the largest mean
multiplicative interaction over the eleven detectors at severity 2
(RDLD_data/results/p3/rq1/rq1_interactions.csv). The conditions are then built and
scored on the TEST pages, so the pages that picked the pairs are not the pages that
test them.

Each pair is built exactly like the main cache: degradation A is read from the cached
single A@s_a, then B is applied at s_b with the same seed as the single B@s_b. The
common random numbers therefore hold, and the interaction can be read against the two
cached singles.

  <data root>/degraded/configs_xsev.csv
  <data root>/degraded/<a>@<s_a>+<b>@<s_b>/<page>.png

Run (test pages):
  RDLD_DATA=../RDLD_data_test python scripts/p7_cross_severity.py --check   # 3 pages, cached route == direct route
  RDLD_DATA=../RDLD_data_test python scripts/p7_cross_severity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from rdld import config
from rdld.doclaynet import load_coco

# (A, B) in physical order; largest mean I_mult at severity 2 on the validation pages
PAIRS = [("speckle", "vibration"), ("texture", "vibration"), ("speckle", "defocus"),
         ("ink_holdout", "vibration"), ("ink_bleeding", "speckle")]
MIXED = [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)]
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 4) - 2)))


def conditions():
    from rdld import degradations as D
    rows = []
    for a, b in PAIRS:
        assert D.ORDER[a] < D.ORDER[b], (a, b)
        for sa, sb in MIXED:
            rows.append({"config": D.config_id([(a, sa), (b, sb)]), "kind": "pair_xsev", "severity": 0,
                         "a": a, "b": b, "sev_a": sa, "sev_b": sb,
                         "single_a": D.config_id([(a, sa)]), "single_b": D.config_id([(b, sb)])})
    return pd.DataFrame(rows)


def build(r, page_id, fname):
    from rdld import degradations as D
    img = cv2.imread(str(config.DEGRADED / r["single_a"] / fname))
    if img is None:
        raise FileNotFoundError(config.DEGRADED / r["single_a"] / fname)
    return D.apply_one(img, r["b"], r["sev_b"], page_id)


def _job(args):
    r, page_id, fname = args
    dst = config.DEGRADED / r["config"] / fname
    if not dst.exists():
        cv2.imwrite(str(dst), build(r, page_id, fname), PNG)


def check(images):
    """The cached route must give the same image as applying both steps to the clean page."""
    from rdld import degradations as D
    C = conditions()
    for r in C.iloc[[0, 13, 29]].to_dict("records"):
        for im in images[:3]:
            cached = build(r, im["id"], im["file_name"])
            direct = D.apply_config(cv2.imread(str(config.IMAGES / im["file_name"])), r["config"], im["id"])
            same = np.array_equal(cached, direct)
            print(f"  {r['config']:32s} page {im['id']:>4}: {'identical' if same else 'DIFFERENT'}")
            if not same:
                raise SystemExit("cached route differs from the direct route")
    print("check passed")


def main():
    images = load_coco()["images"]
    if "--check" in sys.argv:
        return check(images)
    C = conditions()
    C.to_csv(config.DEGRADED / "configs_xsev.csv", index=False)
    jobs = []
    for r in C.to_dict("records"):
        (config.DEGRADED / r["config"]).mkdir(exist_ok=True)
        jobs += [(r, im["id"], im["file_name"]) for im in images]
    with Pool(WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
            if i % 1500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    print(f"cross-severity cache: {len(C)} conditions x {len(images)} pages -> {config.DEGRADED}")


if __name__ == "__main__":
    main()

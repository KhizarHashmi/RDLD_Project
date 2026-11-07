"""
p2_generate_cache.py -- P2 (Alishba, Workstream A): build the degraded-image cache.

Writes every page under every configuration once, so the detectors (Workstream B)
read images instead of regenerating them:

  RDLD_data/degraded/configs.csv            the 109 configurations
  RDLD_data/degraded/<config>/<page>.png    300 pages each (file names = clean corpus)

Singles are generated first; a pair A+B then starts from the cached single-A
image and applies B. Because every call is seeded per (degradation, severity,
page), that is byte-identical to applying A then B from scratch -- verified by
`--check`. Resumable: existing files are skipped.

Run:
  python scripts/p2_generate_cache.py              # full cache, all cores but 2
  python scripts/p2_generate_cache.py --check      # equivalence + determinism check only
  N_PAGES=20 python scripts/p2_generate_cache.py   # pilot on the first 20 pages
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from rdld import config
from rdld.doclaynet import load_coco

PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 4) - 2)))


def _job(args):
    from rdld import degradations as D
    cfg, kind, a, sev, page_id, fname = args
    dst = config.DEGRADED / cfg / fname
    if dst.exists():
        return cfg, 0.0, "skip"
    t0 = time.time()
    if kind == "single":
        img = D.apply_one(cv2.imread(str(config.IMAGES / fname)), a, sev, page_id)
    else:                                    # pair: start from cached single A
        first = config.DEGRADED / D.config_id([(a, sev)]) / fname
        img = cv2.imread(str(first))
        if img is None:
            raise FileNotFoundError(first)
        b = [n for n, _ in D.steps_of(cfg) if n != a][0]
        img = D.apply_one(img, b, sev, page_id)
    cv2.imwrite(str(dst), img, PNG)
    return cfg, time.time() - t0, "done"


def check(images):
    """Cached-pair route == from-scratch route, and two runs are identical."""
    from rdld import degradations as D
    im = images[0]
    clean = cv2.imread(str(config.IMAGES / im["file_name"]))
    bad = 0
    for row in D.experiment_configs():
        if row["kind"] != "pair" or row["severity"] != 2:
            continue
        x = D.apply_config(clean, row["config"], im["id"])
        y = D.apply_one(D.apply_one(clean, row["a"], 2, im["id"]), row["b"], 2, im["id"])
        z = D.apply_config(clean, row["config"], im["id"])
        ok = np.array_equal(x, y) and np.array_equal(x, z)
        bad += not ok
        if not ok:
            print(f"  [FAIL] {row['config']}")
    print(f"  equivalence + determinism: {28 - bad}/28 severity-2 pairs identical")
    return bad == 0


def main():
    from rdld import degradations as D
    gt = load_coco()
    images = gt["images"][: int(os.getenv("N_PAGES", len(gt["images"])))]
    configs = pd.DataFrame(D.experiment_configs())
    config.DEGRADED.mkdir(parents=True, exist_ok=True)
    configs.to_csv(config.DEGRADED / "configs.csv", index=False)

    print("=" * 70)
    print(f" P2 DEGRADED CACHE  |  {len(configs) - 1} configs x {len(images)} pages  |  {WORKERS} workers")
    print("=" * 70)
    if not check(images):
        raise SystemExit("composition is not reproducible -- fix before generating")
    if "--check" in sys.argv:
        return

    timing = {}
    for kind in ("single", "pair"):
        rows = configs[configs.kind == kind]
        for c in rows.config:
            (config.DEGRADED / c).mkdir(exist_ok=True)
        jobs = [(r.config, kind, r.a, int(r.severity), im["id"], im["file_name"])
                for r in rows.itertuples() for im in images]
        t0 = time.time()
        with Pool(WORKERS) as pool:
            for i, (cfg, dt, status) in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
                if status == "done":
                    timing.setdefault(cfg, []).append(dt)
                if i % 2000 == 0 or i == len(jobs):
                    print(f"  {kind:6s} {i:6d}/{len(jobs)}  {time.time() - t0:7.0f}s")

    # per-config generation cost (seconds of CPU per page)
    T = pd.DataFrame([{"config": c, "pages": len(v), "sec_per_page": float(np.mean(v))}
                      for c, v in timing.items()])
    if len(T):
        out = config.out_dir("p2", "generation")
        T.merge(configs, on="config").to_csv(out / "generation_timing.csv", index=False)
    n = sum(1 for _ in config.DEGRADED.glob("*/*.png"))
    gb = sum(f.stat().st_size for f in config.DEGRADED.glob("*/*.png")) / 1e9
    print(f"\n  cache: {n} images, {gb:.1f} GB -> {config.DEGRADED}")


if __name__ == "__main__":
    main()

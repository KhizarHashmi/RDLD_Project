"""
p4_seed_replicate.py -- P4 (Alishba, Workstream A): a second degradation seed.

The page bootstrap captures sampling of pages, not the randomness inside the
degradations themselves (where the watermark lands, glare vs. shadow, fibre layout,
blob positions). This regenerates every severity-2 single and pair with replicate
seed rep=1, so Workstream B can measure how much the interaction estimates move
when only the degradation randomness changes.

  RDLD_data/degraded/configs_rep.csv
  RDLD_data/degraded/r1-<config>/<page>.png

Run:
  python scripts/p4_seed_replicate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from multiprocessing import Pool

import cv2
import pandas as pd

from rdld import config
from rdld.doclaynet import load_coco

SEV, REP = 2, 1
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 4) - 2)))


def _job(args):
    from rdld import degradations as D
    cfg, kind, a, page_id, fname = args
    dst = config.DEGRADED / f"r{REP}-{cfg}" / fname
    if dst.exists():
        return
    if kind == "single":
        img = D.apply_one(cv2.imread(str(config.IMAGES / fname)), a, SEV, page_id, REP)
    else:
        img = cv2.imread(str(config.DEGRADED / f"r{REP}-{D.config_id([(a, SEV)])}" / fname))
        b = [n for n, _ in D.steps_of(cfg) if n != a][0]
        img = D.apply_one(img, b, SEV, page_id, REP)
    cv2.imwrite(str(dst), img, PNG)


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    sel = configs[(configs.severity == SEV) & configs.kind.isin(["single", "pair"])].copy()
    rep = sel.assign(base=sel.config, config="r1-" + sel.config)
    rep.to_csv(config.DEGRADED / "configs_rep.csv", index=False)
    images = load_coco()["images"]
    for kind in ("single", "pair"):
        rows = sel[sel.kind == kind]
        for c in rows.config:
            (config.DEGRADED / f"r{REP}-{c}").mkdir(exist_ok=True)
        jobs = [(r.config, kind, r.a, im["id"], im["file_name"]) for r in rows.itertuples() for im in images]
        with Pool(WORKERS) as pool:
            for i, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
                if i % 3000 == 0 or i == len(jobs):
                    print(f"  {kind} {i}/{len(jobs)}")
    print(f"replicate rep={REP}: {len(sel)} configs x {len(images)} pages")


if __name__ == "__main__":
    main()

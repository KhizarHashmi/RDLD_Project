"""
p3_jackknife.py -- P3 (Khizar, Workstream B): leave-one-page-out tables for BCa intervals.

Percentile bootstrap intervals are lopsided for some statistics (for example the tau gap).
Bias-corrected and accelerated (BCa) intervals fix this, and need the acceleration constant,
which comes from the jackknife: the statistic recomputed with each page left out in turn.
This script builds those tables in the same layout as p3_bootstrap.py, so every analysis can
compute its statistic on them exactly as it does on the bootstrap rows.

Output: RDLD_data/results/p3/jack/<model>/<config>.npy   float32 (n_pages, 11)
        row i = per-class AP with page i left out; columns = DocLayNet classes 1..11

Run:
  python scripts/p3_jackknife.py
  CONFIGS=RDLD_data/degraded/configs_order.csv python scripts/p3_jackknife.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rdld import config, detectors

WORKERS = int(os.getenv("WORKERS", "14"))
CONFIGS = Path(os.getenv("CONFIGS", config.DEGRADED / "configs.csv"))
MODELS = os.getenv("MODELS", ",".join(detectors.MODELS)).split(",")
JACK = config.out_dir("p3", "jack")
_gt = _idx = None


def _job(args):
    global _gt, _idx
    from rdld import fastcoco
    from rdld.doclaynet import load_coco
    if _gt is None:
        _gt = load_coco()
        n = len(_gt["images"])
        _idx = [np.delete(np.arange(n), i) for i in range(n)]
    model, cfg = args
    dst = JACK / model / f"{cfg}.npy"
    if dst.exists():
        return model, cfg, 0.0
    t0 = time.time()
    dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    pe = fastcoco.PerImageEval(_gt, dets)
    np.save(dst, np.array([pe.class_ap(s) for s in _idx], dtype=np.float32))
    return model, cfg, time.time() - t0


def main():
    configs = pd.read_csv(CONFIGS).config.tolist()
    for m in MODELS:
        (JACK / m).mkdir(exist_ok=True)
    jobs = [(m, c) for m in MODELS for c in configs
            if (config.PREDICTIONS / m / f"{c}.json").exists() and not (JACK / m / f"{c}.npy").exists()]
    print(f"jackknife tables to build: {len(jobs)}  ({WORKERS} workers)")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (m, c, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i:4d}/{len(jobs)}  last {m}/{c} {dt:.0f}s   elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

"""
p3_bootstrap.py -- P3 (Khizar, Workstream B): paired page-level bootstrap tables.

For every detector x configuration with a prediction file, compute per-class AP on
the full corpus and on N_BOOT page resamples. The resample index matrix is drawn
once from a fixed seed and shared by every configuration, so any difference
between two configurations (and the interaction term, a difference of four) is a
PAIRED statistic: page sampling noise cancels instead of adding up.

Output: RDLD_data/results/p3/boot/<model>/<config>.npy   float32 (N_BOOT+1, 11)
        row 0 = full corpus, rows 1.. = resamples; columns = DocLayNet classes 1..11
        RDLD_data/results/p3/boot/resample_index.npy      (N_BOOT, 300) page indices

Run:
  python scripts/p3_bootstrap.py
  CONFIGS=RDLD_data/degraded/configs_order.csv python scripts/p3_bootstrap.py
  MODELS=yolo,heron WORKERS=6 python scripts/p3_bootstrap.py
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

N_BOOT = int(os.getenv("N_BOOT", "1000"))
WORKERS = int(os.getenv("WORKERS", "14"))
CONFIGS = Path(os.getenv("CONFIGS", config.DEGRADED / "configs.csv"))
MODELS = os.getenv("MODELS", ",".join(detectors.MODELS)).split(",")
BOOT = config.out_dir("p3", "boot")
_gt = _idx = None


def _job(args):
    global _gt, _idx
    from rdld import fastcoco
    from rdld.doclaynet import load_coco
    if _gt is None:
        _gt, _idx = load_coco(), np.load(BOOT / "resample_index.npy")
    model, cfg = args
    dst = BOOT / model / f"{cfg}.npy"
    if dst.exists():
        return model, cfg, 0.0
    t0 = time.time()
    dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    pe = fastcoco.PerImageEval(_gt, dets)
    rows = [pe.class_ap()] + [pe.class_ap(s) for s in _idx]
    np.save(dst, np.array(rows, dtype=np.float32))
    return model, cfg, time.time() - t0


def main():
    idx_path = BOOT / "resample_index.npy"
    if not idx_path.exists():
        np.save(idx_path, np.random.default_rng(20261103).integers(0, 300, size=(N_BOOT, 300)))
    configs = pd.read_csv(CONFIGS).config.tolist()
    for m in MODELS:
        (BOOT / m).mkdir(exist_ok=True)
    jobs = [(m, c) for m in MODELS for c in configs
            if (config.PREDICTIONS / m / f"{c}.json").exists() and not (BOOT / m / f"{c}.npy").exists()]
    print(f"bootstrap tables to build: {len(jobs)}  ({N_BOOT} resamples, {WORKERS} workers)")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (m, c, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i:4d}/{len(jobs)}  last {m}/{c} {dt:.0f}s   elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

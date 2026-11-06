"""
p2_run_inference.py -- P2/P3 (Khizar, Workstream B): run the 3 detectors over the
degraded-image cache.

Reads RDLD_data/degraded/configs.csv (written by Workstream A) and, for every
detector x configuration, writes RDLD_data/predictions/<model>/<config>.json in
COCO results format. Resumable: existing prediction files are skipped, and the
clean predictions from P1 are reused (same detector wrapper, same score floor).
Image loading runs on a thread pool so the GPU is not left waiting on PNG decode.

Run:
  python scripts/p2_run_inference.py                 # all models, all configs
  MODELS=heron python scripts/p2_run_inference.py    # one model
  N_PAGES=20 python scripts/p2_run_inference.py      # pilot (writes to predictions_pilot/)
  CONFIGS=RDLD_data/degraded/configs_order.csv python scripts/p2_run_inference.py   # extra config list
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import torch
from PIL import Image

from rdld import config, detectors
from rdld.doclaynet import load_coco

BATCH = 8
N_PAGES = int(os.getenv("N_PAGES", "0"))
PRED = config.PREDICTIONS if not N_PAGES else config.DATA_ROOT / "predictions_pilot"
MODELS = os.getenv("MODELS", ",".join(detectors.MODELS)).split(",")
CONFIGS = Path(os.getenv("CONFIGS", config.DEGRADED / "configs.csv"))


def compact(d):
    return {"image_id": d["image_id"], "category_id": d["category_id"],
            "bbox": [round(v, 2) for v in d["bbox"]], "score": round(d["score"], 5),
            "raw_label": d["raw_label"]}


def main():
    gt = load_coco()
    images = gt["images"][:N_PAGES] if N_PAGES else gt["images"]
    configs = pd.read_csv(CONFIGS).config.tolist()
    pool = ThreadPoolExecutor(6)
    timing = []

    for key in MODELS:
        out = PRED / key
        out.mkdir(parents=True, exist_ok=True)
        todo = [c for c in configs if not (out / f"{c}.json").exists()]
        print(f"\n=== {key}: {len(todo)}/{len(configs)} configs to run")
        if not todo:
            continue
        det = detectors.load(key)

        def load(cfg, ims):
            root = config.IMAGES if cfg == "clean" else config.DEGRADED / cfg
            return [Image.open(root / im["file_name"]).convert("RGB") for im in ims]

        t_all = time.time()
        for n, cfg in enumerate(todo, 1):
            t0 = time.time()
            bs = getattr(det, "batch", BATCH)
            batches = [images[i:i + bs] for i in range(0, len(images), bs)]
            futures = [pool.submit(load, cfg, b) for b in batches[:2]]
            dets = []
            for i, b in enumerate(batches):
                pil = futures[i].result()
                if i + 2 < len(batches):
                    futures.append(pool.submit(load, cfg, batches[i + 2]))
                for p, im in zip(det.predict(pil), b):
                    dets.extend(compact(d) for d in detectors.to_coco_results(p, im["id"]))
            (out / f"{cfg}.json").write_text(json.dumps(dets), encoding="utf-8")
            dt = time.time() - t0
            timing.append({"model": key, "config": cfg, "sec": dt, "ms_per_page": dt / len(images) * 1000})
            if n % 10 == 0 or n == len(todo):
                print(f"    {n:3d}/{len(todo)}  {cfg:40s} {dt:5.1f}s  (elapsed {time.time() - t_all:6.0f}s)")
        del det
        torch.cuda.empty_cache()

    if timing:
        tdir = config.out_dir("p2", "inference")
        f = tdir / "inference_timing.csv"
        pd.DataFrame(timing).to_csv(f, mode="a", header=not f.exists(), index=False)


if __name__ == "__main__":
    main()

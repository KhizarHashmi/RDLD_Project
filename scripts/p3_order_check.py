"""
p3_order_check.py -- P3 (Alishba, Workstream A): does composition ORDER matter?

The main experiment applies every pair in a fixed physical order (paper -> print ->
overlay -> lighting -> optics). This script builds the same 28 severity-2 pairs in
the REVERSED order (B first, then A), with the same seeds, so the only difference
from the main cache is the order of application.

  RDLD_data/degraded/configs_order.csv
  RDLD_data/degraded/rev-<config>/<page>.png

The images are then scored by Workstream B (p2_run_inference.py and p3_bootstrap.py
with CONFIGS=.../configs_order.csv), and `--analyse` compares them here:
image-level (MS-SSIM between forward and reversed images) and detection-level
(paired bootstrap of mAP_reversed - mAP_forward).

Run:
  python scripts/p3_order_check.py              # generate reversed-order cache
  python scripts/p3_order_check.py --analyse    # after detection + bootstrap tables exist
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

SEV = 2
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]
WORKERS = int(os.getenv("WORKERS", max(1, (os.cpu_count() or 4) - 2)))
OUT = config.out_dir("p3", "order")


def _job(args):
    from rdld import degradations as D
    cfg, a, b, page_id, fname = args
    dst = config.DEGRADED / f"rev-{cfg}" / fname
    if dst.exists():
        return
    img = cv2.imread(str(config.DEGRADED / D.config_id([(b, SEV)]) / fname))   # B first (cached single)
    cv2.imwrite(str(dst), D.apply_one(img, a, SEV, page_id), PNG)              # then A


def generate():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    pairs = configs[(configs.kind == "pair") & (configs.severity == SEV)]
    rows = [{"config": f"rev-{r.config}", "kind": "pair_reversed", "severity": SEV, "a": r.b, "b": r.a,
             "forward": r.config} for r in pairs.itertuples()]
    pd.DataFrame(rows).to_csv(config.DEGRADED / "configs_order.csv", index=False)
    images = load_coco()["images"]
    jobs = []
    for r in pairs.itertuples():
        (config.DEGRADED / f"rev-{r.config}").mkdir(exist_ok=True)
        jobs += [(r.config, r.a, r.b, im["id"], im["file_name"]) for im in images]
    with Pool(WORKERS) as pool:
        for i, _ in enumerate(pool.imap_unordered(_job, jobs, chunksize=4), 1):
            if i % 2000 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    print(f"reversed-order cache: {len(pairs)} configs x {len(images)} pages")


def analyse():
    import torch
    from rdld.image_quality import all_metrics
    order = pd.read_csv(config.DEGRADED / "configs_order.csv")
    images = load_coco()["images"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # image level: how different are forward and reversed images? (sample 60 pages)
    rows = []
    for r in order.itertuples():
        sims, same = [], 0
        for im in images[::5]:
            f = cv2.imread(str(config.DEGRADED / r.forward / im["file_name"]))
            v = cv2.imread(str(config.DEGRADED / r.config / im["file_name"]))
            t = lambda x: torch.from_numpy(x[None]).to(dev).permute(0, 3, 1, 2).float()
            sims.append(float(all_metrics(t(f), t(v))["ms_ssim"][0]))
            same += bool(np.array_equal(f, v))
        rows.append({"forward": r.forward, "reversed": r.config,
                     "ms_ssim_fwd_vs_rev": float(np.mean(sims)), "frac_pixel_identical": same / len(sims)})
    IMG = pd.DataFrame(rows)

    # detection level: paired bootstrap of mAP difference (tables from Workstream B)
    boot = config.RESULTS / "p3" / "boot"
    det = []
    for m in sorted(p.name for p in boot.iterdir() if p.is_dir()):
        for r in order.itertuples():
            fa, ra = boot / m / f"{r.forward}.npy", boot / m / f"{r.config}.npy"
            if not (fa.exists() and ra.exists()):
                continue
            with np.errstate(all="ignore"):
                f, v = np.nanmean(np.load(fa), axis=1), np.nanmean(np.load(ra), axis=1)
            d = 100 * (v - f)
            lo, hi = np.percentile(d[1:], [2.5, 97.5])
            det.append({"model": m, "forward": r.forward, "delta_mAP_rev_minus_fwd": float(d[0]),
                        "lo": float(lo), "hi": float(hi), "significant": bool(lo > 0 or hi < 0)})
    DET = pd.DataFrame(det)
    T = IMG.merge(DET, on="forward", how="left")
    T.to_csv(OUT / "order_effect.csv", index=False)

    print("image level (MS-SSIM forward vs reversed, 60 pages):")
    print(IMG.sort_values("ms_ssim_fwd_vs_rev").round(4).to_string(index=False))
    if len(DET):
        print("\ndetection level (reversed - forward, mAP points, 95% paired CI):")
        piv = DET.pivot(index="forward", columns="model", values="delta_mAP_rev_minus_fwd")
        print(piv.round(2).to_string())
        print(f"\nsignificant order effects: {int(DET.significant.sum())}/{len(DET)}; "
              f"median |delta| {DET.delta_mAP_rev_minus_fwd.abs().median():.2f} pts")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    analyse() if "--analyse" in sys.argv else generate()

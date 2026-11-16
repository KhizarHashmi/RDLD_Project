"""
p2_evaluate.py -- P2/P3 (Khizar, Workstream B): score every detector x configuration.

For each prediction file in RDLD_data/predictions/<model>/<config>.json:
  mAP@[.5:.95], mAP@.5, mAP@.75, AP small/medium/large, per-class AP
Derived per row:
  D_pts   = 100 * (mAP_clean - mAP_config)          absolute drop, mAP points
  D_rel   = D_pts / (100 * mAP_clean)               relative drop
And, when Workstream A's image-quality table exists, RoDLA's robustness score:
  RD_p    = mean_s (100 - mAP%_{s,p}) / mPE_{s,p}   (Eq. 4), mRD = mean_p RD_p
using the model-independent mPE_IQA. Cells with mPE < 1 (defocus@1 is a pixel
no-op) are excluded from RD because the ratio is undefined there.

Outputs: RDLD_data/results/p2/evaluation/{metrics_long.csv, mrd.csv}

Run:
  python scripts/p2_evaluate.py
  python scripts/p2_evaluate.py --mrd-only     # after iq_per_config.csv exists; reuses metrics_long.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import time
from multiprocessing import Pool

import pandas as pd

from rdld import config, detectors
from rdld.doclaynet import load_coco

WORKERS = int(os.getenv("WORKERS", "12"))
OUT = config.out_dir("p2", "evaluation")
_gt = None


def _score(args):
    global _gt
    from rdld import evaluation
    if _gt is None:
        _gt = load_coco()
    model, cfg = args
    dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    r = evaluation.coco_map(_gt, dets)
    row = {"model": model, "config": cfg, "n_dets": len(dets)}
    row.update({k: r[k] for k in ("mAP", "mAP50", "mAP75", "AP_small", "AP_medium", "AP_large")})
    row.update({f"AP_{c}": v for c, v in r["per_class"].items()})
    return row


def rd_table(M, iq):
    """RoDLA Eq. 4 with mPE_IQA, per model and single degradation."""
    s = M[M.kind == "single"].merge(iq[["config", "mPE_iqa"]], on="config")
    s = s[s.mPE_iqa >= 1.0]
    s["err_over_mpe"] = (100 - 100 * s.mAP) / s.mPE_iqa
    rd = s.groupby(["model", "a"]).err_over_mpe.mean().rename("RD").reset_index()
    mrd = rd.groupby("model").RD.mean().rename("mRD").reset_index()
    return rd, mrd


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    if "--mrd-only" in sys.argv:                      # reuse scores, only (re)compute mRD
        M = pd.read_csv(OUT / "metrics_long.csv")
        rd, mrd = rd_table(M, pd.read_csv(config.RESULTS / "p2" / "image_quality" / "iq_per_config.csv"))
        rd.to_csv(OUT / "rd_per_degradation.csv", index=False)
        mrd.to_csv(OUT / "mrd.csv", index=False)
        print("RD per degradation:\n", rd.pivot(index="a", columns="model", values="RD").round(1).to_string())
        print("\nmRD (RoDLA Eq. 4, mPE_IQA; lower = more robust):\n", mrd.round(1).to_string(index=False))
        return
    jobs = [(m, c) for m in detectors.MODELS for c in configs.config
            if (config.PREDICTIONS / m / f"{c}.json").exists()]
    print(f"scoring {len(jobs)} detector x config pairs on {WORKERS} workers")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        rows = list(pool.imap_unordered(_score, jobs, chunksize=2))
    print(f"  done in {time.time() - t0:.0f}s")

    M = pd.DataFrame(rows).merge(configs, on="config")
    clean = M[M.config == "clean"].set_index("model").mAP
    M["mAP_clean"] = M.model.map(clean)
    M["D_pts"] = 100 * (M.mAP_clean - M.mAP)
    M["D_rel"] = M.D_pts / (100 * M.mAP_clean)
    M = M.sort_values(["model", "kind", "severity", "config"])
    M.to_csv(OUT / "metrics_long.csv", index=False)

    iq_path = config.RESULTS / "p2" / "image_quality" / "iq_per_config.csv"
    if iq_path.exists():
        rd, mrd = rd_table(M, pd.read_csv(iq_path))
        rd.to_csv(OUT / "rd_per_degradation.csv", index=False)
        mrd.to_csv(OUT / "mrd.csv", index=False)
        print("\nmRD (RoDLA Eq. 4, mPE_IQA; lower = more robust):\n", mrd.round(1).to_string(index=False))

    piv = M[M.kind == "single"].pivot_table(index=["a"], columns=["model", "severity"], values="D_pts")
    print("\nsingle-degradation mAP drop (points):\n", piv.round(1).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

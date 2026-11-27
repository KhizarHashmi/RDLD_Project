"""
p8_geometric_interaction.py -- P8 (Khizar, Workstream B): do pairs compound when one member is geometric?

p8_geometric_pairs.py (Alishba) builds, on the test pages at severity 2, two geometric singles
(rotation@2, keystoning@2) and the sixteen pairs of each with one of the eight image degradations.
Geometric degradations move the regions, so every condition folder carries its own annotations.json.
This script

  1. builds the paired page-bootstrap tables of those conditions against their own ground truth,
     with the same resample index as every other table (p3_bootstrap.py, written to p3/boot/),
  2. scores an equivariant reference: the clean predictions moved by the same transform as the page
     and scored against the moved ground truth. A detector that followed the page perfectly would
     score exactly this, so the gap between it and the clean score comes from the upright boxes
     around turned regions, not from the detector (here the gap is a gain, see below),
  3. computes for every detector and pair
       I_add  = D(A+G) - D(A) - D(G)
       I_mult = 100 * [R(A) R(G) - R(A+G)]
     with page-bootstrap intervals and two-sided bootstrap p values, Benjamini-Hochberg over the
     sixteen pairs of each detector, as p3_rq1_interaction.py does for the main pairs, and
       I_frame = 100 * [R(A) R'(G) - R'(A+G)],  R' = mAP / equivariant mAP
     which measures the geometric arm against what a detector that followed the page would score.
     Upright boxes around turned regions are larger, so they make the IoU easier to pass: on the
     test pages the equivariant score is above the clean score (Heron 76.5 vs 65.7 under rotation).
     I_mult and I_frame agreeing in sign means the answer does not hinge on that box effect.

Outputs: <data root>/results/p8/geometric/{geom_interactions,geom_singles,geom_summary,geom_by_degradation}.csv

Run (after p8_geometric_pairs.py and inference with CONFIGS=<data root>/degraded/configs_geom.csv):
  RDLD_DATA=../RDLD_data_test python scripts/p8_geometric_interaction.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import time
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

from rdld import config, detectors, intervals

OUT = config.out_dir("p8", "geometric")
BOOT = config.RESULTS / "p3" / "boot"
EQUIV = config.out_dir("p8", "geometric", "equiv_boot")
WORKERS = int(os.getenv("WORKERS", "10"))
ALPHA = 0.05
_idx = None


def gt_of(cfg):
    return json.loads((config.DEGRADED / cfg / "annotations.json").read_text(encoding="utf-8"))


def moved(dets, transforms):
    """Clean predictions moved with their page: four corners through the page matrix, upright box around them."""
    out = []
    for d in dets:
        t = transforms[str(d["image_id"])]
        x, y, w, h = d["bbox"]
        pts = cv2.perspectiveTransform(np.float32([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]]),
                                       np.array(t["M"]))[0]
        x0, y0 = np.clip(pts.min(axis=0), 0, t["out_size"])
        x1, y1 = np.clip(pts.max(axis=0), 0, t["out_size"])
        out.append({**d, "bbox": [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]})
    return out


def _table(args):
    """Per-class AP on the full set and on every resample, against the condition's own ground truth."""
    global _idx
    from rdld import fastcoco
    if _idx is None:
        _idx = np.load(BOOT / "resample_index.npy")
    model, cfg, kind, dst = args
    t0 = time.time()
    gt = gt_of(cfg)
    if kind == "equiv":
        dets = json.loads((config.PREDICTIONS / model / "clean.json").read_text(encoding="utf-8"))
        dets = moved(dets, json.loads((config.DEGRADED / cfg / "transforms.json").read_text(encoding="utf-8")))
    else:
        dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    pe = fastcoco.PerImageEval(gt, dets)
    rows = [pe.class_ap()] + [pe.class_ap(s[s >= 0]) for s in _idx]
    np.save(dst, np.array(rows, dtype=np.float32))
    return model, cfg, kind, time.time() - t0


def build_tables(C):
    jobs = []
    for m in detectors.MODELS:
        (EQUIV / m).mkdir(exist_ok=True)
        for cfg in C.config:
            dst = BOOT / m / f"{cfg}.npy"
            if not dst.exists():
                if not (config.PREDICTIONS / m / f"{cfg}.json").exists():
                    raise SystemExit(f"missing predictions {m}/{cfg}: run inference first")
                jobs.append((m, cfg, "pred", dst))
        for cfg in C[C.kind == "geom_single"].config:
            dst = EQUIV / m / f"{cfg}.npy"
            if not dst.exists():
                jobs.append((m, cfg, "equiv", dst))
    print(f"bootstrap tables to build: {len(jobs)} on {WORKERS} workers")
    t0 = time.time()
    with Pool(WORKERS) as pool:
        for i, (m, c, k, dt) in enumerate(pool.imap_unordered(_table, jobs), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"  {i:4d}/{len(jobs)}  last {m}/{c} ({k}) {dt:.0f}s   elapsed {time.time() - t0:.0f}s")


def mAP_rows(rows):
    with np.errstate(all="ignore"):
        return np.nanmean(rows, axis=1)[:1 + intervals.n_boot()]


def mAP(model, cfg):
    """Full-set row and bootstrap rows only (the geometric conditions have no jackknife tables)."""
    return mAP_rows(intervals.load_rows(model, cfg))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1)
    return out


def main():
    C = pd.read_csv(config.DEGRADED / "configs_geom.csv").fillna("")
    build_tables(C)
    same = pd.read_csv(config.RESULTS / "p3" / "rq1" / "rq1_interactions.csv")
    sev = int(C.severity.iloc[0])

    singles = []
    for m in detectors.MODELS:
        clean = mAP(m, "clean")
        for g in C[C.kind == "geom_single"].config:
            x = mAP(m, g)
            e = mAP_rows(np.load(EQUIV / m / f"{g}.npy"))
            d_pt, d_lo, d_hi = intervals.percentile_ci(100 * (clean - x))
            e_pt, e_lo, e_hi = intervals.percentile_ci(100 * (clean - e))
            singles.append({"model": m, "config": g, "mAP_clean": 100 * clean[0], "mAP": 100 * x[0],
                            "D": d_pt, "D_lo": d_lo, "D_hi": d_hi,
                            "D_equivariant": e_pt, "D_equivariant_lo": e_lo, "D_equivariant_hi": e_hi,
                            "D_beyond_equivariant": d_pt - e_pt})
    S = pd.DataFrame(singles)
    S.round(3).to_csv(OUT / "geom_singles.csv", index=False)

    rows = []
    for m in detectors.MODELS:
        clean = mAP(m, "clean")
        for r in C[C.kind == "geom_pair"].itertuples():
            xa, xb, xab = mAP(m, r.single_a), mAP(m, r.single_b), mAP(m, r.config)
            da, db, dab = 100 * (clean - xa), 100 * (clean - xb), 100 * (clean - xab)
            row = {"model": m, "config": r.config, "geom": r.geom, "image_deg": r.image_deg, "a": r.a, "b": r.b,
                   "D_a": float(da[0]), "D_b": float(db[0]), "D_ab": float(dab[0])}
            # frame-corrected: on a moved page the reference is the equivariant score, not the clean score
            g_cfg = f"{r.geom}@{sev}"
            e = mAP_rows(np.load(EQUIV / m / f"{g_cfg}.npy"))
            xi, xg = (xa, xb) if r.b == r.geom else (xb, xa)
            frame = 100 * ((xi / clean) * (xg / e) - xab / e)
            for name, v in (("add", dab - da - db), ("mult", 100 * ((xa / clean) * (xb / clean) - xab / clean)),
                            ("frame", frame)):
                pt, lo, hi = intervals.percentile_ci(v)
                row.update({f"I_{name}": pt, f"I_{name}_lo": lo, f"I_{name}_hi": hi, f"p_{name}": intervals.boot_p(v)})
            rows.append(row)
    R = pd.DataFrame(rows)
    for m in detectors.MODELS:
        k = R.model == m
        for name in ("add", "mult", "frame"):
            R.loc[k, f"q_{name}"] = bh(R.loc[k, f"p_{name}"])
    for name in ("add", "mult", "frame"):
        sig = R[f"q_{name}"] < ALPHA
        R[f"class_{name}"] = np.where(sig & (R[f"I_{name}"] > 0), "super",
                                      np.where(sig & (R[f"I_{name}"] < 0), "sub", "n.s."))
    R.to_csv(OUT / "geom_interactions.csv", index=False)

    # reference: the image-image pairs of the main experiment at the same severity, same pages, noncontrol
    ref = same[same.severity == sev]

    def summary(D, label):
        med = D.groupby("model").I_mult.median()
        out = {"set": label, "cells": len(D),
                "mult_super": int((D.class_mult == "super").sum()), "mult_sub": int((D.class_mult == "sub").sum()),
                "add_super": int((D.class_add == "super").sum()), "add_sub": int((D.class_add == "sub").sum()),
                "super_under_both_nulls": int(((D.class_mult == "super") & (D.class_add == "super")).sum()),
                "detectors_median_Imult_positive": f"{int((med > 0).sum())}/{len(med)}",
                "median_I_mult": round(float(D.I_mult.median()), 2), "mean_I_mult": round(float(D.I_mult.mean()), 2)}
        if "I_frame" in D:
            fmed = D.groupby("model").I_frame.median()
            out.update({"frame_super": int((D.class_frame == "super").sum()), "frame_sub": int((D.class_frame == "sub").sum()),
                        "detectors_median_Iframe_positive": f"{int((fmed > 0).sum())}/{len(fmed)}",
                        "median_I_frame": round(float(D.I_frame.median()), 2)})
        return out

    summ = [summary(R, "all geometric pairs")] + [summary(R[R.geom == g], f"with {g}") for g in sorted(R.geom.unique())]
    summ.append(summary(ref, f"main image pairs at severity {sev} (reference)"))
    SU = pd.DataFrame(summ)
    SU.to_csv(OUT / "geom_summary.csv", index=False)

    V = (R.groupby(["geom", "image_deg"])
          .agg(I_mult=("I_mult", "mean"), I_frame=("I_frame", "mean"), I_add=("I_add", "mean"),
               n_super=("class_mult", lambda s: int((s == "super").sum())),
               n_sub=("class_mult", lambda s: int((s == "sub").sum())))
          .round(2).reset_index())
    V.to_csv(OUT / "geom_by_degradation.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(S.groupby("config")[["D", "D_equivariant", "D_beyond_equivariant"]].describe().round(2).T.to_string())
        print()
        print(SU.to_string(index=False))
        print()
        print(V.to_string(index=False))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

"""
p8_detector_exclusion.py -- P8 (Khizar, Workstream B): do the conclusions depend on particular detectors?

Three detectors are open to a confound. RF-DETR was fine-tuned by us on one sixth of the
training pages, DETR may have seen validation-split pages in training, and the two YOLO
generations were trained for different lengths (100 vs 40 epochs). This script recomputes the
headline numbers of RQ1 and RQ3 with those detectors removed, and with each detector removed
in turn, on the current page set.

  RQ1  cells super- and sub-multiplicative (FDR 0.05), their ratio, cells super-additive under
       both nulls, and how many detectors have a positive median I_mult
  RQ3  tau gap (mean Kendall tau vs the clean order, singles minus pairs) with its 95% page
       percentile interval; agreement between the mRD order and the single-retention order

Controls (pairs and singles with defocus at severity 1) are left out, as in the paper.

Outputs: <data root>/results/p8/detector_exclusion/exclusion_summary.csv

Run:
  python scripts/p8_detector_exclusion.py
  RDLD_DATA=../RDLD_data_test python scripts/p8_detector_exclusion.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import combinations

import numpy as np
import pandas as pd

from rdld import config, detectors, intervals

OUT = config.out_dir("p8", "detector_exclusion")
NAMES = {"yolo": "YOLO26-m", "yolo26l": "YOLO26-l", "yolo11m": "YOLOv11-m", "yolo11l": "YOLOv11-l",
         "heron": "Heron", "heron101": "Heron-101", "egret": "Egret-m", "egretxl": "Egret-xl",
         "detr": "DETR", "ddetr": "Deformable DETR", "rfdetr": "RF-DETR"}


def load(m, c):
    with np.errstate(all="ignore"):
        return 100 * np.nanmean(intervals.load_rows(m, c), axis=1)


def tau_vs(A, B):
    idx = np.array(list(combinations(range(A.shape[0]), 2)))
    return (np.sign(A[idx[:, 0]] - A[idx[:, 1]]) * np.sign(B[idx[:, 0]] - B[idx[:, 1]])).mean(axis=0)


def tau(a, b):
    a, b = np.asarray(a)[:, None], np.asarray(b)[:, None]
    return float(tau_vs(a, b)[0])


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    models = list(detectors.MODELS)
    is_control = (configs.severity == 1) & ((configs.a == "defocus") | (configs.b == "defocus"))
    test = configs[(configs.kind != "clean") & ~is_control].reset_index(drop=True)
    single, pair = (test.kind == "single").to_numpy(), (test.kind == "pair").to_numpy()
    X = {(m, c): load(m, c) for m in models for c in ["clean"] + test.config.tolist()}

    I = pd.read_csv(config.RESULTS / "p3" / "rq1" / "rq1_interactions.csv")
    I = I[~((I.severity == 1) & ((I.a == "defocus") | (I.b == "defocus")))]
    mrd = pd.read_csv(config.RESULTS / "p2" / "evaluation" / "mrd.csv").set_index("model").mRD
    ret_single = {m: np.mean([X[(m, c)][0] / X[(m, "clean")][0] for c in test.config[single]]) for m in models}

    sets = {"all 11": models,
            "without RF-DETR": [m for m in models if m != "rfdetr"],
            "without DETR": [m for m in models if m != "detr"],
            "without RF-DETR and DETR": [m for m in models if m not in ("rfdetr", "detr")],
            "YOLO26 only of the YOLO models": [m for m in models if m not in ("yolo11m", "yolo11l")],
            "YOLOv11 only of the YOLO models": [m for m in models if m not in ("yolo", "yolo26l")]}
    sets.update({f"without {NAMES[m]}": [x for x in models if x != m] for m in models
                 if m not in ("rfdetr", "detr")})

    rows = []
    for name, ms in sets.items():
        clean = np.stack([X[(m, "clean")] for m in ms])
        T = np.stack([tau_vs(np.stack([X[(m, c)] for m in ms]), clean) for c in test.config])
        d = T[single].mean(axis=0) - T[pair].mean(axis=0)
        g, lo, hi = intervals.percentile_ci(d)
        S = I[I.model.isin(ms)]
        sup, sub = int((S.class_mult == "super").sum()), int((S.class_mult == "sub").sum())
        both = int(((S.class_mult == "super") & (S.class_add == "super")).sum())
        med = S.groupby("model").I_mult.median()
        rows.append({"detector_set": name, "n_detectors": len(ms), "cells": len(S),
                     "mult_super": sup, "mult_sub": sub, "super_to_sub": round(sup / max(sub, 1), 2),
                     "super_under_both_nulls": both,
                     "detectors_median_Imult_positive": f"{int((med > 0).sum())}/{len(med)}",
                     "tau_single": round(float(T[single, 0].mean()), 3), "tau_pair": round(float(T[pair, 0].mean()), 3),
                     "tau_gap": round(g, 3), "lo": round(lo, 3), "hi": round(hi, 3), "excludes_zero": bool(lo > 0 or hi < 0),
                     "tau_mrd_vs_single_retention": round(tau([-mrd[m] for m in ms], [ret_single[m] for m in ms]), 2)})
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "exclusion_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(R.to_string(index=False))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

"""
p8_cross_severity_interaction.py -- P8 (Khizar, Workstream C): do pairs still compound when the two
degradations are at different severities?

p7_cross_severity.py (Alishba) builds 30 mixed-severity pairs: the five pairs with the largest mean
validation I_mult at severity 2, each at severities (1,2) (2,1) (1,3) (3,1) (2,3) (3,2). Each pair A@s
+ B@t is made by applying B@t to the cached single A@s, so the singles it is compared with are the
existing single conditions A@s and B@t. This script computes, for every detector and mixed pair,

  I_add  = D(A@s+B@t) - D(A@s) - D(B@t)
  I_mult = 100 * [R(A@s) R(B@t) - R(A@s+B@t)]

with page-bootstrap intervals and two-sided bootstrap p values, Benjamini-Hochberg over the 30 cells of
each detector, exactly as p3_rq1_interaction.py does for the same-severity pairs. It then compares each
mixed cell with the same pair at equal severity (s and t), because the five pairs were chosen for
strong interaction and only that comparison is fair.

Outputs: <data root>/results/p8/cross_severity/{xsev_interactions,xsev_summary,xsev_vs_same}.csv

Run (after inference and p3_bootstrap.py on configs_xsev.csv):
  RDLD_DATA=../RDLD_data_test python scripts/p8_cross_severity_interaction.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from rdld import config, detectors, intervals

OUT = config.out_dir("p8", "cross_severity")
ALPHA = 0.05


def mAP(model, cfg):
    """Full-corpus row and bootstrap rows only: the mixed pairs have no jackknife tables."""
    with np.errstate(all="ignore"):
        return np.nanmean(intervals.load_rows(model, cfg), axis=1)[:1 + intervals.n_boot()]


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
    X = pd.read_csv(config.DEGRADED / "configs_xsev.csv")
    same = pd.read_csv(config.RESULTS / "p3" / "rq1" / "rq1_interactions.csv")
    rows = []
    for m in detectors.MODELS:
        clean = mAP(m, "clean")
        for r in X.itertuples():
            xa, xb, xab = mAP(m, r.single_a), mAP(m, r.single_b), mAP(m, r.config)
            da, db, dab = 100 * (clean - xa), 100 * (clean - xb), 100 * (clean - xab)
            row = {"model": m, "config": r.config, "a": r.a, "b": r.b, "sev_a": r.sev_a, "sev_b": r.sev_b,
                   "D_a": float(da[0]), "D_b": float(db[0]), "D_ab": float(dab[0]),
                   "pred_multiplicative": float(100 * clean[0] * (1 - (xa[0] / clean[0]) * (xb[0] / clean[0])))}
            for name, v in (("add", dab - da - db), ("mult", 100 * ((xa / clean) * (xb / clean) - xab / clean))):
                pt, lo, hi = intervals.percentile_ci(v)
                row.update({f"I_{name}": pt, f"I_{name}_lo": lo, f"I_{name}_hi": hi, f"p_{name}": intervals.boot_p(v)})
            for s, tag in ((r.sev_a, "same_at_sev_a"), (r.sev_b, "same_at_sev_b")):
                ref = same[(same.model == m) & (same.severity == s) & (same.a == r.a) & (same.b == r.b)]
                row[f"I_mult_{tag}"] = float(ref.I_mult.iloc[0]) if len(ref) else np.nan
            rows.append(row)
    R = pd.DataFrame(rows)
    for m in detectors.MODELS:
        k = R.model == m
        for name in ("add", "mult"):
            R.loc[k, f"q_{name}"] = bh(R.loc[k, f"p_{name}"])
    for name in ("add", "mult"):
        sig = R[f"q_{name}"] < ALPHA
        R[f"class_{name}"] = np.where(sig & (R[f"I_{name}"] > 0), "super",
                                      np.where(sig & (R[f"I_{name}"] < 0), "sub", "n.s."))
    lo_ref = R[["I_mult_same_at_sev_a", "I_mult_same_at_sev_b"]].min(axis=1)
    hi_ref = R[["I_mult_same_at_sev_a", "I_mult_same_at_sev_b"]].max(axis=1)
    R["between_same_severity"] = (R.I_mult >= lo_ref) & (R.I_mult <= hi_ref)
    # defocus at severity 1 leaves the pixels unchanged: those cells are exact controls (I = 0),
    # kept in the BH family as in p3_rq1_interaction.py but left out of every summary
    R["control"] = ((R.a == "defocus") & (R.sev_a == 1)) | ((R.b == "defocus") & (R.sev_b == 1))
    R.to_csv(OUT / "xsev_interactions.csv", index=False)
    print(f"control cells: {int(R.control.sum())}, max |I_mult| {R[R.control].I_mult.abs().max():.4f}")
    R = R[~R.control]

    med = R.groupby("model").I_mult.median()
    summ = {"cells": len(R),
            "mult_super": int((R.class_mult == "super").sum()), "mult_sub": int((R.class_mult == "sub").sum()),
            "add_super": int((R.class_add == "super").sum()), "add_sub": int((R.class_add == "sub").sum()),
            "super_under_both_nulls": int(((R.class_mult == "super") & (R.class_add == "super")).sum()),
            "detectors_median_Imult_positive": f"{int((med > 0).sum())}/{len(med)}",
            "detectors_with_super_both": int(R[(R.class_mult == "super") & (R.class_add == "super")].model.nunique()),
            "mean_I_mult_xsev": round(R.I_mult.mean(), 2),
            "mean_I_mult_same_lower_sev": round(R[["I_mult_same_at_sev_a", "I_mult_same_at_sev_b"]].min(axis=1).mean(), 2),
            "mean_I_mult_same_higher_sev": round(R[["I_mult_same_at_sev_a", "I_mult_same_at_sev_b"]].max(axis=1).mean(), 2),
            "share_between_same_severity": round(R.between_same_severity.mean(), 3)}
    pd.DataFrame([summ]).to_csv(OUT / "xsev_summary.csv", index=False)

    V = (R.groupby(["a", "b", "sev_a", "sev_b"])
          .agg(I_mult=("I_mult", "mean"), same_a=("I_mult_same_at_sev_a", "mean"), same_b=("I_mult_same_at_sev_b", "mean"),
               n_super=("class_mult", lambda s: int((s == "super").sum())), n_sub=("class_mult", lambda s: int((s == "sub").sum())))
          .round(2).reset_index())
    V.to_csv(OUT / "xsev_vs_same.csv", index=False)

    for k, v in summ.items():
        print(f"{k:34s} {v}")
    print()
    print(V.to_string(index=False))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

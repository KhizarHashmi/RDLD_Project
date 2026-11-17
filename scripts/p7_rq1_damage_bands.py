"""
p7_rq1_damage_bands.py -- P7 (Khizar, Workstream B): is RQ1's compounding only a high-damage effect?

Pairs remove more accuracy than singles, so a reader may ask whether the positive multiplicative
interaction I_mult only appears where the two singles already do a lot of damage (for example
because mAP is bounded and the null fits badly near the floor). This script holds the predicted
damage fixed and looks at I_mult inside damage bands.

For every detector and noncontrol pair (defocus@1 pairs left out):
  predicted damage  P = 100 * (1 - R(A) * R(B))   percent of clean mAP the multiplicative null expects to lose
  interaction       I_mult = 100 * [R(A) * R(B) - R(A+B)]                                     (as in p3_rq1)
Cells are split into five bands by the quintiles of P on the full sample (bands fixed, pooled over
detectors). In each band the mean I_mult is computed on the full sample and on every shared bootstrap
resample, which gives a paired 95% percentile interval and a two-sided bootstrap p-value. The
Spearman correlation of P and I_mult over cells is reported with its interval as well.

Run once per data folder (page set x bootstrap scheme):
  python scripts/p7_rq1_damage_bands.py
  RDLD_DATA=../RDLD_data_test python scripts/p7_rq1_damage_bands.py
  RDLD_DATA=../RDLD_data_docboot python scripts/p7_rq1_damage_bands.py      (document bootstrap)

Outputs: RDLD_data/results/p7/rq1_damage/  damage_bands.csv, damage_bands_by_detector.csv, damage_corr.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from rdld import config, detectors, intervals

OUT = config.out_dir("p7", "rq1_damage")
N_BANDS = 5


def mAP(model, cfg):
    with np.errstate(all="ignore"):
        return np.nanmean(intervals.load_rows(model, cfg), axis=1)


def spearman_rows(x, y):
    """Spearman correlation per row; x, y: (rows, cells)."""
    rx = np.apply_along_axis(rankdata, 1, x)
    ry = np.apply_along_axis(rankdata, 1, y)
    rx -= rx.mean(1, keepdims=True)
    ry -= ry.mean(1, keepdims=True)
    return (rx * ry).sum(1) / np.sqrt((rx ** 2).sum(1) * (ry ** 2).sum(1))


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    pairs = configs[(configs.kind == "pair")]
    pairs = pairs[~((pairs.severity == 1) & ((pairs.a == "defocus") | (pairs.b == "defocus")))]
    P, I, meta = [], [], []
    for m in detectors.MODELS:
        clean = mAP(m, "clean")
        cache = {}

        def get(c):
            if c not in cache:
                cache[c] = mAP(m, c)
            return cache[c]

        for r in pairs.itertuples():
            ra, rb, rab = get(f"{r.a}@{r.severity}") / clean, get(f"{r.b}@{r.severity}") / clean, get(r.config) / clean
            P.append(100 * (1 - ra * rb))
            I.append(100 * (ra * rb - rab))
            meta.append((m, r.config, r.severity))
    n = min(len(v) for v in P + I)                   # same row count for every cell
    P = np.stack([v[:n] for v in P], 1)              # (rows, cells)
    I = np.stack([v[:n] for v in I], 1)
    meta = pd.DataFrame(meta, columns=["model", "config", "severity"])
    print(f"{config.DATA_ROOT.name}: {I.shape[1]} cells, {intervals.n_boot()} resamples")

    edges = np.quantile(P[0], np.linspace(0, 1, N_BANDS + 1))
    band = np.clip(np.searchsorted(edges, P[0], side="right") - 1, 0, N_BANDS - 1)
    rows, drows = [], []
    for k in range(N_BANDS):
        sel = band == k
        v = I[:, sel].mean(1)
        pt, lo, hi = intervals.percentile_ci(v)
        rows.append({"band": k + 1, "P_from": edges[k], "P_to": edges[k + 1], "n_cells": int(sel.sum()),
                     "P_mean": float(P[0, sel].mean()), "I_mult_mean": pt, "lo": lo, "hi": hi,
                     "p": intervals.boot_p(v), "share_cells_positive": float((I[0, sel] > 0).mean())})
        for m in detectors.MODELS:
            s2 = sel & (meta.model == m).to_numpy()
            if s2.any():
                drows.append({"band": k + 1, "model": m, "n_cells": int(s2.sum()), "I_mult_mean": float(I[0, s2].mean())})
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "damage_bands.csv", index=False)
    pd.DataFrame(drows).to_csv(OUT / "damage_bands_by_detector.csv", index=False)

    rho = spearman_rows(P, I)
    pt, lo, hi = intervals.percentile_ci(rho)
    C = pd.DataFrame([{"spearman_P_Imult": pt, "lo": lo, "hi": hi}])
    C.to_csv(OUT / "damage_corr.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(B.to_string(index=False))
        print(C.to_string(index=False))
        D = pd.DataFrame(drows)
        print("detectors with positive mean I_mult per band:",
              D.groupby("band").I_mult_mean.apply(lambda x: f"{(x > 0).sum()}/{len(x)}").to_dict())


if __name__ == "__main__":
    main()

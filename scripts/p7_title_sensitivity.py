"""
p7_title_sensitivity.py -- P7 (Khizar, Workstream B): do the findings depend on the Title class?

The 300 test pages hold only 6 Title boxes, yet Title counts for 1/11 of a macro-averaged mAP.
This recomputes the headline numbers with Title left out, from the saved bootstrap tables.
No inference is rerun.

Compared with and without Title (control pairs excluded from both):
  clean mAP and clean order, mean retention and its order, Kendall tau against the clean order,
  and the RQ1 interaction counts and medians.

Outputs: RDLD_data/results/p7/title_sensitivity/{clean_and_retention.csv, tau.csv, rq1.csv}

Run:
  python scripts/p7_title_sensitivity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from rdld import config, detectors
from rdld.doclaynet import CATEGORIES
from p3_rq1_interaction import bh, boot_stats

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p7", "title_sensitivity")
CLASSES = [CATEGORIES[i] for i in range(1, 12)]
TITLE = CLASSES.index("Title")
ALPHA = 0.05


def load(model, cfg, keep):
    a = np.load(BOOT / model / f"{cfg}.npy")[:, keep]
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=1)


def is_control(r):
    return r.severity == 1 and "defocus" in (r.a, r.b)


def evaluate(models, configs, keep):
    singles = [r for r in configs[configs.kind == "single"].itertuples() if not (r.severity == 1 and r.a == "defocus")]
    pairs = [r for r in configs[configs.kind == "pair"].itertuples() if not is_control(r)]
    table, rq1 = {}, []
    for m in models:
        clean = load(m, "clean", keep)
        cache = {}

        def get(c):
            if c not in cache:
                cache[c] = load(m, c, keep)
            return cache[c]

        table[m] = {"clean": 100 * clean[0]}
        for r in singles + pairs:
            table[m][r.config] = 100 * get(r.config)[0]
        rows = []
        for r in pairs:
            xa, xb, xab = get(f"{r.a}@{r.severity}"), get(f"{r.b}@{r.severity}"), get(r.config)
            i_add = 100 * (clean - xab) - 100 * (clean - xa) - 100 * (clean - xb)
            i_mult = 100 * ((xa / clean) * (xb / clean) - xab / clean)
            rows.append((boot_stats(i_add), boot_stats(i_mult)))
        for j, name in enumerate(("add", "mult")):
            pts = np.array([x[j][0] for x in rows])
            q = bh([x[j][3] for x in rows])
            rq1.append({"model": m, "null": name, "median_I": float(np.median(pts)),
                        "super": int(np.sum((q < ALPHA) & (pts > 0))), "sub": int(np.sum((q < ALPHA) & (pts < 0)))})
    T = pd.DataFrame(table)
    clean_order = T.loc["clean"]
    tau = {"single": [kendalltau(clean_order, T.loc[r.config]).statistic for r in singles],
           "pair": [kendalltau(clean_order, T.loc[r.config]).statistic for r in pairs]}
    pair_cfgs, single_cfgs = [r.config for r in pairs], [r.config for r in singles]
    ret = pd.DataFrame({"clean_mAP": clean_order,
                        "retention_single": (T.loc[single_cfgs] / clean_order).mean() * 100,
                        "retention_pair": (T.loc[pair_cfgs] / clean_order).mean() * 100})
    ret["clean_rank"] = ret.clean_mAP.rank(ascending=False).astype(int)
    ret["retention_rank_single"] = ret.retention_single.rank(ascending=False).astype(int)
    ret["retention_rank_pair"] = ret.retention_pair.rank(ascending=False).astype(int)
    return ret, {k: float(np.mean(v)) for k, v in tau.items()}, pd.DataFrame(rq1)


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    configs = configs[configs.kind.isin(["single", "pair"])]
    models = list(detectors.MODELS)
    with_title = list(range(len(CLASSES)))
    without = [i for i in with_title if i != TITLE]

    ra, ta, qa = evaluate(models, configs, with_title)
    rb, tb, qb = evaluate(models, configs, without)

    R = ra.join(rb, lsuffix="_all", rsuffix="_noTitle")
    R.to_csv(OUT / "clean_and_retention.csv", index_label="model")
    Q = qa.merge(qb, on=["model", "null"], suffixes=("_all", "_noTitle"))
    Q.to_csv(OUT / "rq1.csv", index=False)
    pd.DataFrame([{"classes": "all", **ta}, {"classes": "noTitle", **tb}]).to_csv(OUT / "tau.csv", index=False)

    print(R[["clean_mAP_all", "clean_mAP_noTitle", "clean_rank_all", "clean_rank_noTitle",
             "retention_pair_all", "retention_pair_noTitle",
             "retention_rank_pair_all", "retention_rank_pair_noTitle",
             "retention_rank_single_all", "retention_rank_single_noTitle"]].round(2).to_string())
    print("\nmean Kendall tau vs clean  all:", {k: round(v, 3) for k, v in ta.items()},
          " noTitle:", {k: round(v, 3) for k, v in tb.items()})
    print("\nRQ1 interactions:\n", Q.round(2).to_string(index=False))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

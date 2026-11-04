"""
p4_seed_stability.py -- P4 (Khizar, Workstream B): are the RQ1 interaction estimates
stable when only the degradation randomness changes?

Workstream A regenerated every severity-2 single and pair with a second seed
(r1-<config>). Here the interaction terms are recomputed on that replicate and
compared with the main run (seed 0):

  - agreement of I_mult across seeds (Spearman, mean |difference|)
  - seed-to-seed difference vs. the page-bootstrap CI half-width: if seed noise is
    small relative to page noise, the single-seed main experiment is adequate
  - does the super / sub / n.s. call (sign of a significant I) survive?
  - pooled two-seed estimate

Needs: predictions + bootstrap tables for configs_rep.csv
  CONFIGS=RDLD_data/degraded/configs_rep.csv python scripts/p2_run_inference.py
  CONFIGS=RDLD_data/degraded/configs_rep.csv python scripts/p3_bootstrap.py

Outputs: RDLD_data/results/p4/seeds/

Run:
  python scripts/p4_seed_stability.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rdld import config, detectors

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p4", "seeds")


def mAP(m, c):
    with np.errstate(all="ignore"):
        return 100 * np.nanmean(np.load(BOOT / m / f"{c}.npy"), axis=1)


def i_mult(m, a, b, ab, prefix=""):
    clean = mAP(m, "clean")
    xa, xb, xab = mAP(m, prefix + a), mAP(m, prefix + b), mAP(m, prefix + ab)
    return 100 * ((xa / clean) * (xb / clean) - xab / clean)


def main():
    rep = pd.read_csv(config.DEGRADED / "configs_rep.csv")
    pairs = rep[rep.kind == "pair"]
    rows = []
    for m in detectors.MODELS:
        for r in pairs.itertuples():
            a, b = f"{r.a}@2", f"{r.b}@2"
            i0 = i_mult(m, a, b, r.base)
            i1 = i_mult(m, a, b, r.base, prefix="r1-")
            lo0, hi0 = np.percentile(i0[1:], [2.5, 97.5])
            lo1, hi1 = np.percentile(i1[1:], [2.5, 97.5])
            pooled = (i0 + i1) / 2
            plo, phi = np.percentile(pooled[1:], [2.5, 97.5])
            call = lambda v, lo, hi: "super" if lo > 0 else ("sub" if hi < 0 else "n.s.")
            single_d0 = mAP(m, "clean")[0] - mAP(m, a)[0]
            single_d1 = mAP(m, "clean")[0] - mAP(m, "r1-" + a)[0]
            rows.append({"model": m, "config": r.base, "I_seed0": i0[0], "I_seed1": i1[0],
                         "diff": i1[0] - i0[0], "ci_halfwidth_seed0": (hi0 - lo0) / 2,
                         "call_seed0": call(i0[0], lo0, hi0), "call_seed1": call(i1[0], lo1, hi1),
                         "I_pooled": pooled[0], "pooled_lo": plo, "pooled_hi": phi,
                         "call_pooled": call(pooled[0], plo, phi),
                         "D_single_a_seed0": single_d0, "D_single_a_seed1": single_d1})
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "seed_comparison.csv", index=False)

    summ = []
    for m, g in T.groupby("model"):
        summ.append({"model": m, "n_pairs": len(g),
                     "spearman_I_seed0_vs_seed1": float(spearmanr(g.I_seed0, g.I_seed1).statistic),
                     "mean_abs_diff": float(g["diff"].abs().mean()),
                     "mean_ci_halfwidth": float(g.ci_halfwidth_seed0.mean()),
                     "ratio_seed_noise_to_page_noise": float(g["diff"].abs().mean() / g.ci_halfwidth_seed0.mean()),
                     "call_agreement": float((g.call_seed0 == g.call_seed1).mean()),
                     "opposite_significant_calls": int(((g.call_seed0 == "super") & (g.call_seed1 == "sub")).sum()
                                                       + ((g.call_seed0 == "sub") & (g.call_seed1 == "super")).sum())})
    SM = pd.DataFrame(summ)
    SM.to_csv(OUT / "seed_summary.csv", index=False)
    print("seed stability of severity-2 interaction terms (multiplicative null):")
    print(SM.round(3).to_string(index=False))
    print("\ncalls seed0 -> seed1:")
    print(pd.crosstab([T.model, T.call_seed0], T.call_seed1).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

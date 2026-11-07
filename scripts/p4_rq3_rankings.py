"""
p4_rq3_rankings.py -- P4 (Khizar, Workstream B): RQ3 -- are clean-condition detector
rankings preserved under (compound) degradation?

The bootstrap resample index is shared by every model and configuration, so model
gaps are paired statistics. Three views:

  1. Pairwise gaps. For each model pair and configuration, gap = mAP_m1 - mAP_m2
     with a 95% paired CI. A REVERSAL is a configuration whose gap is significant
     with the opposite sign to the (significant) clean gap.
  2. Rank agreement. Kendall tau between each configuration's model ordering and
     the clean ordering, and the share of bootstrap resamples reproducing the
     clean ordering.
  3. Benchmark disagreement. Robustness ranked by a single-degradation benchmark
     (RoDLA-style: mean relative retention over the 24 singles) vs a compound
     benchmark (mean retention over the 84 pairs). If the order differs, a
     single-degradation leaderboard would mislead.

Outputs: RDLD_data/results/p4/rq3/

Run:
  python scripts/p4_rq3_rankings.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdld import config, detectors

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p4", "rq3")
MODELS = list(detectors.MODELS)


def load(m, c):
    with np.errstate(all="ignore"):
        return 100 * np.nanmean(np.load(BOOT / m / f"{c}.npy"), axis=1)


def ci(v):
    lo, hi = np.percentile(v[1:], [2.5, 97.5])
    return float(v[0]), float(lo), float(hi)


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    X = {(m, c): load(m, c) for m in MODELS for c in configs.config}

    # ---------------------------------------------------------------- 1. pairwise gaps
    gaps = []
    for m1, m2 in combinations(MODELS, 2):
        g0, l0, h0 = ci(X[(m1, "clean")] - X[(m2, "clean")])
        clean_sign = np.sign(g0) if (l0 > 0 or h0 < 0) else 0
        for r in configs.itertuples():
            g, lo, hi = ci(X[(m1, r.config)] - X[(m2, r.config)])
            sig = lo > 0 or hi < 0
            gaps.append({"m1": m1, "m2": m2, "config": r.config, "kind": r.kind, "severity": r.severity,
                         "gap": g, "lo": lo, "hi": hi, "significant": sig,
                         "clean_gap": g0, "reversal": bool(sig and clean_sign != 0 and np.sign(g) == -clean_sign)})
    G = pd.DataFrame(gaps)
    G.to_csv(OUT / "pairwise_gaps.csv", index=False)

    # ---------------------------------------------------------------- 2. rank agreement
    clean_rank = np.argsort(np.argsort([-X[(m, "clean")][0] for m in MODELS]))
    orig = list(detectors.ORIGINAL)
    orig_clean = [X[(m, "clean")][0] for m in orig]
    clean_boot = np.stack([X[(m, "clean")][1:] for m in MODELS])
    ranks = []
    for r in configs.itertuples():
        full = [X[(m, r.config)][0] for m in MODELS]
        rk = np.argsort(np.argsort([-v for v in full]))
        b = np.stack([X[(m, r.config)][1:] for m in MODELS])
        same = np.mean(np.all(np.argsort(-b, axis=0) == np.argsort(-clean_boot, axis=0), axis=0))
        ranks.append({"config": r.config, "kind": r.kind, "severity": r.severity,
                      **{f"mAP_{m}": v for m, v in zip(MODELS, full)},
                      "order": ">".join(np.array(MODELS)[np.argsort([-v for v in full])]),
                      "kendall_tau_vs_clean": float(kendalltau(clean_rank, rk).statistic),
                      "order_original3": ">".join(sorted(orig, key=lambda m: -X[(m, r.config)][0])),
                      "kendall_tau_original3_vs_clean": float(kendalltau(
                          orig_clean, [X[(m, r.config)][0] for m in orig]).statistic),
                      "boot_share_same_order_as_clean": float(same)})
    RK = pd.DataFrame(ranks)
    RK.to_csv(OUT / "rank_agreement.csv", index=False)

    # ---------------------------------------------------------------- 3. benchmark disagreement
    bench = []
    for m in MODELS:
        clean = X[(m, "clean")]
        for kind in ("single", "pair"):
            cs = configs[configs.kind == kind].config
            ret = np.mean([X[(m, c)] / clean for c in cs], axis=0) * 100      # mean retention, %
            p, lo, hi = ci(ret)
            bench.append({"model": m, "benchmark": kind, "mean_retention_pct": p, "lo": lo, "hi": hi})
    B = pd.DataFrame(bench)
    B["rank"] = B.groupby("benchmark").mean_retention_pct.rank(ascending=False).astype(int)
    B.to_csv(OUT / "benchmark_rankings.csv", index=False)

    # paired test: is the retention gap between models the same under both benchmarks?
    bgap = []
    for m1, m2 in combinations(MODELS, 2):
        for kind in ("single", "pair"):
            cs = configs[configs.kind == kind].config
            r1 = np.mean([X[(m1, c)] / X[(m1, "clean")] for c in cs], axis=0) * 100
            r2 = np.mean([X[(m2, c)] / X[(m2, "clean")] for c in cs], axis=0) * 100
            p, lo, hi = ci(r1 - r2)
            bgap.append({"m1": m1, "m2": m2, "benchmark": kind, "retention_gap": p, "lo": lo, "hi": hi})
    BG = pd.DataFrame(bgap)
    BG.to_csv(OUT / "benchmark_retention_gaps.csv", index=False)

    # do models fail on the same configurations?
    D = pd.DataFrame({m: [X[(m, "clean")][0] - X[(m, c)][0] for c in configs.config[1:]] for m in MODELS})
    fail_corr = D.corr(method="spearman")
    fail_corr.to_csv(OUT / "drop_correlation_between_models.csv")

    # ---------------------------------------------------------------- report
    print("clean ordering:", ">".join(np.array(MODELS)[np.argsort([-X[(m, 'clean')][0] for m in MODELS])]),
          " ", {m: round(X[(m, 'clean')][0], 2) for m in MODELS})
    print("\norderings across the 108 degraded configurations:")
    print(RK[RK.kind != "clean"].groupby(["kind"]).order.value_counts().to_string())
    print("\nsignificant reversals of a significant clean gap:")
    rv = G[G.reversal]
    print(rv.groupby(["m1", "m2", "kind"]).size().to_string() if len(rv) else "  none")
    print(f"\nmean Kendall tau vs clean: singles {RK[RK.kind == 'single'].kendall_tau_vs_clean.mean():.3f}, "
          f"pairs {RK[RK.kind == 'pair'].kendall_tau_vs_clean.mean():.3f}")
    print(f"mean Kendall tau vs clean, original 3 detectors: "
          f"singles {RK[RK.kind == 'single'].kendall_tau_original3_vs_clean.mean():.3f}, "
          f"pairs {RK[RK.kind == 'pair'].kendall_tau_original3_vs_clean.mean():.3f}")
    print("\nrobustness ranking, single vs compound benchmark (mean retention %):")
    print(B.pivot(index="model", columns="benchmark", values=["mean_retention_pct", "rank"]).round(2).to_string())
    print("\nretention gap between models, single vs compound benchmark:")
    print(BG.round(2).to_string(index=False))
    print("\nSpearman correlation of per-config mAP drops between models:\n", fail_corr.round(3).to_string())

    # ---------------------------------------------------------------- figure
    pairs_m = list(combinations(MODELS, 2))
    ncol = min(7, len(pairs_m)); nrow = int(np.ceil(len(pairs_m) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.2 * nrow), dpi=130, squeeze=False)
    kc = {"clean": "black", "single": "#4C78A8", "pair": "#E45756"}
    for ax, (m1, m2) in zip(axes.flat, pairs_m):
        for kind in ("single", "pair", "clean"):
            h = RK[RK.kind == kind]
            ax.scatter(h[f"mAP_{m2}"], h[f"mAP_{m1}"], s=40 if kind == "clean" else 10,
                       color=kc[kind], label=kind, zorder=3 if kind == "clean" else 2,
                       marker="*" if kind == "clean" else "o")
        lim = [min(RK[f"mAP_{m1}"].min(), RK[f"mAP_{m2}"].min()) - 2, max(RK[f"mAP_{m1}"].max(), RK[f"mAP_{m2}"].max()) + 2]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_xlabel(f"{m2} mAP"); ax.set_ylabel(f"{m1} mAP")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.flat[len(pairs_m):]:
        ax.axis("off")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Model vs model across all configurations (crossing the diagonal = ranking change)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "fig_model_vs_model.png", facecolor="white"); plt.close()
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

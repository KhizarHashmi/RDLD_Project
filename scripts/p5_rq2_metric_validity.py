"""
p5_rq2_metric_validity.py -- P5 (Khizar, Workstream B): RQ2 -- does mAP predict
downstream document-reading quality?

Three-tier metric ladder (our metric plan):
  tier 1  UW-mAP      per-class AP weighted by downstream consequence
  tier 2  structural  merge / split / miss rates at the operating threshold (Workstream A, P4)
  tier 3  end-to-end  text a region pipeline reads: WER, text recall, table recall (Workstream A, P5)

Analyses:
  1. UW-mAP vs mAP: does the gap move with degradation? Weight sensitivity:
     2,000 random weight vectors (each weight x U(0.5, 1.5)).
  2. Which layout metric best predicts downstream quality? Spearman rho between
     each layout metric and each downstream outcome over (detector x condition),
     with a bootstrap CI over conditions.
  3. Does the ranking agree? Per condition, is the best detector by mAP also the
     best by WER?
  4. Threshold effect: two-segment fit of text recall on mAP retention.
  5. Within-condition correlation: condition means removed, so only the detector
     difference on the same images remains.

Outputs: RDLD_data/results/p5/rq2/

Run:
  python scripts/p5_rq2_metric_validity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdld import config, detectors
from rdld.doclaynet import CATEGORIES

R = config.RESULTS
BOOT = R / "p3" / "boot"
OUT = config.out_dir("p5", "rq2")
CLASSES = [CATEGORIES[i] for i in range(1, 12)]
# downstream-consequence weights, fixed before any result was seen (our metric plan, Sept 15)
WEIGHTS = {"Table": 3.0, "Formula": 2.5, "Picture": 2.0, "Section-header": 1.5, "Title": 1.5,
           "List-item": 1.2, "Caption": 1.0, "Text": 1.0, "Footnote": 0.6, "Page-header": 0.3, "Page-footer": 0.3}
W = np.array([WEIGHTS[c] for c in CLASSES])


def uw(ap, w):
    """Weighted mean over classes present (NaN-aware). ap: (..., 11)."""
    m = ~np.isnan(ap)
    return np.where(m, ap, 0) @ w / (m @ w)


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    models = list(detectors.MODELS)
    AP = {(m, c): 100 * np.load(BOOT / m / f"{c}.npy")[0] for m in models for c in configs.config}

    # ---------------------------------------------------------------- 1. UW-mAP
    rows = []
    for (m, c), ap in AP.items():
        rows.append({"model": m, "config": c, "mAP": float(np.nanmean(ap)), "UW_mAP": float(uw(ap, W))})
    U = pd.DataFrame(rows).merge(configs, on="config")
    U["gap"] = U.mAP - U.UW_mAP
    clean = U[U.config == "clean"].set_index("model")
    U["D_mAP"] = U.model.map(clean.mAP) - U.mAP
    U["D_UW"] = U.model.map(clean.UW_mAP) - U.UW_mAP
    U.to_csv(OUT / "uw_map.csv", index=False)
    gap_by = U.groupby(["model", "kind", "severity"]).gap.mean().unstack(["kind", "severity"])

    rng = np.random.default_rng(7)
    sens = []
    keys = [(m, c) for m in models for c in configs.config]
    APm = np.stack([AP[k] for k in keys])
    base_map = np.nanmean(APm, axis=1)
    sev = configs.set_index("config").severity
    sev_of = np.array([sev[c] for _, c in keys])
    for _ in range(2000):
        w = W * rng.uniform(0.5, 1.5, size=len(W))
        u = uw(APm, w)
        g = base_map - u
        drop_ratio = []
        for m in models:
            k = np.array([mm == m for mm, _ in keys])
            u_clean = u[k & (sev_of == 0)][0]
            m_clean = base_map[k & (sev_of == 0)][0]
            # ratio of mean drops: some configs score above clean, so per-config ratios are unstable
            drop_ratio.append(np.mean(u_clean - u[k & (sev_of == 3)]) / np.mean(m_clean - base_map[k & (sev_of == 3)]))
        sens.append({"gap_clean": g[sev_of == 0].mean(), "gap_sev3": g[sev_of == 3].mean(),
                     "uw_over_map_drop_sev3": float(np.mean(drop_ratio))})
    SE = pd.DataFrame(sens)
    SE.to_csv(OUT / "uw_weight_sensitivity.csv", index=False)

    # ---------------------------------------------------------------- 2. layout metric vs downstream
    M = pd.read_csv(R / "p2" / "evaluation" / "metrics_long.csv")
    S = pd.read_csv(R / "p4" / "structural" / "structural_rates.csv")
    DS = pd.read_csv(R / "p5" / "downstream" / "downstream_configs.csv")
    J = (DS[["model", "config", "kind", "severity", "WER", "text_recall", "text_precision", "table_recall"]]
         .merge(M[["model", "config", "mAP", "mAP50", "mAP75"]], on=["model", "config"])
         .merge(U[["model", "config", "UW_mAP"]], on=["model", "config"])
         .merge(S[["model", "config", "gt_correct", "structural_error", "gt_missed", "pred_duplicate"]], on=["model", "config"]))
    J[["mAP", "mAP50", "mAP75"]] *= 100
    J.to_csv(OUT / "layout_vs_downstream.csv", index=False)
    layout = ["mAP", "mAP50", "mAP75", "UW_mAP", "gt_correct", "structural_error", "pred_duplicate"]
    targets = ["WER", "text_recall", "text_precision", "table_recall"]
    corr = []
    for scope, g in [("pooled", J)] + [(m, J[J.model == m]) for m in models]:
        for a in layout:
            for t in targets:
                h = g[[a, t]].dropna()
                rho = spearmanr(h[a], h[t]).statistic
                bs = []
                for _ in range(500):
                    s = h.sample(len(h), replace=True, random_state=rng.integers(1 << 31))
                    bs.append(spearmanr(s[a], s[t]).statistic)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                corr.append({"scope": scope, "layout_metric": a, "downstream": t, "n": len(h),
                             "spearman": float(rho), "lo": float(lo), "hi": float(hi)})
    CR = pd.DataFrame(corr)
    CR.to_csv(OUT / "metric_vs_downstream_correlation.csv", index=False)

    # pooled rho is driven by how hard each condition is; removing the condition mean leaves
    # only "which detector is better on the same images"
    within = []
    D = J.copy()
    for col in layout + targets:
        D[col] = D[col] - D.groupby("config")[col].transform("mean")
    for a in layout:
        for t in targets:
            h = D[[a, t]].dropna()
            within.append({"layout_metric": a, "downstream": t, "n": len(h),
                           "spearman_within_condition": float(spearmanr(h[a], h[t]).statistic)})
    WC = pd.DataFrame(within)
    WC.to_csv(OUT / "within_condition_correlation.csv", index=False)

    # ---------------------------------------------------------------- 3. ranking agreement
    agree = []
    for c, g in J.groupby("config"):
        agree.append({"config": c, "best_by_mAP": g.loc[g.mAP.idxmax(), "model"],
                      "best_by_mAP50": g.loc[g.mAP50.idxmax(), "model"],
                      "best_by_WER": g.loc[g.WER.idxmin(), "model"],
                      "best_by_text_recall": g.loc[g.text_recall.idxmax(), "model"],
                      "order_mAP": ">".join(g.sort_values("mAP", ascending=False).model),
                      "order_WER": ">".join(g.sort_values("WER").model)})
    AG = pd.DataFrame(agree)
    AG["agree_best"] = AG.best_by_mAP == AG.best_by_WER
    AG["agree_order"] = AG.order_mAP == AG.order_WER
    AG.to_csv(OUT / "ranking_agreement.csv", index=False)

    # ---------------------------------------------------------------- 4. threshold (two-segment fit)
    J["retention"] = J.mAP / J.model.map(J[J.config == "clean"].set_index("model").mAP)
    x, y = J.retention.to_numpy(), J.text_recall.to_numpy()
    best = None
    for bp in np.quantile(x, np.linspace(0.1, 0.9, 81)):
        X = np.c_[np.ones_like(x), x, np.clip(x - bp, 0, None)]
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        sse = float(np.sum((X @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, bp, coef)
    X1 = np.c_[np.ones_like(x), x]
    sse1 = float(np.sum((X1 @ np.linalg.lstsq(X1, y, rcond=None)[0] - y) ** 2))
    n = len(x)
    f_stat = ((sse1 - best[0]) / 2) / (best[0] / (n - 4))
    TH = pd.DataFrame([{"breakpoint_retention": best[1], "slope_below": best[2][1], "slope_above": best[2][1] + best[2][2],
                        "sse_linear": sse1, "sse_segmented": best[0], "F_vs_linear": f_stat, "n": n}])
    TH.to_csv(OUT / "threshold_fit.csv", index=False)

    # ---------------------------------------------------------------- report
    print("UW-mAP minus mAP (gap, points), by condition:")
    print(gap_by.round(2).to_string())
    print(f"\nweight sensitivity (2000 draws): UW drop / mAP drop at severity 3 -> median "
          f"{SE.uw_over_map_drop_sev3.median():.2f}, 95% range [{SE.uw_over_map_drop_sev3.quantile(.025):.2f}, "
          f"{SE.uw_over_map_drop_sev3.quantile(.975):.2f}]")
    print("\nSpearman rho, layout metric vs downstream (pooled, n = %d):" % len(J))
    print(CR[CR.scope == "pooled"].pivot(index="layout_metric", columns="downstream", values="spearman").round(3).to_string())
    print("\nSpearman rho within condition (condition mean removed):")
    print(WC.pivot(index="layout_metric", columns="downstream", values="spearman_within_condition").round(3).to_string())
    print("\nper detector, rho(mAP, WER) and rho(structural_error, WER):")
    print(CR[(CR.scope != "pooled") & (CR.downstream == "WER") & CR.layout_metric.isin(["mAP", "structural_error"])]
          .pivot(index="scope", columns="layout_metric", values="spearman").round(3).to_string())
    print(f"\nbest detector by mAP == best by WER in {AG.agree_best.mean():.0%} of {len(AG)} conditions; "
          f"full order agrees in {AG.agree_order.mean():.0%}")
    print("\nthreshold fit (text recall vs mAP retention):")
    print(TH.round(4).to_string(index=False))

    # ---------------------------------------------------------------- figures
    cols = detectors.COLORS
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=160)
    for ax, t, lab in zip(axes, ["WER", "text_recall", "table_recall"], ["WER (lower better)", "text recall", "table-text recall"]):
        for m in models:
            g = J[J.model == m]
            ax.scatter(g.mAP, g[t], s=12, color=cols[m], label=m)
        rho = CR[(CR.scope == "pooled") & (CR.layout_metric == "mAP") & (CR.downstream == t)].spearman.iloc[0]
        ax.set_xlabel("mAP@[.5:.95]"); ax.set_ylabel(lab); ax.set_title(f"pooled rho = {rho:.2f}")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Layout mAP vs downstream reading quality (clean + severity-2 conditions)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "fig_map_vs_downstream.png", facecolor="white"); plt.close()

    piv = CR[CR.scope == "pooled"].pivot(index="layout_metric", columns="downstream", values="spearman").loc[layout]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-1, vmax=1)
    for (i, j), v in np.ndenumerate(piv.values):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(targets)), targets, fontsize=8); ax.set_yticks(range(len(layout)), layout, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    ax.set_title("Which layout metric tracks downstream quality?")
    plt.tight_layout(); plt.savefig(OUT / "fig_metric_correlation.png", facecolor="white"); plt.close()
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

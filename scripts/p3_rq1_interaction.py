"""
p3_rq1_interaction.py -- P3 (Khizar, Workstream B): RQ1 -- do degradations compound?

Inputs: paired bootstrap tables from p3_bootstrap.py, configs.csv.

For every detector, severity s and pair (A, B), with D = mAP drop from clean in points:

  additive null        I_add  = D(A+B) - [D(A) + D(B)]
  multiplicative null  I_mult = 100 * [R(A) * R(B) - R(A+B)],   R = mAP / mAP_clean

I > 0 means the pair hurts more than the null predicts (super-additive / compounding).
The multiplicative null is reported alongside because mAP is bounded: two large
drops cannot add without passing zero, so the additive null alone would label
saturation as "sub-additive".

Inference: 95% percentile CI and two-sided p from the paired page bootstrap;
Benjamini-Hochberg FDR across the 84 pairs within each detector and null.
Negative control: defocus@1 is a pixel no-op, so its 7 pairs must give I = 0 exactly.

Also: single-degradation curves (RoDLA-style replication) and how well the singles
predict the pair (additive / multiplicative / max) -- the "cheap estimator" question.

Outputs: RDLD_data/results/p3/rq1/

Run:
  python scripts/p3_rq1_interaction.py
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

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p3", "rq1")
ALPHA = 0.05
DEG_ORDER = ["texture", "ink_bleeding", "ink_holdout", "speckle", "watermark",
             "illumination", "defocus", "vibration"]


def mAP(model, cfg):
    a = np.load(BOOT / model / f"{cfg}.npy")
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=1)            # (N_BOOT+1,), row 0 = full corpus


def boot_stats(v):
    point, b = float(v[0]), v[1:]
    lo, hi = np.percentile(b, [2.5, 97.5])
    n = len(b)
    p = min(1.0, 2 * min((np.sum(b <= 0) + 1) / (n + 1), (np.sum(b >= 0) + 1) / (n + 1)))
    return point, float(lo), float(hi), p


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
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    pairs = configs[configs.kind == "pair"]
    rows, singles = [], []
    for m in detectors.MODELS:
        clean = mAP(m, "clean")
        cache = {}

        def get(c):
            if c not in cache:
                cache[c] = mAP(m, c)
            return cache[c]

        for r in configs[configs.kind == "single"].itertuples():
            d = 100 * (clean - get(r.config))
            pt, lo, hi, p = boot_stats(d)
            singles.append({"model": m, "degradation": r.a, "severity": r.severity,
                            "mAP": float(get(r.config)[0]), "D": pt, "D_lo": lo, "D_hi": hi})
        for r in pairs.itertuples():
            ca, cb = f"{r.a}@{r.severity}", f"{r.b}@{r.severity}"
            xa, xb, xab = get(ca), get(cb), get(r.config)
            da, db, dab = 100 * (clean - xa), 100 * (clean - xb), 100 * (clean - xab)
            i_add = dab - da - db
            i_mult = 100 * ((xa / clean) * (xb / clean) - xab / clean)
            row = {"model": m, "severity": r.severity, "a": r.a, "b": r.b, "config": r.config,
                   "mAP_clean": float(clean[0]), "D_a": float(da[0]), "D_b": float(db[0]),
                   "D_ab": float(dab[0]),
                   "pred_additive": float(da[0] + db[0]),
                   "pred_multiplicative": float(100 * clean[0] * (1 - (xa[0] / clean[0]) * (xb[0] / clean[0]))),
                   "pred_max": float(max(da[0], db[0]))}
            for name, v in (("add", i_add), ("mult", i_mult)):
                pt, lo, hi, p = boot_stats(v)
                row.update({f"I_{name}": pt, f"I_{name}_lo": lo, f"I_{name}_hi": hi, f"p_{name}": p})
            rows.append(row)

    S = pd.DataFrame(singles)
    S.to_csv(OUT / "single_degradation_drops.csv", index=False)
    R = pd.DataFrame(rows)
    for m in detectors.MODELS:
        for name in ("add", "mult"):
            k = R.model == m
            R.loc[k, f"q_{name}"] = bh(R.loc[k, f"p_{name}"])
    for name in ("add", "mult"):
        sig = R[f"q_{name}"] < ALPHA
        R[f"class_{name}"] = np.where(sig & (R[f"I_{name}"] > 0), "super",
                                      np.where(sig & (R[f"I_{name}"] < 0), "sub", "n.s."))
    R.to_csv(OUT / "rq1_interactions.csv", index=False)

    # ---------------------------------------------------------------- negative control
    ctrl = R[(R.severity == 1) & ((R.a == "defocus") | (R.b == "defocus"))]
    ctrl_ok = bool(np.allclose(ctrl.I_add, 0, atol=1e-9))
    print(f"negative control (defocus@1 pairs, n={len(ctrl)}): max |I_add| = {ctrl.I_add.abs().max():.2e}"
          f"  -> {'PASS' if ctrl_ok else 'FAIL'}")
    R_test = R.drop(ctrl.index)        # controls are excluded from the substantive counts

    # ---------------------------------------------------------------- summaries
    summ = []
    for (m, s), g in R_test.groupby(["model", "severity"]):
        row = {"model": m, "severity": s, "n_pairs": len(g)}
        for name in ("add", "mult"):
            vc = g[f"class_{name}"].value_counts()
            row.update({f"{name}_super": int(vc.get("super", 0)), f"{name}_sub": int(vc.get("sub", 0)),
                        f"{name}_ns": int(vc.get("n.s.", 0)), f"{name}_median_I": float(g[f"I_{name}"].median())})
        summ.append(row)
    for m, g in R_test.groupby("model"):
        row = {"model": m, "severity": "all", "n_pairs": len(g)}
        for name in ("add", "mult"):
            vc = g[f"class_{name}"].value_counts()
            row.update({f"{name}_super": int(vc.get("super", 0)), f"{name}_sub": int(vc.get("sub", 0)),
                        f"{name}_ns": int(vc.get("n.s.", 0)), f"{name}_median_I": float(g[f"I_{name}"].median())})
        summ.append(row)
    SM = pd.DataFrame(summ)
    SM.to_csv(OUT / "rq1_summary.csv", index=False)

    pred = []
    for m, g in R_test.groupby("model"):
        for name in ("additive", "multiplicative", "max"):
            err = g[f"pred_{name}"] - g.D_ab
            pred.append({"model": m, "predictor": name, "MAE_pts": float(err.abs().mean()),
                         "bias_pts": float(err.mean()),
                         "spearman": float(spearmanr(g[f"pred_{name}"], g.D_ab).statistic)})
    PR = pd.DataFrame(pred)
    PR.to_csv(OUT / "rq1_prediction_from_singles.csv", index=False)

    # pairs that compound under BOTH nulls: the robust super-additive set
    both = R_test[(R_test.class_add == "super") & (R_test.class_mult == "super")]
    both.sort_values("I_mult", ascending=False).to_csv(OUT / "rq1_super_under_both_nulls.csv", index=False)

    # pair structure: mean I_mult per degradation, averaged over partners and models
    long = pd.concat([R_test.assign(deg=R_test.a), R_test.assign(deg=R_test.b)])
    part = long.groupby(["deg", "severity"]).I_mult.mean().unstack().loc[DEG_ORDER]
    part.to_csv(OUT / "rq1_mean_interaction_by_degradation.csv")

    print("\nRQ1 summary (counts of pairs by class; controls excluded):")
    print(SM.to_string(index=False))
    print("\nPredicting the pair drop from the singles:")
    print(PR.round(3).to_string(index=False))
    print(f"\nsuper-additive under BOTH nulls: {len(both)} detector-pair cells")
    print(both[["model", "config", "D_a", "D_b", "D_ab", "I_add", "I_mult", "q_mult"]].round(2).head(15).to_string(index=False))

    # ---------------------------------------------------------------- figures
    figures(S, R, PR)
    print(f"\nwritten -> {OUT}")


def figures(S, R, PR):
    cols = {"yolo": "#4C78A8", "heron": "#F58518", "egret": "#54A24B"}
    # 1. single-degradation curves
    fig, axes = plt.subplots(2, 4, figsize=(13, 5.6), dpi=160, sharey=True)
    for ax, d in zip(axes.flat, DEG_ORDER):
        for m in detectors.MODELS:
            g = S[(S.model == m) & (S.degradation == d)].sort_values("severity")
            x = [0] + g.severity.tolist(); y = [0] + g.D.tolist()
            ax.plot(x, y, "-o", color=cols[m], ms=3, label=m)
            ax.fill_between(g.severity, g.D_lo, g.D_hi, color=cols[m], alpha=0.15)
        ax.set_title(d, fontsize=9); ax.set_xticks([0, 1, 2, 3])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("mAP drop (points)"); axes[1, 0].set_ylabel("mAP drop (points)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Single degradations: mAP drop vs severity (95% bootstrap CI)", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "fig_single_curves.png", facecolor="white"); plt.close()

    # 2. interaction heat maps, multiplicative null
    n = len(DEG_ORDER)
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), dpi=150)
    vmax = np.nanpercentile(np.abs(R.I_mult), 98)
    for i, m in enumerate(detectors.MODELS):
        for j, s in enumerate((1, 2, 3)):
            ax = axes[i, j]
            M = np.full((n, n), np.nan); sig = np.zeros((n, n), bool)
            for r in R[(R.model == m) & (R.severity == s)].itertuples():
                a, b = DEG_ORDER.index(r.a), DEG_ORDER.index(r.b)
                M[b, a] = r.I_mult; sig[b, a] = r.q_mult < ALPHA
            im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            for (y, x), v in np.ndenumerate(M):
                if not np.isnan(v):
                    ax.text(x, y, f"{v:.1f}{'*' if sig[y, x] else ''}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(n), DEG_ORDER, rotation=60, fontsize=6)
            ax.set_yticks(range(n), DEG_ORDER, fontsize=6)
            ax.set_title(f"{m}  severity {s}", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.5, label="I_mult (points; >0 = compounding)")
    fig.suptitle("Interaction under the multiplicative null (* = BH q < 0.05)", fontsize=12)
    plt.savefig(OUT / "fig_interaction_heatmaps.png", facecolor="white", bbox_inches="tight"); plt.close()

    # 3. observed vs predicted pair drop
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), dpi=160, sharex=True, sharey=True)
    sev_c = {1: "#9ecae1", 2: "#4292c6", 3: "#08519c"}
    for ax, m in zip(axes, detectors.MODELS):
        g = R[R.model == m]
        for s in (1, 2, 3):
            h = g[g.severity == s]
            ax.scatter(h.pred_multiplicative, h.D_ab, s=12, color=sev_c[s], label=f"severity {s}")
        lim = [min(g.pred_multiplicative.min(), g.D_ab.min()) - 2, max(g.pred_multiplicative.max(), g.D_ab.max()) + 2]
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.set_title(m); ax.set_xlabel("predicted drop, multiplicative null (pts)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("observed pair drop (pts)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Points above the line compound beyond the independent-damage prediction", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "fig_observed_vs_predicted.png", facecolor="white"); plt.close()


if __name__ == "__main__":
    main()

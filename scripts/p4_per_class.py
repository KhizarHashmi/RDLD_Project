"""
p4_per_class.py -- P4 (Alishba, Workstream A): which region types break first?

Uses Workstream B's paired bootstrap tables (per-class AP, columns = DocLayNet
classes 1..11) and the P1 EDA box table:

  1. per-class AP drop for every single degradation, with paired 95% CI
  2. per-class interaction (multiplicative null) for severity-2 and -3 pairs:
     does compounding hit some region types harder than others?
  3. drop vs. region size: AP_small / AP_medium / AP_large from metrics_long.csv,
     and each class's share of small boxes (EDA)

Title (n = 6) is reported but excluded from class rankings.

Outputs: RDLD_data/results/p4/per_class/

Run:
  python scripts/p4_per_class.py
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

from rdld import config
from rdld.doclaynet import CATEGORIES

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p4", "per_class")
CLASSES = [CATEGORIES[i] for i in range(1, 12)]
RANKED = [c for c in CLASSES if c != "Title"]
DEG_ORDER = ["texture", "ink_bleeding", "ink_holdout", "speckle", "watermark",
             "illumination", "defocus", "vibration"]


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    models = sorted(p.name for p in BOOT.iterdir() if p.is_dir())
    load = lambda m, c: 100 * np.load(BOOT / m / f"{c}.npy")          # (B+1, 11)

    rows, inter = [], []
    for m in models:
        clean = load(m, "clean")
        for r in configs[configs.kind == "single"].itertuples():
            d = clean - load(m, r.config)
            lo, hi = np.nanpercentile(d[1:], [2.5, 97.5], axis=0)
            for k, c in enumerate(CLASSES):
                rows.append({"model": m, "degradation": r.a, "severity": r.severity, "cls": c,
                             "AP_clean": clean[0, k], "D": d[0, k], "lo": lo[k], "hi": hi[k]})
        for r in configs[(configs.kind == "pair") & (configs.severity >= 2)].itertuples():
            xa, xb, xab = (load(m, f"{r.a}@{r.severity}"), load(m, f"{r.b}@{r.severity}"), load(m, r.config))
            with np.errstate(all="ignore"):
                im = 100 * ((xa / clean) * (xb / clean) - xab / clean)
            for k, c in enumerate(CLASSES):
                inter.append({"model": m, "severity": r.severity, "config": r.config, "cls": c, "I_mult": im[0, k]})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "per_class_single_drops.csv", index=False)
    I = pd.DataFrame(inter)
    I.to_csv(OUT / "per_class_interaction.csv", index=False)

    # class vulnerability: mean drop over the 8 singles at severity 3, averaged over models
    v3 = S[S.severity == 3].groupby(["cls"]).D.mean().reindex(CLASSES)
    vi = I.groupby(["cls", "severity"]).I_mult.median().unstack().reindex(CLASSES)
    eda = pd.read_csv(config.RESULTS / "p1" / "eda" / "box_features.csv")
    small = eda.groupby("cls").size_bucket.apply(lambda s: (s == "small").mean()).reindex(CLASSES)
    T = pd.DataFrame({"mean_drop_sev3_pts": v3, "median_I_mult_sev2": vi[2], "median_I_mult_sev3": vi[3],
                      "share_small_boxes": small, "n_boxes": eda.cls.value_counts().reindex(CLASSES)})
    T.to_csv(OUT / "class_vulnerability.csv")
    r = spearmanr(T.loc[RANKED, "share_small_boxes"], T.loc[RANKED, "mean_drop_sev3_pts"])

    M = pd.read_csv(config.RESULTS / "p2" / "evaluation" / "metrics_long.csv")
    sz = []
    for m, g in M.groupby("model"):
        c0 = g[g.config == "clean"].iloc[0]
        for kind in ("single", "pair"):
            h = g[(g.kind == kind) & (g.severity == 3)]
            sz.append({"model": m, "kind": kind,
                       **{f"drop_{b}": 100 * (c0[f"AP_{b}"] - h[f"AP_{b}"].mean()) for b in ("small", "medium", "large")}})
    SZ = pd.DataFrame(sz)
    SZ.to_csv(OUT / "drop_by_size_bucket.csv", index=False)

    print("class vulnerability (severity 3, mean over degradations and models):")
    print(T.round(3).sort_values("mean_drop_sev3_pts", ascending=False).to_string())
    print(f"\nSpearman(share of small boxes, drop), Title excluded: rho={r.statistic:.3f} p={r.pvalue:.3f}")
    print("\nmean AP drop by COCO size bucket, severity 3 (points):")
    print(SZ.round(2).to_string(index=False))

    # figure: class x degradation, severity 3, mean over models
    H = S[S.severity == 3].groupby(["cls", "degradation"]).D.mean().unstack().reindex(index=CLASSES, columns=DEG_ORDER)
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160)
    vmax = np.nanmax(np.abs(H.values))
    imh = ax.imshow(H.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for (y, x), v in np.ndenumerate(H.values):
        ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(DEG_ORDER)), DEG_ORDER, rotation=40, fontsize=8)
    ax.set_yticks(range(len(CLASSES)), [f"{c}{' (n=6)' if c == 'Title' else ''}" for c in CLASSES], fontsize=8)
    fig.colorbar(imh, ax=ax, label="AP drop (points)")
    ax.set_title("Per-class AP drop at severity 3 (mean of 3 detectors)")
    plt.tight_layout(); plt.savefig(OUT / "fig_class_by_degradation.png", facecolor="white"); plt.close()
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

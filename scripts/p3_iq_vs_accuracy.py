"""
p3_iq_vs_accuracy.py -- P3 (Alishba, Workstream A): does image damage explain
detection damage?

Joins Workstream A's image-quality table with Workstream B's detection results:

  1. Across all 108 configurations: rank correlation between image-level effect
     (mPE_IQA, PSNR) and the detectors' mAP drop. High = "how damaged the pixels
     look" predicts "how badly the detector fails".
  2. Interaction at the image level. For each pair, the same additive test as RQ1
     but on image damage d = 1 - MS-SSIM:  I_img = d(A+B) - d(A) - d(B).
     Correlating I_img with the detectors' I_mult separates two explanations of
     compounding: already present in the pixels, or created by the model.

Inputs: RDLD_data/results/p2/image_quality/iq_per_config.csv          (Workstream A)
        RDLD_data/results/p2/evaluation/metrics_long.csv              (Workstream B)
        RDLD_data/results/p3/rq1/rq1_interactions.csv                 (Workstream B)
Outputs: RDLD_data/results/p3/iq_vs_accuracy/

Run:
  python scripts/p3_iq_vs_accuracy.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdld import config

OUT = config.out_dir("p3", "iq_vs_accuracy")
R = config.RESULTS


def main():
    iq = pd.read_csv(R / "p2" / "image_quality" / "iq_per_config.csv")
    M = pd.read_csv(R / "p2" / "evaluation" / "metrics_long.csv")
    I = pd.read_csv(R / "p3" / "rq1" / "rq1_interactions.csv")
    J = M[M.kind != "clean"].merge(iq[["config", "psnr", "ms_ssim", "cw_ssim", "mPE_iqa"]], on="config")

    # 1. image effect vs detection effect
    corr = []
    for m, g in J.groupby("model"):
        for kind, h in [("all", g), ("single", g[g.kind == "single"]), ("pair", g[g.kind == "pair"])]:
            for col in ("mPE_iqa", "psnr"):
                r = spearmanr(h[col], h.D_pts)
                corr.append({"model": m, "configs": kind, "image_metric": col, "n": len(h),
                             "spearman_rho": float(r.statistic), "p": float(r.pvalue)})
    C = pd.DataFrame(corr)
    C.to_csv(OUT / "iq_vs_drop_correlation.csv", index=False)

    # 2. image-level interaction vs detector interaction
    d = dict(zip(iq.config, 1 - iq.ms_ssim))
    I["d_img_a"] = [d[f"{a}@{s}"] for a, s in zip(I.a, I.severity)]
    I["d_img_b"] = [d[f"{b}@{s}"] for b, s in zip(I.b, I.severity)]
    I["d_img_ab"] = I.config.map(d)
    I["I_img"] = 100 * (I.d_img_ab - I.d_img_a - I.d_img_b)
    I.to_csv(OUT / "image_vs_detector_interaction.csv", index=False)
    ic = []
    for m, g in I.groupby("model"):
        r = spearmanr(g.I_img, g.I_mult)
        ic.append({"model": m, "n_pairs": len(g), "spearman_I_img_vs_I_mult": float(r.statistic), "p": float(r.pvalue),
                   "share_pairs_I_img_positive": float((g.I_img > 0).mean())})
    IC = pd.DataFrame(ic)
    IC.to_csv(OUT / "interaction_correlation.csv", index=False)

    print("image effect vs mAP drop (Spearman rho):")
    print(C.pivot_table(index=["configs", "image_metric"], columns="model", values="spearman_rho").round(3).to_string())
    print("\nimage-level interaction vs detector interaction:")
    print(IC.round(3).to_string(index=False))

    # figure
    cols = {"single": "#4C78A8", "pair": "#E45756"}
    models = sorted(J.model.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 4), dpi=160, sharey=True)
    for ax, m in zip(axes, models):
        g = J[J.model == m]
        for k in ("single", "pair"):
            h = g[g.kind == k]
            ax.scatter(h.mPE_iqa, h.D_pts, s=10, alpha=0.75, color=cols[k], label=k)
        rho = C[(C.model == m) & (C.configs == "all") & (C.image_metric == "mPE_iqa")].spearman_rho.iloc[0]
        ax.set_title(f"{m}  (rho = {rho:.2f})"); ax.set_xlabel("mPE_IQA (image damage)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mAP drop (points)"); axes[0].legend(frameon=False)
    fig.suptitle("Image-level damage vs detection damage, 108 configurations", fontsize=11)
    plt.tight_layout(); plt.savefig(OUT / "fig_iq_vs_drop.png", facecolor="white"); plt.close()
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

"""Figures for the paper, drawn from the result CSVs.

IEEE conference layout: one column = 3.5 in, two columns = 7.16 in. Every label is Times New Roman at 8 pt
(the template asks for 8 pt Times in figure labels). Fonts are embedded as TrueType (pdf.fonttype 42).
Detectors are always named in text, so no reading depends on colour alone. Output: figures/*.pdf next to this script.
The data folders (RDLD_data, RDLD_data_test) sit next to the repository, as in docs/PIPELINE.md.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from PIL import Image

B = Path(__file__).resolve().parents[2]
V = B / "RDLD_data/results"
T = B / "RDLD_data_test/results"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#0b0b0b", "ytick.color": "#0b0b0b",
    "savefig.dpi": 600,
})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)

MODELS = ["yolo", "yolo26l", "yolo11m", "yolo11l", "heron", "heron101", "egret", "egretxl", "detr", "ddetr", "rfdetr"]
NAME = {"yolo": "YOLO26-m", "yolo26l": "YOLO26-l", "yolo11m": "YOLOv11-m", "yolo11l": "YOLOv11-l",
        "heron": "Heron", "heron101": "Heron-101", "egret": "Egret-m", "egretxl": "Egret-xl",
        "detr": "DETR", "ddetr": "Deformable DETR", "rfdetr": "RF-DETR"}
DEG = ["texture", "ink_bleeding", "ink_holdout", "speckle", "watermark", "illumination", "defocus", "vibration"]
DEGN = {"texture": "Texture", "ink_bleeding": "Ink bleeding", "ink_holdout": "Ink holdout", "speckle": "Speckle",
        "watermark": "Watermark", "illumination": "Illumination", "defocus": "Defocus", "vibration": "Vibration"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 1: one page, single and compound
def fig_example():
    page = "0020.png"
    conds = [("clean", None, "(a) Clean"), ("speckle@2", "degraded/speckle@2", "(b) Speckle, severity 2"),
             ("vibration@2", "degraded/vibration@2", "(c) Vibration, severity 2"),
             ("pair", "degraded/speckle@2+vibration@2", "(d) Speckle + vibration")]
    D = B / "RDLD_data"
    fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75))
    for ax, (_, sub, lab) in zip(axs, conds):
        p = D / "doclaynet/images" / page if sub is None else D / sub / page
        im = Image.open(p).convert("RGB")
        im = im.crop((40, 40, 600, 470))           # upper-left part of the page: header, text, table
        ax.imshow(im, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(INK2); s.set_linewidth(0.5)
        ax.set_xlabel(lab, labelpad=3)
    fig.subplots_adjust(wspace=0.04)
    save(fig, "fig1_example")


# ------------------------------------------------------------------ Fig. 2: mean interaction per pair
def fig_interaction():
    r = pd.read_csv(V / "p3/rq1/rq1_interactions.csv")
    cmap = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a9a", "#b3261e"])
    vmax = 8.0
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.95))
    for ax, sev in zip(axs, [1, 2, 3]):
        x = r[r.severity == sev].groupby(["a", "b"]).I_mult.mean()
        M = np.full((8, 8), np.nan)
        for (a, b), v in x.items():
            i, j = DEG.index(a), DEG.index(b)
            M[max(i, j), min(i, j)] = v
        im = ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), aspect="equal")
        for i in range(8):
            for j in range(8):
                if not np.isnan(M[i, j]):
                    ctrl = sev == 1 and "defocus" in (DEG[i], DEG[j])
                    txt = "ctl" if ctrl else f"{M[i, j]:.1f}".replace("-0.0", "0.0").replace("-", "−")
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8 if not ctrl else 7,
                            color="#ffffff" if abs(M[i, j]) > 5.5 else INK)
        ax.set_xticks(range(8)); ax.set_yticks(range(8))
        ax.set_xticklabels([DEGN[d] for d in DEG], rotation=55, ha="right", rotation_mode="anchor")
        ax.set_yticklabels([DEGN[d] for d in DEG] if sev == 1 else [])
        ax.set_title(f"Severity {sev}", pad=3)
        ax.set_xlim(-0.5, 6.5); ax.set_ylim(7.5, 0.5)      # lower triangle only
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.015)
    cb.set_label("Mean interaction (retention points)")
    cb.outline.set_linewidth(0.4)
    save(fig, "fig2_interaction")


# ------------------------------------------------------------------ Fig. 3: retention, singles vs pairs, with mRD rank
def fig_retention():
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=False)
    for ax, (root, title) in zip(axs, [(V, "(a) Validation pages"), (T, "(b) Test-split pages")]):
        br = pd.read_csv(root / "p4/rq3/benchmark_rankings.csv")
        s = br[br.benchmark == "single"].set_index("model")
        p = br[br.benchmark == "pair"].set_index("model")
        mrd = pd.read_csv(root / "p2/evaluation/mrd.csv").set_index("model").mRD
        mrd_rank = mrd.rank(method="min").astype(int)
        order = p.mean_retention_pct.sort_values().index.tolist()          # most robust at the top
        y = np.arange(len(order))
        for yi, k in zip(y, order):
            ax.plot([p.loc[k, "mean_retention_pct"], s.loc[k, "mean_retention_pct"]], [yi, yi], color=GRID, lw=2.2, zorder=1)
            ax.errorbar(s.loc[k, "mean_retention_pct"], yi, xerr=[[s.loc[k, "mean_retention_pct"] - s.loc[k, "lo"]], [s.loc[k, "hi"] - s.loc[k, "mean_retention_pct"]]],
                        fmt="o", ms=4.2, mfc="white", mec=BLUE, mew=1.1, ecolor=BLUE, elinewidth=0.7, capsize=0, zorder=3)
            ax.errorbar(p.loc[k, "mean_retention_pct"], yi, xerr=[[p.loc[k, "mean_retention_pct"] - p.loc[k, "lo"]], [p.loc[k, "hi"] - p.loc[k, "mean_retention_pct"]]],
                        fmt="s", ms=4.2, mfc=ORANGE, mec=ORANGE, ecolor=ORANGE, elinewidth=0.7, capsize=0, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([NAME[k] for k in order])
        ax.set_xlim(45, 90)
        ax.set_xlabel("Mean mAP retention (% of clean mAP)")
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(title, pad=3)
        ax2 = ax.twinx()                       # same y positions; right-hand labels only, no second scale
        ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(y)
        ax2.set_yticklabels([str(mrd_rank[k]) for k in order])
        ax2.tick_params(length=0)
        for sp in ax2.spines.values():
            sp.set_visible(False)
        ax2.set_ylabel("Rank by mRD (1 = most robust)", rotation=270, labelpad=9)
    h1 = plt.Line2D([], [], marker="o", ls="", mfc="white", mec=BLUE, mew=1.1, ms=4.2, label="Single degradations (23)")
    h2 = plt.Line2D([], [], marker="s", ls="", mfc=ORANGE, mec=ORANGE, ms=4.2, label="Degradation pairs (77)")
    fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(wspace=0.62, bottom=0.2)
    save(fig, "fig3_retention")


# ------------------------------------------------------------------ Fig. 4: within-condition correlation with downstream quality
def fig_rq2():
    metrics = [("mAP", "mAP"), ("mAP50", "mAP@0.5"), ("mAP75", "mAP@0.75"), ("UW_mAP", "UW-mAP"),
               ("gt_correct", "Correct regions"), ("pred_duplicate", "Duplicate share")]
    outs = [("WER", "(a) Word error rate"), ("text_precision", "(b) Text precision")]
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.05), sharey=True)
    for ax, (o, title) in zip(axs, outs):
        y = np.arange(len(metrics))
        for off, (root, col, lab, hatch) in zip([-0.19, 0.19], [(V, BLUE, "Validation", None), (T, ORANGE, "Test split", None)]):
            w = pd.read_csv(root / "p5/rq2/within_condition_correlation.csv")
            vals = [w[(w.layout_metric == m) & (w.downstream == o)].spearman_within_condition.iloc[0] for m, _ in metrics]
            ax.barh(y + off, vals, height=0.36, color=col, label=lab, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_xlim(-1, 1); ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"])
        ax.set_yticks(y); ax.set_yticklabels([n for _, n in metrics])
        if o == "WER":
            ax.invert_yaxis()          # shared y: invert once
        ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        ax.set_title(title, pad=3)
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("Within-condition Spearman correlation", fontsize=8, y=0.06)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.1))
    fig.subplots_adjust(wspace=0.12, bottom=0.27)
    save(fig, "fig4_rq2")


if __name__ == "__main__":
    fig_example(); fig_interaction(); fig_retention(); fig_rq2()
    print("figures written to", FIG)

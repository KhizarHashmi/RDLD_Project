"""
p0_make_figures.py -- (Khizar, Workstream B) Generate presentation figures from REAL project data.

Produces:
  fig_clean_vs_degraded.png   clean DocLayNet page | single degradation | compound
  fig_gt_boxes.png            what layout detection outputs (annotated page)
  fig_pipeline_effect.png     why layout errors cascade
  fig_interaction_concept.png additive vs super-additive, as a chart
  fig_degradation_grid.png    4 single degradations on one page
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os, json

from rdld import config, third_party

third_party.enable()
_cwd = os.getcwd()
os.chdir(config.THIRD_PARTY / "rodla_perturbation")

import cv2
import numpy as np
from blur import apply_defocus
from noise import apply_speckle, apply_texture
from inconsistency import apply_ink_bleeding, apply_illumination

os.chdir(_cwd)

OUT = config.out_dir("p0", "figures")
IMGS = config.IMAGES
COCO = json.loads(config.ANNOTATIONS.read_text(encoding="utf-8"))

WHITE = (255, 255, 255)


def label_bar(img, text, h=48, color=(40, 40, 40), fs=0.85):
    """Add a white caption bar under an image."""
    bar = np.full((h, img.shape[1], 3), 255, np.uint8)
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)[0]
    x = max(6, (img.shape[1] - sz[0]) // 2)
    cv2.putText(bar, text, (x, h - 16), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 2)
    return np.vstack([img, bar])


def pad(img, px=10):
    return cv2.copyMakeBorder(img, px, px, px, px, cv2.BORDER_CONSTANT, value=WHITE)


# choose a page with tables (visually compelling + downstream-relevant)
by_img = {}
for a in COCO["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)
cats = {c["id"]: c["name"] for c in COCO["categories"]}
pick = None
for rec in COCO["images"]:
    anns = by_img.get(rec["id"], [])
    names = [cats[a["category_id"]] for a in anns]
    if names.count("Table") >= 2 and 10 <= len(anns) <= 20:
        pick = rec
        break
if pick is None:
    pick = COCO["images"][0]

clean = cv2.imread(str(IMGS / pick["file_name"]))
print(f"using page {pick['file_name']}  {clean.shape}")

# ---------------------------------------------------------------- FIG 1
# clean | single | compound  -- the core motivation figure
single = apply_defocus(clean.copy(), degree=2)
compound = apply_ink_bleeding(apply_defocus(clean.copy(), degree=2).copy(), degree=2)
compound = apply_illumination(compound.copy(), degree=2)

row = np.hstack([
    pad(label_bar(clean, "CLEAN  (what benchmarks use)")),
    pad(label_bar(single, "1 DEGRADATION  (what RoDLA tests)")),
    pad(label_bar(compound, "3 AT ONCE  (what real scans are)")),
])
cv2.imwrite(str(OUT / "fig_clean_vs_degraded.png"), row)
print("wrote fig_clean_vs_degraded.png", row.shape)

# ---------------------------------------------------------------- FIG 2
# ground-truth boxes: "this is what layout detection produces"
COL = {"Table": (0, 0, 220), "Text": (0, 140, 0), "Section-header": (220, 60, 0),
       "Picture": (0, 165, 255), "Page-header": (128, 0, 128),
       "Page-footer": (128, 128, 0), "List-item": (0, 190, 190),
       "Caption": (200, 0, 200), "Formula": (80, 80, 255),
       "Footnote": (110, 110, 110), "Title": (255, 0, 255)}
gt = clean.copy()
for a in by_img[pick["id"]]:
    x, y, w, h = a["bbox"]
    n = cats[a["category_id"]]
    c = COL.get(n, (0, 0, 0))
    cv2.rectangle(gt, (int(x), int(y)), (int(x + w), int(y + h)), c, 3)
    cv2.rectangle(gt, (int(x), max(0, int(y) - 22)), (int(x) + 9 * len(n) + 10, int(y)), c, -1)
    cv2.putText(gt, n, (int(x) + 5, int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
cv2.imwrite(str(OUT / "fig_gt_boxes.png"), pad(gt))
print("wrote fig_gt_boxes.png")

# ---------------------------------------------------------------- FIG 3
# 4 single degradations in a 2x2 grid
d1 = label_bar(apply_defocus(clean.copy(), degree=3), "Defocus blur")
d2 = label_bar(apply_ink_bleeding(clean.copy(), degree=3), "Ink bleed-through")
d3 = label_bar(apply_illumination(clean.copy(), degree=3), "Uneven illumination")
d4 = label_bar(apply_texture(clean.copy(), degree=2), "Paper texture")
grid = np.vstack([
    np.hstack([pad(d1), pad(d2)]),
    np.hstack([pad(d3), pad(d4)]),
])
cv2.imwrite(str(OUT / "fig_degradation_grid.png"), grid)
print("wrote fig_degradation_grid.png", grid.shape)

# ---------------------------------------------------------------- FIG 4
# interaction concept as a bar chart (matplotlib)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
labels = ["Blur\nalone", "Ink bleed\nalone", "Expected\nif additive", "Actual?\n(unmeasured)"]
vals = [5, 4, 9, 15]
colors = ["#6B9BD1", "#6B9BD1", "#9AA5B1", "#D9534F"]
bars = ax.bar(labels, vals, color=colors, width=0.62, edgecolor="white", linewidth=1.5)
bars[3].set_hatch("//")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.35, f"{v}",
            ha="center", fontsize=13, fontweight="bold")
ax.set_ylabel("mAP points lost", fontsize=12)
ax.set_ylim(0, 18)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=11)
ax.annotate("", xy=(3, 13.6), xytext=(2, 9.6),
            arrowprops=dict(arrowstyle="->", lw=2.2, color="#D9534F"))
ax.text(2.52, 12.1, "the gap\nwe measure", color="#D9534F",
        fontsize=11, fontweight="bold", ha="center")
ax.set_title("If degradations compound, every published robustness number is optimistic",
             fontsize=12.5, pad=14)
plt.tight_layout()
plt.savefig(OUT / "fig_interaction_concept.png", facecolor="white")
plt.close()
print("wrote fig_interaction_concept.png")

# ---------------------------------------------------------------- FIG 5
# cascade diagram
fig, ax = plt.subplots(figsize=(10, 2.8), dpi=200)
ax.axis("off")
stages = ["Page\nimage", "LAYOUT\nDETECTION", "OCR\n(reading)", "Extraction\n/ QA"]
xs = [0.06, 0.30, 0.55, 0.80]
for i, (x, s) in enumerate(zip(xs, stages)):
    face = "#D9534F" if i == 1 else "#E8EDF2"
    txtc = "white" if i == 1 else "#2C3E50"
    ax.add_patch(plt.Rectangle((x, 0.42), 0.15, 0.34, facecolor=face,
                               edgecolor="#9AA5B1", lw=1.4, transform=ax.transAxes))
    ax.text(x + 0.075, 0.59, s, ha="center", va="center", fontsize=11,
            color=txtc, fontweight="bold" if i == 1 else "normal",
            transform=ax.transAxes)
    if i < 3:
        ax.annotate("", xy=(xs[i + 1] - 0.005, 0.59), xytext=(x + 0.155, 0.59),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=2, color="#5A6B7B"))
ax.text(0.375, 0.20, "an error here...", ha="center", fontsize=11.5,
        color="#D9534F", fontweight="bold", transform=ax.transAxes)
ax.annotate("", xy=(0.88, 0.30), xytext=(0.45, 0.20),
            xycoords=ax.transAxes, textcoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=2, color="#D9534F",
                            connectionstyle="arc3,rad=-0.25"))
ax.text(0.90, 0.20, "...breaks everything after it", ha="center", fontsize=11.5,
        color="#D9534F", fontweight="bold", transform=ax.transAxes)
plt.tight_layout()
plt.savefig(OUT / "fig_cascade.png", facecolor="white", bbox_inches="tight")
plt.close()
print("wrote fig_cascade.png")

print(f"\nAll figures -> {OUT}")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name:34s} {f.stat().st_size/1024:7.0f} KB")

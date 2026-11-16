"""
p1_eda.py -- P1 (Alishba, Workstream A): exploratory analysis of the clean corpus.

Characterises what the detectors will be tested on, and produces the per-page and
per-box feature tables that later phases join against (EDA and the
start of stage 5 feature engineering):

  box level   class, relative area, COCO size bucket, aspect ratio
  page level  regions, classes present, page coverage, ink ratio, mean intensity,
              colourfulness, document category

Outputs: RDLD_data/results/p1/eda/
  box_features.csv, page_features.csv, class_by_category.csv, size_buckets.csv,
  summary_by_category.csv, fig_*.png

Run:
  python scripts/p1_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdld import config
from rdld.doclaynet import CATEGORIES, DOC_CATEGORIES, load_coco

OUT = config.out_dir("p1", "eda")
gt = load_coco()
man = pd.read_csv(config.MANIFEST, dtype={"id": str})
man["image_id"] = man["id"].astype(int)
cat_of = dict(zip(man["image_id"], man["doc_category"]))
im_by = {im["id"]: im for im in gt["images"]}

# ---------------------------------------------------------------- box table
rows = []
for a in gt["annotations"]:
    im = im_by[a["image_id"]]
    x, y, w, h = a["bbox"]
    area = w * h
    rows.append({
        "ann_id": a["id"], "image_id": a["image_id"], "doc_category": cat_of[a["image_id"]],
        "cls": CATEGORIES[a["category_id"]], "x": x, "y": y, "w": w, "h": h, "area_px": area,
        "rel_area": area / (im["width"] * im["height"]),
        # COCO size buckets, as used by mAP_small / medium / large
        "size_bucket": "small" if area < 32 ** 2 else ("medium" if area < 96 ** 2 else "large"),
        "aspect": w / h if h > 0 else np.nan,
    })
B = pd.DataFrame(rows)
B.to_csv(OUT / "box_features.csv", index=False)

# ---------------------------------------------------------------- page table
prow = []
for im in gt["images"]:
    g = np.asarray(Image.open(config.IMAGES / im["file_name"]).convert("RGB"), dtype=np.float32)
    gray = g.mean(axis=2)
    # coverage: fraction of page pixels inside at least one GT box
    mask = np.zeros(gray.shape, bool)
    sub = B[B.image_id == im["id"]]
    for x, y, w, h in sub[["x", "y", "w", "h"]].round().astype(int).itertuples(index=False):
        mask[max(0, y):y + h, max(0, x):x + w] = True
    prow.append({
        "image_id": im["id"], "file_name": im["file_name"], "doc_category": cat_of[im["id"]],
        "n_regions": len(sub), "n_classes": sub.cls.nunique(),
        "n_tables": int((sub.cls == "Table").sum()), "n_pictures": int((sub.cls == "Picture").sum()),
        "coverage": mask.mean(),
        "ink_ratio": float((gray < 128).mean()),          # dark-pixel fraction
        "mean_intensity": float(gray.mean()),
        "colourfulness": float(g.std(axis=2).mean()),    # 0 for pure greyscale pages
        "median_rel_area": float(sub.rel_area.median()) if len(sub) else np.nan,
    })
P = pd.DataFrame(prow)
P.to_csv(OUT / "page_features.csv", index=False)

# ---------------------------------------------------------------- tables
cls_order = B.cls.value_counts().index.tolist()
CxC = pd.crosstab(B.cls, B.doc_category).loc[cls_order, DOC_CATEGORIES]
CxC.to_csv(OUT / "class_by_category.csv")
SZ = pd.crosstab(B.cls, B.size_bucket).reindex(columns=["small", "medium", "large"], fill_value=0).loc[cls_order]
SZ.to_csv(OUT / "size_buckets.csv")
S = P.groupby("doc_category").agg(
    pages=("image_id", "size"), regions_mean=("n_regions", "mean"), regions_sd=("n_regions", "std"),
    classes_mean=("n_classes", "mean"), tables=("n_tables", "sum"), pictures=("n_pictures", "sum"),
    coverage_mean=("coverage", "mean"), ink_ratio_mean=("ink_ratio", "mean"),
    colourful_pages=("colourfulness", lambda s: int((s > 2).sum())),
).loc[DOC_CATEGORIES]
S.to_csv(OUT / "summary_by_category.csv")

print("=" * 70)
print(" P1 EDA - clean corpus")
print("=" * 70)
print(f"\n{len(P)} pages, {len(B)} boxes\n")
print("class x document category (counts):\n", CxC.to_string(), "\n")
print("COCO size buckets per class:\n", SZ.to_string(), "\n")
print("per document category:\n", S.round(3).to_string(), "\n")
print(f"overall: small {int((B.size_bucket == 'small').sum())}  medium {int((B.size_bucket == 'medium').sum())}"
      f"  large {int((B.size_bucket == 'large').sum())}")

# ---------------------------------------------------------------- figures
COL = "#4C78A8"


def finish(ax, title, path):
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout(); plt.savefig(OUT / path, facecolor="white"); plt.close()


# 1. class share heat map by document category
share = CxC / CxC.sum(axis=0)
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
imh = ax.imshow(share.values, cmap="Blues", aspect="auto", vmin=0, vmax=share.values.max())
ax.set_xticks(range(len(DOC_CATEGORIES)), [c.replace("_", "\n") for c in DOC_CATEGORIES], fontsize=8)
ax.set_yticks(range(len(cls_order)), cls_order, fontsize=8)
for i in range(share.shape[0]):
    for j in range(share.shape[1]):
        n = CxC.values[i, j]
        ax.text(j, i, f"{n}", ha="center", va="center", fontsize=7,
                color="white" if share.values[i, j] > share.values.max() * 0.6 else "#333")
fig.colorbar(imh, ax=ax, label="share of category's boxes")
finish(ax, "Region classes by document category (cell = box count)", "fig_class_by_category.png")

# 2. regions per page by category
fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
ax.boxplot([P[P.doc_category == c].n_regions for c in DOC_CATEGORIES], showfliers=True)
ax.set_xticks(range(1, 7), [c.replace("_", "\n") for c in DOC_CATEGORIES], fontsize=8)
ax.set_ylabel("regions per page")
finish(ax, "Layout density: annotated regions per page", "fig_regions_per_page.png")

# 3. relative box area by class (log scale)
fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
ax.boxplot([B[B.cls == c].rel_area for c in cls_order], showfliers=False)
ax.set_yscale("log")
ax.set_xticks(range(1, len(cls_order) + 1), [f"{c}\n(n={int((B.cls == c).sum())})" for c in cls_order], fontsize=7)
ax.set_ylabel("box area / page area (log)")
finish(ax, "Region size by class", "fig_box_size_by_class.png")

# 4. page-level image statistics by category
fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
for ax, col, lab in [(axes[0], "ink_ratio", "dark-pixel fraction"), (axes[1], "coverage", "page area inside GT boxes")]:
    ax.boxplot([P[P.doc_category == c][col] for c in DOC_CATEGORIES], showfliers=False)
    ax.set_xticks(range(1, 7), [c.replace("_", "\n") for c in DOC_CATEGORIES], fontsize=7)
    ax.set_ylabel(lab); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("Ink density"); axes[1].set_title("Layout coverage")
plt.tight_layout(); plt.savefig(OUT / "fig_page_stats.png", facecolor="white"); plt.close()

print(f"\nwritten -> {OUT}")

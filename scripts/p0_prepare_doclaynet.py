"""
p0_prepare_doclaynet.py -- (Alishba, Workstream A) Extract a STRATIFIED DocLayNet subset to disk.

DocLayNet-v1.2 validation split = 6,489 pages across 6 document categories.
IMPORTANT: shards are sorted by category, NOT shuffled. Stratified sampling
therefore requires reading all 7 shards.

Outputs a COCO-format dataset ready for detection + perturbation:

  RDLD_data/doclaynet/
    images/0000.png ...          page images (1025x1025)
    annotations.json             COCO format, 11 classes
    manifest.csv                 per-page metadata for EDA

Run:
  python scripts/p0_prepare_doclaynet.py                # 300 pages, 50/category
  N_PER_CATEGORY=100 python scripts/p0_prepare_doclaynet.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import io
import json
import glob
import csv
import collections

import pyarrow.parquet as pq
from PIL import Image

from rdld.config import DOCLAYNET as OUT, IMAGES as IMGDIR
from rdld.doclaynet import CATEGORIES
IMGDIR.mkdir(parents=True, exist_ok=True)

N_PER_CAT = int(os.getenv("N_PER_CATEGORY", "50"))

CACHE = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub"
shards = sorted(glob.glob(str(CACHE / "datasets--ds4sd--DocLayNet-v1.2" /
                              "snapshots" / "*" / "data" / "validation-*.parquet")))
if not shards:
    raise SystemExit("No validation shards found. Download them first (see RUNBOOK).")

print("=" * 70)
print(" DOCLAYNET STRATIFIED EXTRACTION")
print("=" * 70)
print(f"\n{len(shards)} shards  |  target {N_PER_CAT} pages/category "
      f"({N_PER_CAT*6} total)\n")

taken = collections.Counter()
images, annotations, manifest = [], [], []
ann_id = 0
img_id = 0

for sp in shards:
    if all(taken[c] >= N_PER_CAT for c in
           ["financial_reports", "scientific_articles", "government_tenders",
            "laws_and_regulations", "manuals", "patents"]):
        break

    shard_name = sp.split("validation-")[1][:5]
    print(f"--- shard {shard_name}")
    tbl = pq.read_table(sp, columns=["image", "bboxes", "category_id",
                                     "area", "metadata"])
    rows = tbl.to_pylist()

    for r in rows:
        cat = r["metadata"]["doc_category"]
        if taken[cat] >= N_PER_CAT:
            continue
        if not r["category_id"]:          # skip pages with no annotations
            continue

        stem = f"{img_id:04d}"
        img = Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB")
        img.save(IMGDIR / f"{stem}.png")

        W, H = img.size
        images.append({
            "id": img_id, "file_name": f"{stem}.png",
            "width": W, "height": H,
        })

        for bbox, cid, area in zip(r["bboxes"], r["category_id"], r["area"]):
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(cid),
                "bbox": [float(v) for v in bbox],   # COCO: [x,y,w,h]
                "area": float(area), "iscrowd": 0,
            })
            ann_id += 1

        md = r["metadata"]
        manifest.append({
            "id": stem, "doc_category": cat,
            "collection": md.get("collection", ""),
            "original_filename": md.get("original_filename", ""),
            "page_no": md.get("page_no", ""),
            "width": W, "height": H,
            "n_regions": len(r["category_id"]),
            "classes_present": "|".join(sorted(
                {CATEGORIES.get(int(c), str(c)) for c in r["category_id"]})),
        })

        taken[cat] += 1
        img_id += 1

    print(f"    running totals: {dict(taken)}")

# ------------------------------------------------------------------ write
coco = {
    "info": {
        "description": "DocLayNet stratified subset for compound-degradation study",
        "source": "ds4sd/DocLayNet-v1.2 validation split",
        "license": "CDLA-Permissive-1.0",
    },
    "images": images,
    "annotations": annotations,
    "categories": [{"id": k, "name": v, "supercategory": "layout"}
                   for k, v in sorted(CATEGORIES.items())],
}
(OUT / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")

with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
    w.writeheader()
    w.writerows(manifest)

# ------------------------------------------------------------------ report
print("\n" + "=" * 70)
print(" RESULT")
print("=" * 70)
print(f"  pages extracted : {len(images)}")
print(f"  annotations     : {len(annotations)}")
print(f"  avg regions/page: {len(annotations)/max(1,len(images)):.1f}")

print("\n  per document category:")
for k, v in taken.most_common():
    print(f"    {k:24s} {v:4d}")

print("\n  per region class:")
cls = collections.Counter(CATEGORIES.get(a["category_id"], "?")
                          for a in annotations)
for k, v in cls.most_common():
    print(f"    {k:16s} {v:6d}")

mb = sum(f.stat().st_size for f in IMGDIR.glob("*.png")) / 1e6
print(f"\n  images  -> {IMGDIR}  ({mb:.0f} MB)")
print(f"  COCO    -> {OUT/'annotations.json'}")
print(f"  manifest-> {OUT/'manifest.csv'}")
print("\nNEXT: scripts/p1_validate_corpus.py, then p1_eda.py")

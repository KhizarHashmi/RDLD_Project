"""
p7_prepare_train_data.py -- P7 (Alishba, Workstream A): fine-tuning data for the
DINOv2-backbone detector (RF-DETR), with a leakage check against the test pages.

Every other detector in the study is a released checkpoint. RF-DETR has no released
layout checkpoint, so Workstream B fine-tunes it; this script builds its data.

  train      every page of the downloaded DocLayNet-v1.2 TRAIN shards (every 6th of
             72 shards; shards are sorted by document category, so taking every 6th
             spreads the subset over all six categories)
  select     model-selection pages from the VALIDATION split, stratified per
             category, EXCLUDING the 300 study pages (matched on original file and
             page number) -- the study pages are never seen during training or
             checkpoint selection

Leakage checks (all must pass):
  1. no study page (file, page) appears in train or select
  2. no study DOCUMENT (original file) appears in train   (DocLayNet splits by document)
  3. train and select share no page

Outputs: RDLD_data/finetune/{train,select}/images/*.png, annotations.json (COCO, 11 classes)
         RDLD_data/finetune/data_summary.csv

Run:
  python scripts/p7_prepare_train_data.py
  N_SELECT_PER_CATEGORY=100 python scripts/p7_prepare_train_data.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collections
import glob
import io
import json
import os

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from rdld import config
from rdld.doclaynet import CATEGORIES

N_SELECT = int(os.getenv("N_SELECT_PER_CATEGORY", "100"))
OUT = config.DATA_ROOT / "finetune"
SNAP = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "datasets--ds4sd--DocLayNet-v1.2" / "snapshots"
COLS = ["image", "bboxes", "category_id", "area", "metadata"]


def study_pages():
    m = pd.read_csv(config.MANIFEST, dtype=str)
    return set(zip(m.original_filename, m.page_no.astype(int))), set(m.original_filename)


def page_key(md):
    return md.get("original_filename", ""), int(md.get("page_no", -1))


class Split:
    """Streams one COCO split to disk: images written as they arrive, annotations kept."""

    def __init__(self, name):
        self.name, self.dir = name, OUT / name / "images"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.images, self.anns, self.meta = [], [], []

    def add(self, r):
        i, md = len(self.images), r["metadata"]
        fn = f"{i:05d}.png"
        if not (self.dir / fn).exists():
            Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(self.dir / fn)
        self.images.append({"id": i, "file_name": fn, "width": md["coco_width"], "height": md["coco_height"]})
        for b, c, a in zip(r["bboxes"], r["category_id"], r["area"]):
            self.anns.append({"id": len(self.anns), "image_id": i, "category_id": int(c),
                              "bbox": [float(v) for v in b], "area": float(a), "iscrowd": 0})
        self.meta.append({"split": self.name, "doc_category": md["doc_category"],
                          "original_filename": md["original_filename"], "page_no": md["page_no"]})

    def close(self):
        coco = {"images": self.images, "annotations": self.anns,
                "categories": [{"id": k, "name": v} for k, v in sorted(CATEGORIES.items())]}
        (OUT / self.name / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")


def main():
    pages, docs = study_pages()
    train_shards = sorted(glob.glob(str(SNAP / "*" / "data" / "train-*.parquet")))
    val_shards = sorted(glob.glob(str(SNAP / "*" / "data" / "validation-*.parquet")))
    print(f"train shards on disk: {len(train_shards)}, validation shards: {len(val_shards)}")
    if not train_shards:
        raise SystemExit("no train shards downloaded yet")

    # ---- pass 1, metadata only: decide membership and check leakage before writing anything
    tr_keys = set()
    for sp in train_shards:
        for r in pq.read_table(sp, columns=["category_id", "metadata"]).to_pylist():
            if r["category_id"]:
                tr_keys.add(page_key(r["metadata"]))
    taken, se_keys = collections.Counter(), set()
    for sp in val_shards:
        for r in pq.read_table(sp, columns=["category_id", "metadata"]).to_pylist():
            md, k = r["metadata"], page_key(r["metadata"])
            if r["category_id"] and k not in pages and taken[md["doc_category"]] < N_SELECT:
                se_keys.add(k)
                taken[md["doc_category"]] += 1
    checks = {"study pages in train": len(tr_keys & pages), "study pages in select": len(se_keys & pages),
              "study documents in train": len({k[0] for k in tr_keys} & docs),
              "pages shared train/select": len(tr_keys & se_keys)}
    for k, v in checks.items():
        print(f"  [{'PASS' if v == 0 else 'FAIL'}] {k}: {v}")
    if any(checks.values()):
        raise SystemExit("leakage check failed")

    # ---- pass 2: stream images shard by shard
    splits = {"train": Split("train"), "select": Split("select")}
    for name, shards, keys in [("train", train_shards, tr_keys), ("select", val_shards, se_keys)]:
        for sp in shards:
            pf = pq.ParquetFile(sp)
            for batch in pf.iter_batches(batch_size=64, columns=COLS):
                for r in batch.to_pylist():
                    if r["category_id"] and page_key(r["metadata"]) in keys:
                        splits[name].add(r)
            print(f"  {name} {Path(sp).name}: {len(splits[name].images)} pages")
        splits[name].close()

    S = pd.DataFrame(splits["train"].meta + splits["select"].meta)
    S.to_csv(OUT / "data_summary.csv", index=False)
    for n, sp in splits.items():
        print(f"\n{n:6s}: {len(sp.images)} pages, {len(sp.anns)} boxes")
    print(S.groupby(["split", "doc_category"]).size().unstack(0).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

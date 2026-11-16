"""
p7_test_split_leakage.py -- (Khizar, Workstream B) Overlap check for the official-test-split repeat.

The main study's 300 pages come from the DocLayNet validation split. The repeat draws 300
pages (50 per category, same rule as P0) from the official test split. Before any result
on those pages is used, this checks that none of their source documents appear in
  - the RF-DETR training data (the 12 train shards used by p7_prepare_train_data.py),
  - the validation split (the 300 main pages and the 600 RF-DETR selection pages come from it).
Documents are compared by original file name; pages by (file name, page number).

Run (RDLD_DATA points at the test-split data folder):
  RDLD_DATA=../RDLD_data_test python scripts/p7_test_split_leakage.py

Output: RDLD_DATA/results/p7/test_split_leakage.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import glob
import os

import pandas as pd
import pyarrow.parquet as pq

from rdld import config

CACHE = Path(os.path.expanduser("~")) / ".cache" / "huggingface" / "hub" / "datasets--ds4sd--DocLayNet-v1.2"


def docs(pattern):
    """(file name, page) pairs of every page in the matching cached shards."""
    rows = []
    for sp in sorted(glob.glob(str(CACHE / "snapshots" / "*" / "data" / pattern))):
        md = pq.read_table(sp, columns=["metadata"]).column("metadata").to_pylist()
        rows += [(m["original_filename"], m["page_no"]) for m in md]
    return pd.DataFrame(rows, columns=["doc", "page"])


def main():
    test = pd.read_csv(config.MANIFEST, dtype={"id": str})
    t_docs, t_pages = set(test.original_filename), set(zip(test.original_filename, test.page_no))
    out = []
    for name, pattern in [("RF-DETR train shards (12 of 72)", "train-*.parquet"),
                          ("validation split (all 7 shards)", "validation-*.parquet")]:
        d = docs(pattern)
        out.append({"against": name, "n_pages": len(d), "n_docs": d.doc.nunique(),
                    "test_docs_shared": len(t_docs & set(d.doc)),
                    "test_pages_shared": len(t_pages & set(zip(d.doc, d.page)))})
    O = pd.DataFrame(out)
    O.to_csv(config.out_dir("p7") / "test_split_leakage.csv", index=False)
    print(f"test pages: {len(test)} from {len(t_docs)} documents; per category:",
          test.doc_category.value_counts().to_dict())
    print(O.to_string(index=False))
    if O.test_docs_shared.sum():
        raise SystemExit("LEAKAGE: test documents found in training or validation data")
    print("no shared documents")


if __name__ == "__main__":
    main()

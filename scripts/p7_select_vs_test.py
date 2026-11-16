"""
p7_select_vs_test.py -- P7 (Khizar, Workstream B): do the detectors generalise alike?

RF-DETR's training data is known to exclude the 300 test pages and their documents. For the
released detectors we rely on what their publishers state. This compares every detector on two
page sets, the 600 selection pages (validation split) and the 300 test pages, both balanced
across the six document categories.

Page difficulty should move every detector the same way. A detector scoring unusually well on
one set compared with the others would need explaining, for example by overlap with its
training data. This is a screening check, not proof either way.

Also reports ground-truth counts per class, because a macro-averaged mAP on few instances is noisy.

Outputs: RDLD_data/results/p7/select_vs_test/{select_vs_test.csv, per_class.csv, gt_counts.csv}

Run:
  python scripts/p7_select_vs_test.py
  MODELS=heron,rfdetr python scripts/p7_select_vs_test.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image

from rdld import config, detectors, evaluation, fastcoco
from rdld.doclaynet import CATEGORIES, load_coco

N_BOOT = int(os.getenv("N_BOOT", "1000"))
MODELS = os.getenv("MODELS", ",".join(detectors.MODELS)).split(",")
SEL = config.DATA_ROOT / "finetune" / "select"
OUT = config.out_dir("p7", "select_vs_test")


def predict(det, gt):
    dets, bs = [], getattr(det, "batch", 8)
    for i in range(0, len(gt["images"]), bs):
        chunk = gt["images"][i:i + bs]
        pil = [Image.open(SEL / "images" / im["file_name"]).convert("RGB") for im in chunk]
        for p, im in zip(det.predict(pil), chunk):
            dets.extend(detectors.to_coco_results(p, im["id"]))
    return dets


def main():
    sel = json.loads((SEL / "annotations.json").read_text(encoding="utf-8"))
    test = load_coco()
    counts = [{"class": CATEGORIES[c],
               "test_boxes": sum(a["category_id"] == c for a in test["annotations"]),
               "select_boxes": sum(a["category_id"] == c for a in sel["annotations"])} for c in CATEGORIES]
    pd.DataFrame(counts).to_csv(OUT / "gt_counts.csv", index=False)
    print(pd.DataFrame(counts).to_string(index=False), flush=True)

    rows, per_class = [], []
    for key in MODELS:
        det = detectors.load(key)
        ds = predict(det, sel)
        del det
        torch.cuda.empty_cache()
        dt = json.loads((config.PREDICTIONS / key / "clean.json").read_text(encoding="utf-8"))
        rs, rt = evaluation.coco_map(sel, ds), evaluation.coco_map(test, dt)
        ps, pt = fastcoco.PerImageEval(sel, ds), fastcoco.PerImageEval(test, dt)
        rng = np.random.default_rng(0)
        ns, nt = len(sel["images"]), len(test["images"])
        gap = np.array([ps.map(rng.integers(0, ns, ns)) - pt.map(rng.integers(0, nt, nt)) for _ in range(N_BOOT)])
        lo, hi = 100 * np.percentile(gap, [2.5, 97.5])
        rows.append({"model": key, "select_mAP": 100 * rs["mAP"], "test_mAP": 100 * rt["mAP"],
                     "select_mAP50": 100 * rs["mAP50"], "test_mAP50": 100 * rt["mAP50"],
                     "gap_select_minus_test": 100 * (rs["mAP"] - rt["mAP"]), "gap_lo": lo, "gap_hi": hi})
        for c in rs["per_class"]:
            per_class.append({"model": key, "class": c, "select_AP": 100 * rs["per_class"][c], "test_AP": 100 * rt["per_class"][c]})
        print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False), flush=True)
        pd.DataFrame(rows).to_csv(OUT / "select_vs_test.csv", index=False)
        pd.DataFrame(per_class).to_csv(OUT / "per_class.csv", index=False)

    print("\n" + pd.DataFrame(rows).round(2).to_string(index=False))
    P = pd.DataFrame(per_class)
    P["gap"] = P.select_AP - P.test_AP
    print("\nper-class gap (select - test):\n", P.pivot(index="class", columns="model", values="gap").round(1).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

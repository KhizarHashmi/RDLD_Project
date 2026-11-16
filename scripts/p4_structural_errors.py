"""
p4_structural_errors.py -- P4 (Alishba, Workstream A): structural error rates.

mAP integrates over all confidence levels and lets a low-score wrong box cost
almost nothing. A downstream system instead takes the boxes above one operating
threshold and uses them as-is. This scores exactly that output against the
ground truth with an error taxonomy adapted from the LED benchmark (missing,
hallucination, split, merge, duplicate, misclassification, size/localisation).

At score >= TAU (default 0.5), per page, class-agnostic geometry first:

  per ground-truth region (mutually exclusive, in this priority):
    correct        one-to-one match IoU >= 0.5 and same class
    misclassified  one-to-one match IoU >= 0.5, different class
    merged         a single prediction covers it and >=1 other GT (each >= 50% covered)
    split          >= 2 predictions lying mostly (>= 70%) inside it, none matching it
    localisation   best IoU in [0.1, 0.5)
    missed         best IoU < 0.1
  per prediction:
    duplicate      IoU >= 0.5 with a GT that is already matched by another prediction
    hallucination  IoU < 0.1 with every GT and not mostly inside any GT

Outputs: RDLD_data/results/p4/structural/
  structural_rates.csv          rates per detector x configuration (+ Table-only rates)
  structural_vs_clean.csv       change from clean

Run:
  python scripts/p4_structural_errors.py
  TAU=0.3 python scripts/p4_structural_errors.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

from rdld import config
from rdld.doclaynet import NAME_TO_ID, load_coco

TAU = float(os.getenv("TAU", "0.5"))
WORKERS = int(os.getenv("WORKERS", "14"))
OUT = config.out_dir("p4", "structural")
GT_KINDS = ["correct", "misclassified", "merged", "split", "localisation", "missed"]
TABLE = NAME_TO_ID["Table"]
_gt = None


def xyxy(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.c_[b[:, 0], b[:, 1], b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]]


def overlaps(g, p):
    """IoU, share of each GT covered (IoA_g) and share of each pred inside (IoA_p): (G, P)."""
    x1 = np.maximum(g[:, None, 0], p[None, :, 0]); y1 = np.maximum(g[:, None, 1], p[None, :, 1])
    x2 = np.minimum(g[:, None, 2], p[None, :, 2]); y2 = np.minimum(g[:, None, 3], p[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ag = ((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]))[:, None]
    ap = ((p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1]))[None, :]
    return inter / np.maximum(ag + ap - inter, 1e-9), inter / np.maximum(ag, 1e-9), inter / np.maximum(ap, 1e-9)


def classify_page(gt_boxes, gt_cls, pr_boxes, pr_cls, pr_score):
    G, P = len(gt_boxes), len(pr_boxes)
    kinds = np.array(["missed"] * G, dtype=object)
    pr_kind = np.array(["ok"] * P, dtype=object)
    if P == 0:
        return kinds, pr_kind
    iou, ioa_g, ioa_p = overlaps(gt_boxes, pr_boxes)
    # one-to-one greedy matching by score, class-agnostic, IoU >= 0.5
    match_of_pred, gt_taken = -np.ones(P, int), np.zeros(G, bool)
    for j in np.argsort(-pr_score):
        cand = np.where((iou[:, j] >= 0.5) & ~gt_taken)[0]
        if len(cand):
            i = cand[np.argmax(iou[cand, j])]
            match_of_pred[j], gt_taken[i] = i, True
    for j, i in enumerate(match_of_pred):
        if i >= 0:
            kinds[i] = "correct" if pr_cls[j] == gt_cls[i] else "misclassified"
    for i in range(G):
        if kinds[i] != "missed":
            continue
        covers = np.where(ioa_g[i] >= 0.5)[0]
        merged = any((ioa_g[:, j] >= 0.5).sum() >= 2 for j in covers)
        inside = np.where((ioa_p[i] >= 0.7) & (iou[i] < 0.5))[0]
        best = iou[i].max()
        kinds[i] = ("merged" if merged else "split" if len(inside) >= 2
                    else "localisation" if best >= 0.1 else "missed")
    for j in range(P):
        if match_of_pred[j] >= 0:
            continue
        if (iou[:, j] >= 0.5).any():
            pr_kind[j] = "duplicate"
        elif iou[:, j].max() < 0.1 and ioa_p[:, j].max() < 0.5:
            pr_kind[j] = "hallucination"
    return kinds, pr_kind


def _job(args):
    global _gt
    if _gt is None:
        g = load_coco()
        by = defaultdict(list)
        for a in g["annotations"]:
            by[a["image_id"]].append(a)
        _gt = {im["id"]: (xyxy([a["bbox"] for a in by[im["id"]]]), np.array([a["category_id"] for a in by[im["id"]]]))
               for im in g["images"]}
    model, cfg = args
    dets = json.loads((config.PREDICTIONS / model / f"{cfg}.json").read_text(encoding="utf-8"))
    by = defaultdict(list)
    for d in dets:
        if d["score"] >= TAU:
            by[d["image_id"]].append(d)
    gk, tk, pk = Counter(), Counter(), Counter()
    n_pred = 0
    for iid, (gb, gc) in _gt.items():
        ds = by.get(iid, [])
        pb = xyxy([d["bbox"] for d in ds]) if ds else np.zeros((0, 4))
        kinds, pr_kind = classify_page(gb, gc, pb, np.array([d["category_id"] for d in ds], int),
                                       np.array([d["score"] for d in ds]))
        gk.update(kinds); tk.update(kinds[gc == TABLE]); pk.update(pr_kind); n_pred += len(ds)
    n_gt, n_tab = sum(gk.values()), sum(tk.values())
    row = {"model": model, "config": cfg, "tau": TAU, "n_gt": n_gt, "n_pred": n_pred}
    row.update({f"gt_{k}": gk[k] / n_gt for k in GT_KINDS})
    row.update({f"table_{k}": tk[k] / max(n_tab, 1) for k in GT_KINDS})
    row.update({"pred_duplicate": pk["duplicate"] / max(n_pred, 1), "pred_hallucination": pk["hallucination"] / max(n_pred, 1)})
    row["structural_error"] = row["gt_merged"] + row["gt_split"]
    return row


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    models = sorted(p.name for p in config.PREDICTIONS.iterdir() if p.is_dir())
    jobs = [(m, c) for m in models for c in configs.config if (config.PREDICTIONS / m / f"{c}.json").exists()]
    with Pool(WORKERS) as pool:
        rows = list(pool.imap_unordered(_job, jobs, chunksize=2))
    S = pd.DataFrame(rows).merge(configs, on="config").sort_values(["model", "kind", "severity", "config"])
    S.to_csv(OUT / "structural_rates.csv", index=False)
    rate_cols = [c for c in S.columns if c.startswith(("gt_", "table_", "pred_")) or c == "structural_error"]
    clean = S[S.config == "clean"].set_index("model")[rate_cols]
    V = S.copy()
    for c in rate_cols:
        V[c] = 100 * (V[c] - V.model.map(clean[c]))
    V.to_csv(OUT / "structural_vs_clean.csv", index=False)

    print(f"operating threshold tau = {TAU}")
    print("\nclean-condition error profile (% of GT regions / % of predictions):")
    print((100 * clean[[f"gt_{k}" for k in GT_KINDS] + ["pred_duplicate", "pred_hallucination"]]).round(2).to_string())
    print("\nmean change vs clean, severity 3 (percentage points):")
    show = ["gt_correct", "gt_misclassified", "gt_merged", "gt_split", "gt_missed", "table_merged", "table_split", "pred_hallucination"]
    print(V[V.severity == 3].groupby(["model", "kind"])[show].mean().round(2).to_string())
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

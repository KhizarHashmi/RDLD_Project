"""COCO-style detection metrics and prediction-noise diagnostics (Workstream B)."""
import contextlib
import copy
import io

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from rdld.doclaynet import CATEGORIES


def _coco_from_dict(d):
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def coco_map(gt_dict, dets, img_ids=None):
    """mAP@[.5:.95], mAP@.5, mAP@.75 and per-class AP (NaN if class absent).

    gt_dict: COCO ground-truth dict. dets: COCO results list. img_ids: subset.
    """
    gt = _coco_from_dict(gt_dict)
    ids = img_ids if img_ids is not None else sorted(gt.getImgIds())
    idset = set(ids)
    dets = [d for d in dets if d["image_id"] in idset]
    out = {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0,
           "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0,
           "per_class": {n: float("nan") for n in CATEGORIES.values()}}
    if not dets:
        return out
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(copy.deepcopy(dets))
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    out["mAP"], out["mAP50"], out["mAP75"] = (float(ev.stats[i]) for i in range(3))
    out["AP_small"], out["AP_medium"], out["AP_large"] = (float(ev.stats[i]) for i in range(3, 6))
    prec = ev.eval["precision"]                       # [T, R, K, A, M]
    for k, cid in enumerate(ev.params.catIds):
        p = prec[:, :, k, 0, -1]
        p = p[p > -1]
        out["per_class"][CATEGORIES[cid]] = float(p.mean()) if p.size else float("nan")
    return out


def iou_matrix(a, b):
    """Pairwise IoU between xyxy box arrays a (N,4) and b (M,4)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def xywh_to_xyxy(dets):
    return np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][0] + d["bbox"][2],
                      d["bbox"][1] + d["bbox"][3]] for d in dets]).reshape(-1, 4)


def cross_label_duplicates(dets, score_thr, iou_thr=0.9):
    """Fraction of predictions (score >= score_thr) that sit on the same region as a
    differently-labelled prediction (IoU >= iou_thr). The clean-input label-instability floor."""
    by_img = {}
    for d in dets:
        if d["score"] >= score_thr:
            by_img.setdefault(d["image_id"], []).append(d)
    n_total = n_dup = 0
    for ds in by_img.values():
        boxes = xywh_to_xyxy(ds)
        labels = np.array([d["category_id"] for d in ds])
        m = iou_matrix(boxes, boxes)
        np.fill_diagonal(m, 0)
        conflict = (m >= iou_thr) & (labels[:, None] != labels[None, :])
        n_total += len(ds)
        n_dup += int(conflict.any(axis=1).sum())
    return n_dup / max(n_total, 1), n_dup, n_total


def agnostic_dedup(dets, iou_thr=0.9):
    """Class-agnostic suppression: on near-identical boxes keep only the highest score."""
    by_img = {}
    for d in dets:
        by_img.setdefault(d["image_id"], []).append(d)
    out = []
    for ds in by_img.values():
        ds = sorted(ds, key=lambda d: -d["score"])
        boxes = xywh_to_xyxy(ds)
        m = iou_matrix(boxes, boxes)
        alive = np.ones(len(ds), bool)
        for i in range(len(ds)):
            if not alive[i]:
                continue
            sup = m[i] >= iou_thr
            sup[: i + 1] = False
            alive &= ~sup
        out.extend(d for d, a in zip(ds, alive) if a)
    return out

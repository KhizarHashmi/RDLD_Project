"""Fast resampled COCO mAP for paired page-level bootstrap (Workstream B).

pycocotools does two things: per-image matching (evaluate) and precision/recall
accumulation (accumulate). Matching depends only on one image's boxes, so it is
run once; accumulation is then redone in numpy for any multiset of pages. That
turns a bootstrap of B resamples from B full COCOeval runs into one COCOeval plus
B cheap re-accumulations, which is what makes paired bootstrap CIs on the
interaction term affordable.

Accumulation mirrors COCOeval.accumulate for area 'all', maxDets=100 (the
primary mAP@[.5:.95]); `verify()` checks the full-sample result equals pycocotools.
"""
import contextlib
import copy
import io

import numpy as np
from pycocotools.cocoeval import COCOeval

from rdld.evaluation import _coco_from_dict

REC_THRS = np.linspace(0.0, 1.0, 101)
IOU_THRS = np.linspace(0.5, 0.95, 10)


class PerImageEval:
    """Per (class, page) matching results for one prediction set."""

    def __init__(self, gt_dict, dets):
        gt = _coco_from_dict(gt_dict)
        self.img_ids = sorted(gt.getImgIds())
        self.cat_ids = sorted(gt.getCatIds())
        with contextlib.redirect_stdout(io.StringIO()):
            dt = gt.loadRes(copy.deepcopy(dets)) if dets else None
        n_img, n_cat = len(self.img_ids), len(self.cat_ids)
        # per class: list over pages of (scores, matched[T,n], ignore[T,n]); and gt counts
        self.scores = [[np.zeros(0)] * n_img for _ in range(n_cat)]
        self.tp = [[np.zeros((10, 0), bool)] * n_img for _ in range(n_cat)]
        self.valid = [[np.zeros((10, 0), bool)] * n_img for _ in range(n_cat)]
        self.npos = np.zeros((n_cat, n_img), int)
        if dt is None:
            for k, c in enumerate(self.cat_ids):
                for i, iid in enumerate(self.img_ids):
                    self.npos[k, i] = len(gt.getAnnIds(imgIds=iid, catIds=c, iscrowd=False))
            return
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = self.img_ids
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.params.maxDets = [100]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate()
        # evalImgs is ordered [cat][area][img]
        for k in range(n_cat):
            for i in range(n_img):
                e = ev.evalImgs[k * n_img + i]
                if e is None:
                    continue
                order = np.argsort(-np.array(e["dtScores"]), kind="mergesort")[:100]
                s = np.array(e["dtScores"])[order]
                m = e["dtMatches"][:, order] > 0
                ig = e["dtIgnore"][:, order].astype(bool)
                self.scores[k][i] = s
                self.tp[k][i] = m & ~ig
                self.valid[k][i] = ~ig
                self.npos[k, i] = int((~np.array(e["gtIgnore"], dtype=bool)).sum())

    def _flatten(self):
        """Per class: all detections sorted once by score, with their page index.
        A resample is then just a weight (multiplicity) per page."""
        self._flat = []
        for k in range(len(self.cat_ids)):
            pages = np.concatenate([np.full(len(s), i) for i, s in enumerate(self.scores[k])]).astype(int)
            s = np.concatenate(self.scores[k])
            order = np.argsort(-s, kind="mergesort")
            tp = np.concatenate(self.tp[k], axis=1)[:, order]
            va = np.concatenate(self.valid[k], axis=1)[:, order]
            self._flat.append((pages[order], tp, va & ~tp))

    def class_ap(self, sample=None):
        """AP@[.5:.95] per class for a page multiset (indices into img_ids); NaN if no GT."""
        if not hasattr(self, "_flat"):
            self._flatten()
        n_img = len(self.img_ids)
        w = np.ones(n_img) if sample is None else np.bincount(np.asarray(sample), minlength=n_img).astype(float)
        out = np.full(len(self.cat_ids), np.nan)
        for k, (pages, tp, fp) in enumerate(self._flat):
            npos = float((self.npos[k] * w).sum())
            if npos == 0:
                continue
            if pages.size == 0:
                out[k] = 0.0
                continue
            wd = w[pages]
            tpc = np.cumsum(tp * wd, axis=1)
            fpc = np.cumsum(fp * wd, axis=1)
            rc = tpc / npos
            pr = tpc / np.maximum(tpc + fpc, np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]      # precision envelope
            aps = []
            for t in range(len(IOU_THRS)):
                inds = np.searchsorted(rc[t], REC_THRS, side="left")
                q = np.zeros(len(REC_THRS))
                ok = inds < pr.shape[1]
                q[ok] = pr[t, inds[ok]]
                aps.append(q.mean())
            out[k] = float(np.mean(aps))
        return out

    def map(self, sample=None):
        return float(np.nanmean(self.class_ap(sample)))


def verify(gt_dict, dets, tol=1e-9):
    """Full-sample fast mAP must equal pycocotools' mAP@[.5:.95]."""
    from rdld.evaluation import coco_map
    a = PerImageEval(gt_dict, dets).map()
    b = coco_map(gt_dict, dets)["mAP"]
    return abs(a - b) <= tol, a, b

"""One interface over the three layout detectors (Workstream B).

Every detector returns predictions in DocLayNet's 11-class space (ids 1-11).
A prediction whose native label has no DocLayNet equivalent is dropped and
counted, never relabelled: an unmapped class would otherwise score as a false
positive and bias the ranking comparison (RQ3).
"""
import glob
import os
from collections import Counter

import numpy as np
import torch

from rdld.doclaynet import NAME_TO_ID

MODELS = {
    "yolo": ("hantian/yolo-doclaynet", "YOLO, CNN one-stage (yolo26m)"),
    "heron": ("ds4sd/docling-layout-heron", "RT-DETR v2 transformer"),
    "egret": ("ds4sd/docling-layout-egret-medium", "D-FINE transformer"),
}

# heron and egret share docling's 17-class space; the first 11 are DocLayNet's
DOCLING_TO_DOCLAYNET = {
    "caption": "Caption", "footnote": "Footnote", "formula": "Formula",
    "list_item": "List-item", "page_footer": "Page-footer",
    "page_header": "Page-header", "picture": "Picture",
    "section_header": "Section-header", "table": "Table", "text": "Text",
    "title": "Title",
}
# no DocLayNet equivalent -> dropped: document_index, code, checkbox_selected,
# checkbox_unselected, form, key_value_region

SCORE_FLOOR = 0.01   # keep low-confidence boxes: COCO mAP integrates over the full PR curve
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Prediction:
    """Boxes in xyxy pixels, DocLayNet ids, scores, plus what was dropped."""

    def __init__(self, boxes, scores, labels, raw_labels, dropped):
        self.boxes = boxes          # (N, 4) float xyxy
        self.scores = scores        # (N,)
        self.labels = labels        # (N,) DocLayNet ids 1..11
        self.raw_labels = raw_labels
        self.dropped = dropped      # Counter of native labels with no mapping


class HFDetector:
    """RT-DETR v2 / D-FINE checkpoints through transformers."""

    def __init__(self, key, threshold=SCORE_FLOOR):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.key, self.model_id = key, MODELS[key][0]
        self.threshold = threshold
        self.proc = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForObjectDetection.from_pretrained(self.model_id).to(DEVICE).eval()
        self.id2label = self.model.config.id2label

    @torch.no_grad()
    def predict(self, images):
        """images: list of PIL RGB images."""
        inputs = self.proc(images=images, return_tensors="pt").to(DEVICE)
        outputs = self.model(**inputs)
        sizes = torch.tensor([[im.height, im.width] for im in images], device=DEVICE)
        res = self.proc.post_process_object_detection(outputs, target_sizes=sizes,
                                                      threshold=self.threshold)
        preds = []
        for r in res:
            raw = [self.id2label[int(i)] for i in r["labels"]]
            keep = [n in DOCLING_TO_DOCLAYNET for n in raw]
            dropped = Counter(n for n, k in zip(raw, keep) if not k)
            k = np.array(keep, dtype=bool)
            preds.append(Prediction(
                boxes=r["boxes"].cpu().numpy()[k],
                scores=r["scores"].cpu().numpy()[k],
                labels=np.array([NAME_TO_ID[DOCLING_TO_DOCLAYNET[n]]
                                 for n, kk in zip(raw, keep) if kk], dtype=int),
                raw_labels=[n for n, kk in zip(raw, keep) if kk],
                dropped=dropped))
        return preds


class YOLODetector:
    """hantian/yolo-doclaynet through ultralytics; label space is DocLayNet's own."""

    def __init__(self, key="yolo", threshold=SCORE_FLOOR, imgsz=1024):
        from ultralytics import YOLO
        self.key, self.model_id = key, MODELS[key][0]
        self.threshold, self.imgsz = threshold, imgsz
        hub = os.path.expanduser("~/.cache/huggingface/hub/models--hantian--yolo-doclaynet")
        weights = glob.glob(os.path.join(hub, "snapshots", "*", "yolo26m-doclaynet.pt"))
        if not weights:
            from huggingface_hub import hf_hub_download
            weights = [hf_hub_download(self.model_id, "yolo26m-doclaynet.pt")]
        self.model = YOLO(weights[0])
        self.names = self.model.names

    def predict(self, images):
        res = self.model.predict(images, conf=self.threshold, imgsz=self.imgsz,
                                 device=0 if DEVICE == "cuda" else "cpu",
                                 max_det=300, verbose=False)
        preds = []
        for r in res:
            b = r.boxes
            raw = [self.names[int(c)] for c in b.cls.cpu().numpy()]
            preds.append(Prediction(
                boxes=b.xyxy.cpu().numpy(),
                scores=b.conf.cpu().numpy(),
                labels=np.array([NAME_TO_ID[n] for n in raw], dtype=int),
                raw_labels=raw, dropped=Counter()))
        return preds


def load(key, **kw):
    return YOLODetector(key, **kw) if key == "yolo" else HFDetector(key, **kw)


def to_coco_results(pred, image_id):
    """Prediction -> COCO results-format dicts (bbox as xywh)."""
    out = []
    for (x1, y1, x2, y2), s, c, raw in zip(pred.boxes, pred.scores, pred.labels, pred.raw_labels):
        out.append({"image_id": int(image_id), "category_id": int(c),
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(s), "raw_label": raw})
    return out

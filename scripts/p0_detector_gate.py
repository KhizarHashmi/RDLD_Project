"""
p0_detector_gate.py -- (Khizar, Workstream B) GATE C: can we run a layout detector at all?

RoDLA's own model needs mmdetection + mmcv + compiled DCNv3 CUDA ops.
We avoid that entirely by using an INDEPENDENT released detector.

Candidates tried in order (first that works wins):
  1. docling-layout-heron   (RT-DETR, HuggingFace, mAP 0.69 on DocLayNet)
  2. docling-layout-egret   (alternative size)
  3. DocLayout-YOLO         (YOLOv10-based, if ultralytics present)

Run:
  python scripts/p0_detector_gate.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

from rdld import config

IMG = config.TEST_PAGE
OUT = config.out_dir("p0", "detections")

import torch
import numpy as np
import cv2

print("=" * 70)
print(" GATE C: LAYOUT DETECTOR FEASIBILITY")
print("=" * 70)
print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CANDIDATES = [
    ("ds4sd/docling-layout-heron", "RT-DETR heron (DocLayNet mAP 0.69)"),
    ("ds4sd/docling-layout-egret-medium", "RT-DETR egret medium"),
    ("ds4sd/docling-layout-old", "docling legacy layout"),
]


def try_hf_detector(model_id, label):
    """Try loading an HF object-detection model for document layout."""
    from transformers import AutoImageProcessor, AutoModelForObjectDetection
    from PIL import Image

    print(f"\n--- trying {model_id}")
    t0 = time.time()
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForObjectDetection.from_pretrained(model_id).to(DEVICE).eval()
    load_s = time.time() - t0
    print(f"    loaded in {load_s:.1f}s")

    img = Image.open(IMG).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    infer_s = time.time() - t0

    res = proc.post_process_object_detection(
        outputs, target_sizes=torch.tensor([[img.height, img.width]]).to(DEVICE),
        threshold=0.3)[0]

    id2label = model.config.id2label
    n = len(res["scores"])
    print(f"    inference {infer_s*1000:.0f} ms -> {n} regions @ conf>0.3")
    if torch.cuda.is_available():
        print(f"    VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    dets = []
    for score, lab, box in zip(res["scores"], res["labels"], res["boxes"]):
        dets.append({
            "label": id2label.get(int(lab), str(int(lab))),
            "score": float(score),
            "box": [float(x) for x in box],
        })
    for d in sorted(dets, key=lambda d: -d["score"])[:12]:
        b = d["box"]
        print(f"      {d['label']:<18} {d['score']:.3f}  "
              f"[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

    # visualise
    vis = cv2.imread(str(IMG))
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, f"{d['label'][:12]} {d['score']:.2f}",
                    (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1)
    safe = model_id.replace("/", "_")
    vis_path = OUT / f"{safe}.png"
    cv2.imwrite(str(vis_path), vis)
    print(f"    visualisation -> {vis_path.name}")

    return {"model_id": model_id, "n_regions": n, "infer_ms": infer_s * 1000,
            "labels": sorted(set(d["label"] for d in dets)), "dets": dets}


results = []
for model_id, label in CANDIDATES:
    try:
        r = try_hf_detector(model_id, label)
        results.append(r)
        break                      # first success is enough for the gate
    except Exception as e:
        print(f"    [FAIL] {type(e).__name__}: {str(e)[:110]}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)

if results:
    r = results[0]
    ok_regions = r["n_regions"] >= 2
    ok_speed = r["infer_ms"] < 5000
    print(f"  [{'PASS' if True else ''}] detector loads and runs      {r['model_id']}")
    print(f"  [{'PASS' if ok_regions else 'WARN'}] produces regions            {r['n_regions']} found")
    print(f"  [{'PASS' if ok_speed else 'WARN'}] inference speed usable       {r['infer_ms']:.0f} ms/page")
    print(f"\n  classes detected: {r['labels']}")
    est = r["infer_ms"] / 1000 * 500 * 84 / 3600
    print(f"\n  Scale estimate: 500 images x 84 pair-configs = 42,000 inferences")
    print(f"                  ~= {est:.1f} GPU-hours at {r['infer_ms']:.0f} ms/page")
    print("""
  ==> GATE C PASSED. An independent layout detector runs on this machine
      with NO mmdetection and NO compiled CUDA ops.

  NEXT: download DocLayNet val split for ground-truth annotations,
        then measure mAP under single vs composite perturbations.
""")
else:
    print("""  [FAIL] No detector loaded.

  Fallbacks to try:
    - pip install ultralytics, then DocLayout-YOLO weights from HF
    - pip install paddlepaddle paddleocr, then PP-DocLayout
    - check network/proxy for HuggingFace access
""")

print("LOOK AT the visualisation in RDLD_data/results/p0/detections/ -- do the boxes")
print("actually land on headline / paragraph / column regions?")

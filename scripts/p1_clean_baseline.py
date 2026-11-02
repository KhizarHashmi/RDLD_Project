"""
p1_clean_baseline.py -- P1 (Khizar, Workstream B): clean-condition baseline.

For each of the 3 detectors on the 300 clean DocLayNet pages:
  1. inference -> prediction cache  RDLD_data/predictions/<model>/clean.json
  2. mAP@[.5:.95], mAP@.5, mAP@.75, per-class AP, per-document-category mAP
  3. page-level bootstrap 95% CI on mAP (the noise floor for later deltas)
  4. label-instability floor: cross-label duplicates on clean input
  5. effect of class-agnostic dedup on mAP (decides the dedup policy)
  6. run-to-run determinism (second pass over 50 pages)

Outputs: RDLD_data/results/p1/baseline/

Run:
  python scripts/p1_clean_baseline.py
  N_BOOT=200 python scripts/p1_clean_baseline.py     # quicker CI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from PIL import Image

from rdld import config, detectors, evaluation, fastcoco
from rdld.doclaynet import CATEGORIES, DOC_CATEGORIES, load_coco

BATCH = 8
N_BOOT = int(os.getenv("N_BOOT", "1000"))
OP_THRESHOLDS = (0.3, 0.5)          # operating points for the instability measurement
OUT = config.out_dir("p1", "baseline")

gt = load_coco()
images = gt["images"]
manifest = pd.read_csv(config.MANIFEST, dtype={"id": str})
cat_of = dict(zip(manifest["id"].astype(int), manifest["doc_category"]))


def run(det, ims):
    """Inference over ims in batches; returns COCO results, dropped Counter, ms/page."""
    det.predict([Image.open(config.IMAGES / ims[0]["file_name"]).convert("RGB")])  # warm-up
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dets, dropped, t0 = [], Counter(), time.time()
    for i in range(0, len(ims), BATCH):
        chunk = ims[i:i + BATCH]
        pil = [Image.open(config.IMAGES / im["file_name"]).convert("RGB") for im in chunk]
        for p, im in zip(det.predict(pil), chunk):
            dets.extend(detectors.to_coco_results(p, im["id"]))
            dropped.update(p.dropped)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return dets, dropped, (time.time() - t0) / len(ims) * 1000


summary, per_class, per_cat, instab, boots = [], [], [], [], []
for key, (model_id, family) in detectors.MODELS.items():
    print(f"\n=== {key}  {model_id}  ({family})")
    det = detectors.load(key)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    dets, dropped, ms = run(det, images)
    vram = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else float("nan")

    pdir = config.PREDICTIONS / key
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "clean.json").write_text(json.dumps(dets), encoding="utf-8")

    # --- accuracy
    r = evaluation.coco_map(gt, dets)
    dd = evaluation.agnostic_dedup(dets, iou_thr=0.9)
    r_dd = evaluation.coco_map(gt, dd)
    print(f"    mAP {r['mAP']:.4f}  mAP50 {r['mAP50']:.4f}  mAP75 {r['mAP75']:.4f}"
          f"   | dedup mAP {r_dd['mAP']:.4f}   {ms:.0f} ms/page  {vram:.2f} GB")

    # --- bootstrap CI
    t = time.time()
    pe = fastcoco.PerImageEval(gt, dets)
    rng = np.random.default_rng(0)
    bs = np.array([pe.map(rng.integers(0, len(images), len(images))) for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"    bootstrap ({N_BOOT}) 95% CI [{lo:.4f}, {hi:.4f}]  sd {bs.std(ddof=1):.4f}"
          f"  ({time.time() - t:.0f}s)")
    boots.extend({"model": key, "b": i, "mAP": v} for i, v in enumerate(bs))

    summary.append({
        "model": key, "model_id": model_id, "family": family,
        "mAP": r["mAP"], "mAP50": r["mAP50"], "mAP75": r["mAP75"],
        "mAP_ci_lo": lo, "mAP_ci_hi": hi, "mAP_boot_sd": bs.std(ddof=1),
        "mAP_dedup": r_dd["mAP"], "dedup_delta": r_dd["mAP"] - r["mAP"],
        "n_dets_floor": len(dets), "ms_per_page": ms, "vram_gb": vram,
        "dropped_unmapped_floor": sum(dropped.values()),
        "dropped_detail": "; ".join(f"{k}={v}" for k, v in dropped.most_common()),
    })
    for cls, ap in r["per_class"].items():
        per_class.append({"model": key, "class": cls, "AP": ap, "AP_dedup": r_dd["per_class"][cls]})

    # --- per document category
    for c in DOC_CATEGORIES:
        ids = [im["id"] for im in images if cat_of[im["id"]] == c]
        rc = evaluation.coco_map(gt, dets, ids)
        per_cat.append({"model": key, "doc_category": c, "mAP": rc["mAP"], "mAP50": rc["mAP50"]})

    # --- instability floor
    for thr in OP_THRESHOLDS:
        frac, n_dup, n_tot = evaluation.cross_label_duplicates(dets, thr, iou_thr=0.9)
        instab.append({"model": key, "score_thr": thr, "iou_thr": 0.9,
                       "n_preds": n_tot, "n_cross_label_dup": n_dup, "frac": frac})
        print(f"    cross-label duplicates @score>={thr}: {n_dup}/{n_tot} = {frac:.2%}")

    # --- determinism: second pass over the first 48 pages, compared as SETS
    # (transformer heads emit top-k queries; near-tied scores may reorder between
    #  runs, so a position-by-position comparison would report false mismatches)
    sub = images[:48]
    a, _, _ = run(det, sub)
    ids = {im["id"] for im in sub}
    a_ref = [d for d in dets if d["image_id"] in ids]
    key = lambda d: (d["image_id"], d["category_id"], round(d["score"], 4), tuple(round(v, 1) for v in d["bbox"]))
    same_set = sorted(map(key, a)) == sorted(map(key, a_ref))
    m_a = evaluation.coco_map(gt, a, sorted(ids))["mAP"]
    m_ref = evaluation.coco_map(gt, a_ref, sorted(ids))["mAP"]
    summary[-1].update({"rerun_identical_set": same_set, "rerun_mAP_diff": m_a - m_ref})
    print(f"    re-run on 48 pages: identical prediction set={same_set}  mAP diff={m_a - m_ref:.2e}")

    del det
    torch.cuda.empty_cache()

# ------------------------------------------------------------------ write
S = pd.DataFrame(summary); S.to_csv(OUT / "summary.csv", index=False)
PC = pd.DataFrame(per_class); PC.to_csv(OUT / "per_class_ap.csv", index=False)
PK = pd.DataFrame(per_cat); PK.to_csv(OUT / "per_category_map.csv", index=False)
IN = pd.DataFrame(instab); IN.to_csv(OUT / "instability.csv", index=False)
pd.DataFrame(boots).to_csv(OUT / "bootstrap_map.csv", index=False)

gt_counts = Counter(CATEGORIES[a["category_id"]] for a in gt["annotations"])

# ------------------------------------------------------------------ figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

order = sorted(CATEGORIES.values(), key=lambda c: -gt_counts[c])
piv = PC.pivot(index="class", columns="model", values="AP").loc[order, list(detectors.MODELS)]
fig, ax = plt.subplots(figsize=(11, 4.6), dpi=160)
piv.plot.bar(ax=ax, width=0.8, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_xticklabels([f"{c}\n(n={gt_counts[c]})" for c in order], rotation=0, fontsize=8)
ax.set_ylabel("AP@[.5:.95]"); ax.set_xlabel(""); ax.set_ylim(0, 1)
ax.set_title("P1 clean baseline — per-class AP on 300 DocLayNet pages")
ax.spines[["top", "right"]].set_visible(False); ax.legend(title="detector", frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_per_class_ap.png", facecolor="white"); plt.close()

piv = PK.pivot(index="doc_category", columns="model", values="mAP")[list(detectors.MODELS)]
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
piv.plot.bar(ax=ax, width=0.8, color=["#4C78A8", "#F58518", "#54A24B"])
ax.set_xticklabels([c.replace("_", "\n") for c in piv.index], rotation=0, fontsize=8)
ax.set_ylabel("mAP@[.5:.95]"); ax.set_xlabel(""); ax.set_ylim(0, 1)
ax.set_title("P1 clean baseline — mAP by document category (50 pages each)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(title="detector", frameon=False)
plt.tight_layout(); plt.savefig(OUT / "fig_per_category_map.png", facecolor="white"); plt.close()

print("\n" + "=" * 70)
print(S[["model", "mAP", "mAP_ci_lo", "mAP_ci_hi", "mAP50", "mAP75", "mAP_dedup",
         "ms_per_page", "dropped_unmapped_floor"]].to_string(index=False))
print(f"\nwritten -> {OUT}")

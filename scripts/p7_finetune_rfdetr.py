"""
p7_finetune_rfdetr.py -- P7 (Khizar, Workstream B): fine-tune a DINOv2-backbone
detector (RF-DETR medium) on DocLayNet, for inclusion in the robustness study.

Every other detector is a released layout checkpoint; RF-DETR has none, so it is
fine-tuned here from its COCO weights (Roboflow/rf-detr-medium: DINOv2-S windowed
backbone, 4-layer DETR decoder). Training is deliberately plain so the result is
comparable to the released detectors:

  - CLEAN training only -- no degradation augmentation (the study measures robustness
    that a detector did not train for)
  - no flips (mirrored text is not a document), no mosaic / multi-scale
  - data: Workstream A's p7_prepare_train_data.py (train = DocLayNet train subset;
    select = validation pages outside the 300 study pages, used only to pick the
    checkpoint). The study pages are never seen.

Optimiser AdamW (lr 1e-4 head/decoder, 1e-5 backbone, wd 1e-4), cosine schedule with
warm-up, bf16 autocast, batch 4 x grad-accumulation 4 (effective 16), resolution 672.
Checkpoint selected by COCO mAP@[.5:.95] on the select set after every epoch.

Outputs: RDLD_data/finetune/rfdetr/{best/, last/, train_log.csv}
         RDLD_data/results/p7/finetune_summary.csv

Resumable: after every epoch the weights plus optimiser / scheduler / RNG state are
written to last/ (via a temporary directory, so an interrupted save never leaves a
half-written checkpoint). Re-running the script continues from the last finished epoch;
the data order of epoch e is seeded with SEED + e, so a resumed run sees the same batches
as an uninterrupted one. RESUME=0 forces a fresh start.

Run:
  python scripts/p7_finetune_rfdetr.py
  EPOCHS=1 MAX_STEPS=50 python scripts/p7_finetune_rfdetr.py     # smoke test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import math
import os
import shutil
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # base weights are cached; never touch the network

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rdld import config
from rdld.doclaynet import CATEGORIES
from rdld.evaluation import coco_map

BASE = os.getenv("BASE", "Roboflow/rf-detr-medium")
RES = int(os.getenv("RES", "672"))
EPOCHS = int(os.getenv("EPOCHS", "12"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "0"))          # >0: stop early (smoke test)
RESUME = os.getenv("RESUME", "1") == "1"
BATCH, ACCUM = int(os.getenv("BATCH", "4")), int(os.getenv("ACCUM", "4"))
LR, LR_BACKBONE, WD = 1e-4, 1e-5, 1e-4
SEED = 20261104
DATA = config.DATA_ROOT / "finetune"
OUT = Path(os.getenv("OUT_DIR", str(DATA / "rfdetr")))
RES_DIR = OUT if os.getenv("OUT_DIR") else config.out_dir("p7")
SMOKE = int(os.getenv("SMOKE", "0"))                  # >0: use only the first SMOKE pages (test runs)
DEV = "cuda"
ID2LABEL = {i - 1: n for i, n in CATEGORIES.items()}          # model ids 0..10 -> DocLayNet names


class CocoPages(Dataset):
    def __init__(self, split):
        self.root = DATA / split / "images"
        coco = json.loads((DATA / split / "annotations.json").read_text(encoding="utf-8"))
        self.images = coco["images"][:SMOKE] if SMOKE else coco["images"]
        self.by = {}
        for a in coco["annotations"]:
            self.by.setdefault(a["image_id"], []).append(a)
        self.coco = coco

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        im = self.images[i]
        anns = [{"bbox": a["bbox"], "category_id": a["category_id"] - 1, "area": a["area"], "iscrowd": 0}
                for a in self.by.get(im["id"], [])]
        return Image.open(self.root / im["file_name"]).convert("RGB"), {"image_id": im["id"], "annotations": anns}


class Collate:
    """Top-level (picklable) so DataLoader workers can use it under Windows' spawn start method."""

    def __init__(self, proc):
        self.proc = proc

    def __call__(self, batch):
        imgs, anns = zip(*batch)
        enc = self.proc(images=list(imgs), annotations=list(anns), return_tensors="pt")
        return enc["pixel_values"], enc.get("pixel_mask"), enc["labels"]


@torch.no_grad()
def evaluate(model, proc, ds, bs=8):
    model.eval()
    res = []
    for j in range(0, len(ds), bs):
        imgs = [ds[k][0] for k in range(j, min(j + bs, len(ds)))]
        ids = [ds.images[k]["id"] for k in range(j, min(j + bs, len(ds)))]
        x = proc(images=imgs, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**x)
        sizes = torch.tensor([[im.height, im.width] for im in imgs], device=DEV)
        for r, iid in zip(proc.post_process_object_detection(out, target_sizes=sizes, threshold=0.01), ids):
            for (x1, y1, x2, y2), s, c in zip(r["boxes"].float().tolist(), r["scores"].float().tolist(), r["labels"].tolist()):
                res.append({"image_id": iid, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": s})
    model.train()
    return coco_map(ds.coco, res)


def save_checkpoint(model, proc, opt, sched, state):
    """Write last/ atomically: weights + training state into a temp dir, then swap it in."""
    tmp, last, old = OUT / "last_tmp", OUT / "last", OUT / "last_old"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp); proc.save_pretrained(tmp)
    torch.save({**state, "opt": opt.state_dict(), "sched": sched.state_dict(),
                "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
                "rng_np": np.random.get_state()}, tmp / "train_state.pt")
    if last.exists():
        shutil.rmtree(old, ignore_errors=True)
        last.rename(old)
    tmp.rename(last)
    shutil.rmtree(old, ignore_errors=True)


def main():
    from transformers import AutoImageProcessor, RfDetrForObjectDetection
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not (OUT / "last").exists() and (OUT / "last_old" / "train_state.pt").exists():
        (OUT / "last_old").rename(OUT / "last")        # interrupted between the two renames
    ck = OUT / "last" / "train_state.pt"
    resume = RESUME and ck.exists()
    src = str(OUT / "last") if resume else BASE
    proc = AutoImageProcessor.from_pretrained(src, size={"height": RES, "width": RES})
    model = RfDetrForObjectDetection.from_pretrained(
        src, id2label=ID2LABEL, label2id={v: k for k, v in ID2LABEL.items()},
        ignore_mismatched_sizes=not resume).to(DEV)
    model.train()

    train, sel = CocoPages("train"), CocoPages("select")
    gen = torch.Generator()
    loader = DataLoader(train, batch_size=BATCH, shuffle=True, num_workers=4, collate_fn=Collate(proc),
                        persistent_workers=True, generator=gen)
    bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": LR_BACKBONE}, {"params": rest, "lr": LR}], weight_decay=WD)
    steps_total = EPOCHS * math.ceil(len(loader) / ACCUM)
    if MAX_STEPS:
        steps_total = min(steps_total, MAX_STEPS)
    warm = max(1, int(0.03 * steps_total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm))))
    print(f"base {BASE} @ {RES}px | train {len(train)} pages, select {len(sel)} | "
          f"{EPOCHS} epochs, {steps_total} optimiser steps (batch {BATCH} x accum {ACCUM})", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    log, best, step, start, t_prev = [], -1.0, 0, 1, 0.0
    if resume:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        torch.set_rng_state(st["rng_torch"]); torch.cuda.set_rng_state_all(st["rng_cuda"])
        np.random.set_state(st["rng_np"])
        log, best, step, start, t_prev = st["log"], st["best"], st["step"], st["epoch"] + 1, st["elapsed_s"]
        print(f"RESUMED after epoch {st['epoch']} (step {step}, best select mAP {100 * best:.2f})", flush=True)
    t0 = time.time() - t_prev                          # elapsed_s counts training time across restarts
    for ep in range(start, EPOCHS + 1):
        gen.manual_seed(SEED + ep)                     # epoch-specific data order, identical on resume
        run = 0.0
        for i, (pix, mask, labels) in enumerate(loader, 1):
            labels = [{k: v.to(DEV) for k, v in l.items()} for l in labels]
            kw = {"pixel_values": pix.to(DEV), "labels": labels}
            if mask is not None:
                kw["pixel_mask"] = mask.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**kw).loss / ACCUM
            loss.backward()
            run += loss.item() * ACCUM
            if i % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                if step % 50 == 0:
                    print(f"  ep {ep} step {step}/{steps_total} loss {run / i:.4f} "
                          f"lr {sched.get_last_lr()[1]:.2e} {time.time() - t0:.0f}s", flush=True)
                if MAX_STEPS and step >= MAX_STEPS:
                    break
        m = evaluate(model, proc, sel)
        row = {"epoch": ep, "step": step, "train_loss": run / max(i, 1), "select_mAP": m["mAP"],
               "select_mAP50": m["mAP50"], "elapsed_s": time.time() - t0}
        log.append(row)
        pd.DataFrame(log).to_csv(OUT / "train_log.csv", index=False)
        print(f"epoch {ep}: loss {row['train_loss']:.4f}  select mAP {100 * m['mAP']:.2f} "
              f"(mAP50 {100 * m['mAP50']:.2f})  {row['elapsed_s']:.0f}s", flush=True)
        if m["mAP"] > best:
            best = m["mAP"]
            shutil.rmtree(OUT / "best_tmp", ignore_errors=True)
            model.save_pretrained(OUT / "best_tmp"); proc.save_pretrained(OUT / "best_tmp")
            shutil.rmtree(OUT / "best", ignore_errors=True)
            (OUT / "best_tmp").rename(OUT / "best")
            print(f"  new best -> {OUT / 'best'}", flush=True)
        save_checkpoint(model, proc, opt, sched, {"epoch": ep, "step": step, "best": best, "log": log,
                                                  "elapsed_s": row["elapsed_s"]})
        if MAX_STEPS and step >= MAX_STEPS:
            break

    L = pd.DataFrame(log)
    pd.DataFrame([{"base": BASE, "resolution": RES, "train_pages": len(train), "select_pages": len(sel),
                   "epochs_run": len(L), "best_epoch": int(L.loc[L.select_mAP.idxmax(), "epoch"]),
                   "best_select_mAP": float(L.select_mAP.max()), "train_hours": float(L.elapsed_s.iloc[-1] / 3600)}]
                 ).to_csv(RES_DIR / "finetune_summary.csv", index=False)
    print(f"\nbest select mAP {100 * best:.2f}; written -> {OUT}")


if __name__ == "__main__":
    main()

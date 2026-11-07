"""
p2_image_quality.py -- P2 (Alishba, Workstream A): how much does each configuration
change the image, independent of any detector?

For every (configuration, page): PSNR, MS-SSIM, CW-SSIM against the clean page.
Per configuration: means, and the image-quality Perturbation Effect

    mPE_IQA(s,p) = 100 * mean over {MS-SSIM, CW-SSIM} of (1 - metric)

RoDLA's mPE (Eq. 3) also adds the mAP degradation of a fixed reference detector
(Faster R-CNN). We do not run that detector, and using one of our three would make
mRD partly divide a model's loss by itself, so the primary mPE here is IQA-only and
therefore model-independent -- RoDLA's own first design requirement. A
RoDLA-style variant is computed in Workstream B once detector results exist.

Outputs: RDLD_data/results/p2/image_quality/{iq_per_page.csv, iq_per_config.csv,
         sanity.csv, fig_iq_by_degradation.png}

Run:
  python scripts/p2_image_quality.py            # needs the cache from p2_generate_cache.py
  python scripts/p2_image_quality.py --sanity   # metric sanity checks only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch

from rdld import config
from rdld.doclaynet import load_coco
from rdld.image_quality import all_metrics

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 6
OUT = config.out_dir("p2", "image_quality")


def tensor(imgs):
    return torch.from_numpy(np.stack(imgs)).to(DEV).permute(0, 3, 1, 2).float()


def sanity(images):
    """Identity gives the maximum; scores fall as severity rises."""
    from rdld import degradations as D
    clean = cv2.imread(str(config.IMAGES / images[0]["file_name"]))
    rows = []
    for name in ("defocus", "texture", "ink_bleeding"):
        for s in (0, 1, 2, 3):
            deg = clean if s == 0 else D.apply_one(clean, name, s, images[0]["id"])
            m = all_metrics(tensor([clean]), tensor([deg]))
            rows.append({"degradation": name, "severity": s, **{k: float(v[0]) for k, v in m.items()}})
    S = pd.DataFrame(rows)
    S.to_csv(OUT / "sanity.csv", index=False)
    print(S.round(4).to_string(index=False))
    ok = True
    for name, g in S.groupby("degradation"):
        for m in ("ms_ssim", "cw_ssim"):
            v = g.sort_values("severity")[m].to_numpy()
            mono = bool(np.all(np.diff(v) <= 1e-6))
            ident = abs(v[0] - 1) < 1e-4
            ok &= mono and ident
            print(f"  [{'PASS' if mono and ident else 'FAIL'}] {name:13s} {m}: identity={v[0]:.4f} monotone={mono}")
    return ok


def main():
    gt = load_coco()
    images = gt["images"]
    if not sanity(images):
        raise SystemExit("metric sanity failed")
    if "--sanity" in sys.argv:
        return
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    configs = configs[configs.kind != "clean"]

    clean = {im["file_name"]: cv2.imread(str(config.IMAGES / im["file_name"])) for im in images}
    parts = OUT / "per_config"                       # one file per configuration -> resumable
    parts.mkdir(exist_ok=True)
    t0 = time.time()
    pool = ThreadPoolExecutor(8)
    for i, cfg in enumerate(configs.config, 1):
        if (parts / f"{cfg}.csv").exists():
            continue
        rows = []
        names = [im["file_name"] for im in images]
        deg = list(pool.map(lambda f: cv2.imread(str(config.DEGRADED / cfg / f)), names))
        for j in range(0, len(names), BATCH):
            m = all_metrics(tensor([clean[f] for f in names[j:j + BATCH]]), tensor(deg[j:j + BATCH]))
            for k, f in enumerate(names[j:j + BATCH]):
                rows.append({"config": cfg, "file_name": f, **{n: float(v[k]) for n, v in m.items()}})
        pd.DataFrame(rows).to_csv(parts / f"{cfg}.csv", index=False)
        print(f"  {i:3d}/{len(configs)} {cfg:40s} {time.time() - t0:6.0f}s")

    P = pd.concat([pd.read_csv(parts / f"{c}.csv") for c in configs.config], ignore_index=True)
    P.to_csv(OUT / "iq_per_page.csv", index=False)
    C = P.groupby("config").agg(psnr=("psnr", "mean"), ms_ssim=("ms_ssim", "mean"),
                                cw_ssim=("cw_ssim", "mean"), ms_ssim_sd=("ms_ssim", "std")).reset_index()
    C["mPE_iqa"] = 100 * ((1 - C.ms_ssim) + (1 - C.cw_ssim)) / 2
    C = configs.merge(C, on="config")
    C.to_csv(OUT / "iq_per_config.csv", index=False)

    s = C[C.kind == "single"]
    print("\nsingle degradations, mPE_IQA by severity:")
    print(s.pivot(index="a", columns="severity", values="mPE_iqa").round(2).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    piv = s.pivot(index="a", columns="severity", values="mPE_iqa")
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 4), dpi=160)
    piv.plot.bar(ax=ax, width=0.8, color=["#9ecae1", "#4292c6", "#08519c"])
    ax.set_ylabel("mPE_IQA  (0 = unchanged)"); ax.set_xlabel("")
    ax.set_xticklabels(piv.index, rotation=0, fontsize=8)
    ax.set_title("Image-level perturbation effect of each single degradation (300 pages)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(title="severity", frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "fig_iq_by_degradation.png", facecolor="white"); plt.close()
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

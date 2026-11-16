"""
p0_verify_perturbations.py -- P0 FEASIBILITY GATE (Alishba, Workstream A) for the compositionality project.

Answers three questions:
  1. Do RoDLA's perturbation functions run on this machine?
  2. Which ones need heavy deps (detectron2) and which are standalone?
  3. CAN THEY BE COMPOSED?  <-- the whole project depends on this

Run:
  python scripts/p0_verify_perturbations.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import time
import itertools

from rdld import config, third_party

third_party.enable()
os.chdir(config.THIRD_PARTY / "rodla_perturbation")   # modules use relative paths (Arialn.ttf)

import cv2
import numpy as np

IMG = config.TEST_PAGE
OUT = config.out_dir("p0", "perturbed")

print("=" * 70)
print(" RoDLA PERTURBATION FEASIBILITY CHECK")
print("=" * 70)

img = cv2.imread(str(IMG))
if img is None:
    raise SystemExit(f"Could not read {IMG}")
print(f"\nTest image: {IMG.name}  {img.shape[1]}x{img.shape[0]}")

# ---------------------------------------------------------------- imports
print("\n--- MODULE IMPORTS ---")
available = {}

def try_import(label, fn_names, module):
    global available
    try:
        mod = __import__(module)
        for n in fn_names:
            available[n] = getattr(mod, n)
        print(f"  [OK]   {module:16s} -> {', '.join(fn_names)}")
        return True
    except Exception as e:
        print(f"  [FAIL] {module:16s} {type(e).__name__}: {str(e)[:60]}")
        return False

try_import("blur", ["apply_defocus", "apply_vibration"], "blur")
try_import("noise", ["apply_speckle", "apply_texture"], "noise")
try_import("content", ["add_watermark", "add_image_background"], "content")
try_import("inconsistency",
           ["apply_ink_holdout", "apply_ink_bleeding", "apply_illumination"],
           "inconsistency")
# spatial needs detectron2 + imgaug and also transforms ANNOTATIONS
try_import("spatial", ["apply_rotation", "apply_keystoning", "apply_warping"], "spatial")

print(f"\n  {len(available)} perturbation functions loaded")

# ---------------------------------------------------------------- single
print("\n--- SINGLE PERTURBATIONS (degree 1,2,3) ---")
IMAGE_ONLY = ["apply_defocus", "apply_vibration", "apply_speckle", "apply_texture",
              "add_watermark", "apply_ink_holdout", "apply_ink_bleeding",
              "apply_illumination"]

working = {}
for name in IMAGE_ONLY:
    if name not in available:
        continue
    fn = available[name]
    ok_degrees = []
    for deg in (1, 2, 3):
        try:
            t0 = time.time()
            out = fn(img.copy(), degree=deg)
            dt = time.time() - t0
            if out is None or not isinstance(out, np.ndarray):
                raise ValueError("no array returned")
            cv2.imwrite(str(OUT / f"{name}_d{deg}.png"), out)
            ok_degrees.append(deg)
            if deg == 2:
                working[name] = fn
                timing = dt
        except Exception as e:
            print(f"  [FAIL] {name:22s} d{deg}: {type(e).__name__}: {str(e)[:45]}")
    if ok_degrees:
        print(f"  [OK]   {name:22s} degrees {ok_degrees}  ({timing*1000:.0f} ms @ d2)")

print(f"\n  {len(working)} image-only perturbations WORKING")

# ---------------------------------------------------------------- compose
print("\n" + "=" * 70)
print(" THE DECISIVE TEST: CAN PERTURBATIONS BE COMPOSED?")
print("=" * 70)
print("""
RoDLA applies ONE perturbation per dataset (verified in apply_perturbations.py:
  for key in key_list: for degree in degree_list: create_pert_dataset(key, degree)
Real scanned documents have SEVERAL simultaneously. If composition works here,
the compositionality project is feasible.
""")

names = sorted(working)
pairs = list(itertools.combinations(names, 2))
print(f"Testing all {len(pairs)} pairs from {len(names)} perturbations at degree 2...\n")

ok_pairs, bad_pairs = [], []
for a, b in pairs:
    try:
        stage1 = working[a](img.copy(), degree=2)
        stage2 = working[b](stage1.copy(), degree=2)
        if stage2 is None or not isinstance(stage2, np.ndarray):
            raise ValueError("no array")
        if stage2.shape != img.shape:
            raise ValueError(f"shape changed {img.shape}->{stage2.shape}")
        ok_pairs.append((a, b))
        cv2.imwrite(str(OUT / f"COMPOSED_{a}__{b}.png"), stage2)
    except Exception as e:
        bad_pairs.append((a, b, f"{type(e).__name__}: {str(e)[:40]}"))

print(f"  COMPOSABLE : {len(ok_pairs)}/{len(pairs)} pairs")
for a, b, err in bad_pairs:
    print(f"  [FAIL] {a} + {b}  {err}")

# triples, to check depth
print("\n--- TRIPLE COMPOSITION (sample) ---")
triples = list(itertools.combinations(names, 3))[:5]
ok_triples = 0
for combo in triples:
    try:
        cur = img.copy()
        for n in combo:
            cur = working[n](cur.copy(), degree=2)
        ok_triples += 1
        cv2.imwrite(str(OUT / f"TRIPLE_{'__'.join(combo)}.png"), cur)
    except Exception as e:
        print(f"  [FAIL] {' + '.join(combo)}: {type(e).__name__}")
print(f"  {ok_triples}/{len(triples)} sampled triples composed successfully")

# ---------------------------------------------------------------- scale
print("\n--- EXPERIMENT SPACE SIZE ---")
n = len(names)
print(f"  single          : {n} x 3 degrees      = {n*3}")
print(f"  pairs           : {n*(n-1)//2} x 3 degrees      = {n*(n-1)//2*3}")
print(f"  pairs, deg pairs: {n*(n-1)//2} x 9 combos      = {n*(n-1)//2*9}")
print(f"  triples         : {n*(n-1)*(n-2)//6} x 3 degrees      = {n*(n-1)*(n-2)//6*3}")

# ---------------------------------------------------------------- verdict
print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
gate1 = len(working) >= 6
gate2 = len(ok_pairs) >= (len(pairs) * 0.8 if pairs else 0)
gate3 = ok_triples >= 3

print(f"  [{'PASS' if gate1 else 'FAIL'}] >=6 perturbations work standalone      ({len(working)})")
print(f"  [{'PASS' if gate2 else 'FAIL'}] >=80% of pairs compose                  ({len(ok_pairs)}/{len(pairs)})")
print(f"  [{'PASS' if gate3 else 'FAIL'}] triples compose                          ({ok_triples}/{len(triples)})")

if gate1 and gate2 and gate3:
    print("""
  ==> FEASIBLE. Composition works with image-only perturbations and needs
      NO detectron2, NO mmdetection, NO GPU.

      Next: get a real layout dataset (DocLayNet/PubLayNet val) + a released
      layout detector, then measure whether degradation is additive.
""")
else:
    print("\n  ==> Investigate the failures above before proceeding.\n")

print(f"Perturbed samples written to: {OUT}")
print("LOOK AT THEM. Confirm the degradations resemble real scan artefacts.")

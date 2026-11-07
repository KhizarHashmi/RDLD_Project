"""Seeded, composable image-only degradations (Workstream A).

The eight RoDLA functions are called unchanged. This module adds what RoDLA
does not have: applying several of them to one page.

Three rules make the composition experiment valid:

1. Fixed physical order. A compound is applied in the order the artefacts arise
   in a real document's life: paper -> printing -> content overlay -> capture
   lighting -> optics. `apply_texture` re-renders the page on synthetic paper
   (it renormalises intensities and drops colour), so applying it after another
   degradation would erase that degradation; paper therefore comes first.
2. Common random numbers. Every function call is seeded from
   (degradation, severity, page, replicate). Degradation A inside the pair A+B
   draws exactly the same randomness as the single-A condition, so the
   interaction term compares like with like instead of adding sampling noise.
3. Image-only. Boxes never move, so the clean ground truth applies to every
   degraded page.
"""
import hashlib
import importlib
import itertools
import random

import numpy as np

from rdld import third_party

third_party.enable()

# name -> (module, function), listed in physical order of occurrence
DEGRADATIONS = {
    "texture": ("noise", "apply_texture"),               # paper fibres
    "ink_bleeding": ("inconsistency", "apply_ink_bleeding"),  # printing: ink spread
    "ink_holdout": ("inconsistency", "apply_ink_holdout"),    # printing: ink missing
    "speckle": ("noise", "apply_speckle"),               # printing: ink clumps / blotches
    "watermark": ("content", "add_watermark"),           # content overlay
    "illumination": ("inconsistency", "apply_illumination"),  # capture: glare / shadow
    "defocus": ("blur", "apply_defocus"),                # capture: optics
    "vibration": ("blur", "apply_vibration"),            # capture: motion
}
ORDER = {n: i for i, n in enumerate(DEGRADATIONS)}
SEVERITIES = (1, 2, 3)

_fns = {}


def _fn(name):
    if name not in _fns:
        mod, fn = DEGRADATIONS[name]
        _fns[name] = getattr(importlib.import_module(mod), fn)
    return _fns[name]


def seed_for(name, severity, page_id, rep=0):
    h = hashlib.md5(f"{name}|{severity}|{page_id}|{rep}".encode()).hexdigest()
    return int(h[:8], 16)


def apply_one(img, name, severity, page_id, rep=0):
    """One degradation, deterministic given (name, severity, page, rep). img: BGR uint8."""
    s = seed_for(name, severity, page_id, rep)
    np.random.seed(s)
    random.seed(s)
    out = _fn(name)(img.copy(), degree=severity)
    if out.shape != img.shape:
        raise ValueError(f"{name}@{severity} changed shape {img.shape} -> {out.shape}")
    return out


def steps_of(config_id):
    """'defocus@2+texture@2' -> [('defocus', 2), ('texture', 2)]; 'clean' -> []."""
    if config_id == "clean":
        return []
    return [(n, int(s)) for n, s in (p.split("@") for p in config_id.split("+"))]


def config_id(steps, ordered=True):
    if not steps:
        return "clean"
    if ordered:
        steps = sorted(steps, key=lambda t: ORDER[t[0]])
    return "+".join(f"{n}@{s}" for n, s in steps)


def apply_config(img, cfg, page_id, rep=0, ordered=True):
    """Apply every step of a config. Steps run in physical order unless ordered=False
    (used only by the composition-order check, which applies them as written)."""
    steps = steps_of(cfg)
    if ordered:
        steps = sorted(steps, key=lambda t: ORDER[t[0]])
    for n, s in steps:
        img = apply_one(img, n, s, page_id, rep)
    return img


def experiment_configs():
    """clean + 24 singles + 84 same-severity pairs = 109 configurations."""
    rows = [{"config": "clean", "kind": "clean", "severity": 0, "a": "", "b": ""}]
    for s in SEVERITIES:
        for n in DEGRADATIONS:
            rows.append({"config": config_id([(n, s)]), "kind": "single", "severity": s, "a": n, "b": ""})
    for s in SEVERITIES:
        for a, b in itertools.combinations(DEGRADATIONS, 2):      # already in physical order
            rows.append({"config": config_id([(a, s), (b, s)]), "kind": "pair", "severity": s, "a": a, "b": b})
    return rows

"""Bootstrap intervals shared by the analysis scripts.

A statistic vector v holds, in order:
  v[0]                 the statistic on the full corpus
  v[1 : 1+n_boot]      the statistic on each bootstrap resample (p3_bootstrap.py)
  v[1+n_boot :]        the statistic with each page left out in turn (p3_jackknife.py), if built

Percentile intervals and bootstrap p-values use only the bootstrap rows. BCa intervals
(Efron 1987) also need the jackknife rows, for the acceleration constant; without them
they are NaN.
"""
from functools import lru_cache

import numpy as np
from scipy.stats import norm

from rdld import config


@lru_cache(maxsize=None)
def n_boot():
    return int(np.load(config.RESULTS / "p3" / "boot" / "resample_index.npy", mmap_mode="r").shape[0])


def load_rows(model, cfg):
    """Per-class AP rows for one detector x configuration: full, bootstrap, then jackknife if present."""
    boot = np.load(config.RESULTS / "p3" / "boot" / model / f"{cfg}.npy")
    jack = config.RESULTS / "p3" / "jack" / model / f"{cfg}.npy"
    return np.concatenate([boot, np.load(jack)]) if jack.exists() else boot


def parts(v):
    nb = n_boot()
    v = np.asarray(v, float)
    return float(v[0]), v[1:1 + nb], v[1 + nb:]


def percentile_ci(v, alpha=0.05):
    point, b, _ = parts(v)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def boot_p(v):
    """Two-sided bootstrap p-value for 'statistic = 0'."""
    _, b, _ = parts(v)
    n = len(b)
    return min(1.0, 2 * min((np.sum(b <= 0) + 1) / (n + 1), (np.sum(b >= 0) + 1) / (n + 1)))


def bca_ci(v, alpha=0.05):
    """BCa interval (lo, hi); NaN when the jackknife rows are missing."""
    point, b, j = parts(v)
    if len(j) == 0:
        return np.nan, np.nan
    if np.ptp(b) == 0:                          # degenerate (e.g. a pixel no-op control): no spread
        return point, point
    d = j.mean() - j
    den = 6 * np.sum(d ** 2) ** 1.5
    a = np.sum(d ** 3) / den if den > 0 else 0.0
    prop = (np.sum(b < point) + 0.5 * np.sum(b == point)) / len(b)
    prop = min(max(prop, 0.5 / len(b)), 1 - 0.5 / len(b))
    z0 = norm.ppf(prop)
    z = norm.ppf([alpha / 2, 1 - alpha / 2])
    q = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo, hi = np.percentile(b, 100 * q)
    return float(lo), float(hi)

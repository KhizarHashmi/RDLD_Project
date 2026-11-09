"""
p4_rq3_robustness.py -- P4 (Khizar, Workstream B): how far can the RQ3 claim be trusted?

p4_rq3_rankings.py reports that rankings hold up less well under pairs than under singles.
This script tests that claim against the obvious objections. All intervals are paired page
bootstraps on the shared resample index. Pixel no-op controls (defocus@1) are left out unless
a column says otherwise.

  1. Tau gap. Mean Kendall tau vs the clean order, singles minus pairs, with a 95% CI.
     Run on all detectors, on distinct architectures only (no scale siblings), on the
     original three, without RF-DETR, and on the first eight (before the added YOLO models).
  2. Damage control. Pairs damage pages more than singles, so a lower tau could only mean
     "more damage". Fits tau on is_pair plus mean retention in each resample, once linear in
     retention and once with retention bands; the is_pair coefficient is the pair effect at
     equal damage. Also reports the gap by severity and tau inside each band.
  3. Noise floor. Tau between seed 0 and seed 1 of the same severity-2 condition: how much
     the order moves when only the degradation randomness changes.
  4. Reversals with FDR. Significant reversals of a significant clean gap, uncorrected and
     after Benjamini-Hochberg across all tests.
  5. Benchmark agreement. Tau between the robustness order from singles and from pairs, and
     how often the most robust detector differs.

Outputs: RDLD_data/results/p4/rq3_robustness/

Run:
  python scripts/p4_rq3_robustness.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from itertools import combinations

import numpy as np
import pandas as pd

from rdld import config, detectors, intervals

BOOT = config.RESULTS / "p3" / "boot"
OUT = config.out_dir("p4", "rq3_robustness")
ALPHA = 0.05
SIBLINGS = ("heron101", "egretxl",      # same architectures as heron and egret, at another scale
            "yolo26l", "yolo11m", "yolo11l")   # the YOLO family counts once, as yolo (yolo26m)
BANDS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.01)]


def load(m, c):
    with np.errstate(all="ignore"):     # row 0 = full corpus, then bootstrap rows, then jackknife rows if built
        return 100 * np.nanmean(intervals.load_rows(m, c), axis=1)


def tau_vs(A, B):
    """Kendall tau between two orderings, per resample. A, B: (n_models, n_resamples)."""
    idx = np.array(list(combinations(range(A.shape[0]), 2)))
    sa = np.sign(A[idx[:, 0]] - A[idx[:, 1]])
    sb = np.sign(B[idx[:, 0]] - B[idx[:, 1]])
    return (sa * sb).mean(axis=0)


def ci(v):
    return intervals.percentile_ci(v)


def bca(v):
    lo, hi = intervals.bca_ci(v)
    return {"lo_bca": lo, "hi_bca": hi}


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(q, 1)
    return out


def main():
    configs = pd.read_csv(config.DEGRADED / "configs.csv")
    models = list(detectors.MODELS)
    X = {(m, c): load(m, c) for m in models for c in configs.config}
    is_control = (configs.severity == 1) & ((configs.a == "defocus") | (configs.b == "defocus"))
    test = configs[(configs.kind != "clean") & ~is_control].reset_index(drop=True)
    sets = {"all": models, "distinct_architectures": [m for m in models if m not in SIBLINGS],
            "original3": list(detectors.ORIGINAL),
            # RF-DETR was fine-tuned by us on one sixth of the training split: is any claim carried by it?
            "without_rfdetr": [m for m in models if m != "rfdetr"],
            # the eight detectors reported before the YOLO size x generation set was added
            "first8": [m for m in models if m not in ("yolo26l", "yolo11m", "yolo11l")]}

    # ---------------------------------------------------------------- 1 and 2
    gaps, damage, sev_rows, band_rows = [], [], [], []
    for name, ms in sets.items():
        clean = np.stack([X[(m, "clean")] for m in ms])
        T = np.stack([tau_vs(np.stack([X[(m, c)] for m in ms]), clean) for c in test.config])    # (configs, resamples)
        Rt = np.stack([np.mean([X[(m, c)] / X[(m, "clean")] for m in ms], axis=0) for c in test.config])
        single, pair = (test.kind == "single").to_numpy(), (test.kind == "pair").to_numpy()
        ts, tp = T[single].mean(axis=0), T[pair].mean(axis=0)
        g, lo, hi = ci(ts - tp)
        gaps.append({"detectors": name, "n_detectors": len(ms), "tau_single": ts[0], "tau_pair": tp[0],
                     "gap": g, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0), **bca(ts - tp)})

        coef = np.empty(T.shape[1])
        for b in range(T.shape[1]):
            D = np.c_[np.ones(len(test)), Rt[:, b], pair.astype(float)]
            coef[b] = np.linalg.lstsq(D, T[:, b], rcond=None)[0][2]
        p, lo, hi = ci(coef)
        damage.append({"detectors": name, "estimator": "linear in retention", "pair_effect_at_equal_damage": p,
                       "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0), **bca(coef)})
        # tau is not linear in retention, so also compare the two kinds within retention bands
        edges = [b[0] for b in BANDS[1:]]
        for b in range(T.shape[1]):
            band = np.digitize(Rt[:, b], edges)
            D = np.c_[np.eye(len(BANDS))[band], pair.astype(float)]
            coef[b] = np.linalg.lstsq(D, T[:, b], rcond=None)[0][-1]
        p, lo, hi = ci(coef)
        damage.append({"detectors": name, "estimator": "within retention bands", "pair_effect_at_equal_damage": p,
                       "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0), **bca(coef)})

        for s in (1, 2, 3):
            k = (test.severity == s).to_numpy()
            d = T[k & single].mean(axis=0) - T[k & pair].mean(axis=0)
            p, lo, hi = ci(d)
            sev_rows.append({"detectors": name, "severity": s, "tau_single": T[k & single, 0].mean(),
                             "tau_pair": T[k & pair, 0].mean(), "gap": p, "lo": lo, "hi": hi})
        for a, b in BANDS:
            k = (Rt[:, 0] >= a) & (Rt[:, 0] < b)
            band_rows.append({"detectors": name, "retention_from": a, "retention_to": b,
                              "n_single": int((k & single).sum()), "n_pair": int((k & pair).sum()),
                              "tau_single": float(T[k & single, 0].mean()) if (k & single).any() else np.nan,
                              "tau_pair": float(T[k & pair, 0].mean()) if (k & pair).any() else np.nan})
    GP = pd.DataFrame(gaps); GP.to_csv(OUT / "tau_gap.csv", index=False)
    DM = pd.DataFrame(damage); DM.to_csv(OUT / "pair_effect_at_equal_damage.csv", index=False)
    SV = pd.DataFrame(sev_rows); SV.to_csv(OUT / "tau_gap_by_severity.csv", index=False)
    BD = pd.DataFrame(band_rows); BD.to_csv(OUT / "tau_by_retention_band.csv", index=False)

    # ---------------------------------------------------------------- 3. noise floor
    rep_csv = config.DEGRADED / "configs_rep.csv"   # absent on the test-split repeat (no seed replicate)
    rep = pd.read_csv(rep_csv) if rep_csv.exists() else pd.DataFrame(columns=["base", "kind"])
    floor = []
    for name, ms in sets.items() if len(rep) else []:
        taus = {"single": [], "pair": []}
        for r in rep.itertuples():
            a = np.stack([load(m, r.base) for m in ms])
            b = np.stack([load(m, f"r1-{r.base}") for m in ms])
            taus[r.kind].append(tau_vs(a, b))
        for kind, v in taus.items():
            p, lo, hi = ci(np.mean(v, axis=0))
            floor.append({"detectors": name, "kind": kind, "n_conditions": len(v),
                          "tau_seed0_vs_seed1": p, "lo": lo, "hi": hi})
    FL = pd.DataFrame(floor); FL.to_csv(OUT / "seed_noise_floor.csv", index=False)

    # ---------------------------------------------------------------- 4. reversals with FDR
    rows = []
    for m1, m2 in combinations(models, 2):
        g0 = X[(m1, "clean")] - X[(m2, "clean")]
        _, l0, h0 = ci(g0)
        if not (l0 > 0 or h0 < 0):
            continue                     # no significant clean gap, so nothing to reverse
        for r in test.itertuples():
            v = X[(m1, r.config)] - X[(m2, r.config)]
            g, lo, hi = ci(v)
            p = intervals.boot_p(v)
            rows.append({"m1": m1, "m2": m2, "config": r.config, "kind": r.kind, "severity": r.severity,
                         "clean_gap": g0[0], "gap": g, "lo": lo, "hi": hi, "p": p,
                         "opposite_sign": bool(np.sign(g) == -np.sign(g0[0])), "ci_excludes_zero": bool(lo > 0 or hi < 0)})
    RV = pd.DataFrame(rows)
    RV["q"] = bh(RV.p)
    RV["reversal_uncorrected"] = RV.opposite_sign & RV.ci_excludes_zero
    RV["reversal_fdr"] = RV.opposite_sign & (RV.q < ALPHA)
    RV.to_csv(OUT / "reversals_fdr.csv", index=False)
    RS = RV.groupby("kind").agg(tests=("p", "size"), reversal_uncorrected=("reversal_uncorrected", "sum"),
                                reversal_fdr=("reversal_fdr", "sum")).reset_index()
    RS["fdr_rate_pct"] = 100 * RS.reversal_fdr / RS.tests
    RS.to_csv(OUT / "reversal_summary.csv", index=False)

    # ---------------------------------------------------------------- 5. benchmark agreement
    agree = []
    for name, ms in sets.items():
        ret = {k: np.stack([np.mean([X[(m, c)] / X[(m, "clean")] for c in test[test.kind == k].config], axis=0)
                            for m in ms]) for k in ("single", "pair")}
        p, lo, hi = ci(tau_vs(ret["single"], ret["pair"]))
        top_s, top_p = ret["single"].argmax(axis=0), ret["pair"].argmax(axis=0)
        nb = intervals.n_boot()
        agree.append({"detectors": name, "tau_single_vs_pair_benchmark": p, "lo": lo, "hi": hi,
                      "most_robust_single": ms[top_s[0]], "most_robust_pair": ms[top_p[0]],
                      "boot_share_top_differs": float(np.mean(top_s[1:1 + nb] != top_p[1:1 + nb]))})
    AG = pd.DataFrame(agree); AG.to_csv(OUT / "benchmark_agreement.csv", index=False)

    # ---------------------------------------------------------------- report
    for title, T in (("tau gap, singles minus pairs", GP), ("pair effect on tau at equal damage (negative = pairs scramble more)", DM),
                     ("gap by severity", SV), ("tau by retention band", BD), ("seed noise floor", FL),
                     ("reversals of a significant clean gap", RS), ("single vs pair benchmark", AG)):
        print(f"\n{title}:")
        print(T.round(3).to_string(index=False))
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

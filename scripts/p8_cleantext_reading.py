"""
p8_cleantext_reading.py -- P8 (Khizar, Workstream B): does the RQ2 answer depend on OCR noise?

The main reading (p5_ocr_downstream.py, TEXT=degraded) reads the words that Tesseract finds on the
degraded image. With TEXT=clean the same script reads the words found on the clean page, through the
regions the detector predicted on the degraded page. Any change in reading quality then comes from
the layout alone. This script puts the two readings side by side:

  1. mean WER, text recall and text precision per condition kind, degraded vs clean text
  2. the within-condition Spearman correlations used in RQ2, for both readings
  3. how often the detector with the best mAP also has the lowest WER, and how well the detector
     order by WER agrees between the two readings (Kendall tau per condition)

Inputs:  results/p5/downstream, results/p5/downstream_cleantext, p2/evaluation/metrics_long.csv,
         p4/structural/structural_rates.csv
Outputs: <data root>/results/p8/cleantext_reading/{summary_by_kind,within_condition,agreement}.csv

Run:
  python scripts/p8_cleantext_reading.py
  RDLD_DATA=../RDLD_data_test python scripts/p8_cleantext_reading.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import kendalltau, spearmanr

from rdld import config

R = config.RESULTS
OUT = config.out_dir("p8", "cleantext_reading")
TARGETS = ["WER", "text_recall", "text_precision"]


def joined(sub):
    DS = pd.read_csv(R / "p5" / sub / "downstream_configs.csv")
    M = pd.read_csv(R / "p2" / "evaluation" / "metrics_long.csv")
    S = pd.read_csv(R / "p4" / "structural" / "structural_rates.csv")
    J = (DS[["model", "config", "kind", "severity"] + TARGETS]
         .merge(M[["model", "config", "mAP"]], on=["model", "config"])
         .merge(S[["model", "config", "pred_duplicate"]], on=["model", "config"]))
    J["mAP"] *= 100
    return J


def main():
    J = {"degraded": joined("downstream"), "clean text": joined("downstream_cleantext")}

    summ = []
    for name, d in J.items():
        for kind, g in d.groupby("kind"):
            summ.append({"reading": name, "kind": kind, "n": len(g), **{t: round(g[t].mean(), 4) for t in TARGETS}})
    SU = pd.DataFrame(summ)
    SU.to_csv(OUT / "summary_by_kind.csv", index=False)

    within = []
    for name, d in J.items():
        D = d.copy()
        for c in ["mAP", "pred_duplicate"] + TARGETS:
            D[c] = D[c] - D.groupby("config")[c].transform("mean")
        for a in ["mAP", "pred_duplicate"]:
            for t in TARGETS:
                within.append({"reading": name, "layout_metric": a, "downstream": t, "n": len(D),
                               "spearman_within_condition": round(float(spearmanr(D[a], D[t]).statistic), 3)})
    WC = pd.DataFrame(within)
    WC.to_csv(OUT / "within_condition.csv", index=False)

    agree = []
    a, b = J["degraded"], J["clean text"]
    for cfg in a.config.unique():
        x = a[a.config == cfg].set_index("model").sort_index()
        y = b[b.config == cfg].set_index("model").loc[x.index]
        agree.append({"config": cfg, "kind": x.kind.iloc[0],
                      "best_map_is_best_wer_degraded": x.mAP.idxmax() == x.WER.idxmin(),
                      "best_map_is_best_wer_cleantext": y.mAP.idxmax() == y.WER.idxmin(),
                      "tau_wer_order_degraded_vs_cleantext": round(float(kendalltau(x.WER, y.WER).statistic), 3)})
    AG = pd.DataFrame(agree)
    AG.to_csv(OUT / "agreement.csv", index=False)

    print(SU.to_string(index=False))
    print()
    print(WC.pivot_table(index=["layout_metric", "downstream"], columns="reading",
                         values="spearman_within_condition").to_string())
    print(f"\nbest mAP is best WER: degraded {AG.best_map_is_best_wer_degraded.sum()}/{len(AG)}, "
          f"clean text {AG.best_map_is_best_wer_cleantext.sum()}/{len(AG)}")
    print(f"Kendall tau of the WER order, degraded vs clean text: median {AG.tau_wer_order_degraded_vs_cleantext.median():.2f}, "
          f"min {AG.tau_wer_order_degraded_vs_cleantext.min():.2f}")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()

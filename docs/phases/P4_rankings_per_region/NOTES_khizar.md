# P4 - Rankings & Stability - Khizar (Workstream B)

**Dates:** 2026-11-04

## 1. Goal
(1) RQ3: does the clean ranking of the three detectors survive degradation, and does a compound-
degradation benchmark rank robustness differently from a single-degradation one? (2) Do the RQ1
interaction estimates depend on the particular random draw of the degradations?

## 2. What was done

| Step | Command | Output |
|---|---|---|
| Ranking analysis | `python scripts/p4_rq3_rankings.py` | `RDLD_data/results/p4/rq3/` |
| Seed stability (needs `r1-*` predictions + bootstrap tables) | `python scripts/p4_seed_stability.py` | `RDLD_data/results/p4/seeds/` |
| mRD (RoDLA Eq. 4, IQA-only mPE) | `python scripts/p2_evaluate.py --mrd-only` | `RDLD_data/results/p2/evaluation/{mrd.csv, rd_per_degradation.csv}` |

**RQ3 tests**, all on the shared paired bootstrap tables: per configuration, the pairwise mAP gap
between detectors with a 95% CI; a *significant reversal* = the clean gap is significant and the
degraded gap is significant in the opposite direction. Kendall τ between the clean ordering and each
degraded ordering. Benchmark-level robustness = mean retention (degraded mAP / clean mAP) over the
singles benchmark vs the pairs benchmark, with CIs on the retention gap between models. Spearman
correlation of per-configuration drops between models.

**Seed test.** Workstream A regenerated every severity-2 single and pair with a second seed (`r1-*`,
36 configurations). I_mult is recomputed on the replicate and compared with seed 0.

## 3. Results (evidence)

### 3.1 Orderings across the 108 degraded configurations (`p4/rq3/rank_agreement.csv`, `p4_rq3_rankings.log`)
Clean: **YOLO (68.82) > heron (64.23) > egret (61.52)**.

| Ordering | singles (24) | pairs (84) |
|---|---|---|
| YOLO > heron > egret (= clean) | 18 | 46 |
| heron > egret > YOLO | 5 | 31 |
| heron > YOLO > egret | 1 | 7 |

- **Mean Kendall τ vs clean: singles 0.694, pairs 0.452.**
- **Significant reversals of a significant clean gap** (`p4/rq3/pairwise_gaps.csv`): all of them are
  YOLO vs egret - **5 of 24 singles, 27 of 84 pairs**. On those pairs egret, the weakest clean detector,
  significantly beats YOLO, the strongest.

### 3.2 Robustness ranking: single vs compound benchmark (`p4/rq3/benchmark_rankings.csv`, `benchmark_retention_gaps.csv`)

| Detector | mean retention, singles | mean retention, pairs | rank (both) |
|---|---|---|---|
| heron | 83.12% | 67.29% | 1 |
| YOLO | 79.41% | 61.96% | 2 |
| egret | 79.35% | 60.75% | 3 |

| Gap (retention points) | singles [95% CI] | pairs [95% CI] |
|---|---|---|
| YOLO − heron | −3.71 [−5.67, −2.13] | −5.33 [−7.89, −2.98] |
| YOLO − egret | 0.07 [−2.70, 1.96] | 1.21 [−1.95, 3.73] |
| heron − egret | 3.77 [1.56, 5.42] | **6.54** [3.87, 8.62] |

The robustness *ranking* is the same on both benchmarks, but the compound benchmark **widens** the gaps:
heron's advantage over egret almost doubles (3.8 → 6.5 points).

### 3.3 Robustness metric changes the ranking (`p2/evaluation/mrd.csv`)

| Detector | mRD (lower = more robust) | rank by mRD | rank by retention |
|---|---|---|---|
| YOLO | **7.8** | 1 | 2 |
| heron | 8.9 | 2 | 1 |
| egret | 9.9 | 3 | 3 |

RoDLA's mRD has (100 − mAP) in the numerator, so it scores *absolute* error, which includes clean error;
retention scores *relative* loss. YOLO's higher clean mAP makes it "most robust" by mRD while it loses
more of its own accuracy than heron does.

### 3.4 Do the detectors fail on the same configurations? (`p4/rq3/drop_correlation_between_models.csv`)
Spearman ρ of per-configuration mAP drops: heron–egret **0.984**, YOLO–heron 0.905, YOLO–egret 0.904.
The two transformers degrade almost identically; the CNN has its own failure profile (texture,
speckle - see P2), which is where the reversals come from. Figure: `p4/rq3/fig_model_vs_model.png`.

### 3.5 Seed stability of RQ1 (`p4/seeds/seed_summary.csv`, `seed_comparison.csv`; 28 severity-2 pairs)

| Detector | ρ(I seed 0, I seed 1) | mean |ΔI| (pts) | mean CI half-width (pts) | seed / page noise | same call | opposite significant calls |
|---|---|---|---|---|---|---|---|
| YOLO | 0.968 | 0.69 | 2.51 | 0.28 | 82% | 0 |
| heron | 0.901 | 0.80 | 2.24 | 0.36 | 89% | 0 |
| egret | 0.944 | 1.03 | 2.48 | 0.41 | 82% | 0 |

Seed-to-seed variation is about a third of page-sampling variation. **No pair changes from super- to
sub-additive (or back)**; every disagreement is significant vs n.s. at the boundary.

## 4. Decisions and why
1. **Reversal = significant in both conditions, opposite signs.** A ranking "change" between two
   non-significant gaps is noise; the strict definition counts only defensible reversals.
2. **Two robustness summaries (retention and mRD) are both reported**, because they rank differently
   and the difference is itself a finding about benchmark design.
3. **Second seed at severity 2 only.** Enough to bound seed noise relative to page noise; the main
   experiment's single seed is justified by the 0.28–0.41 ratio.

## 5. Problems found
- **Kendall τ with 3 items is coarse** (values −1, −1/3, 1/3, 1); the means are interpretable but
  per-configuration τ is not. The pairwise-gap test is the primary RQ3 evidence.
- **Seed agreement on calls is 82–89%, not 100%**, because several pairs sit near q = 0.05. The pooled
  two-seed estimate is in `seed_comparison.csv` (`I_pooled`, `call_pooled`) if a reviewer asks.

## 6. Main takeaway
RQ3 answer: **rankings are not preserved under compound degradation** - agreement with the clean
order falls from τ = 0.69 (singles) to 0.45 (pairs), with 27 significant reversals on pairs vs 5 on
singles. The robustness ranking itself is stable, but compound testing doubles the measured gap between
the two transformers. RQ1 conclusions are seed-robust (seed noise ≈ ⅓ of page noise, zero sign flips).

## 7. Hand-off
- P5: the ranking question at the downstream level - does the mAP winner per condition also win on WER?
- P6: §3.3 (metric choice changes the robustness ranking) belongs in the discussion of the paper.

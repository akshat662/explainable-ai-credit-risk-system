# Threshold Analysis Report (Phase 8)

## 1. Objective

Convert the frozen model's default probability into an approve/reject decision under an explicit, documented expected-cost model. This is a business-decisioning exercise, not a modeling exercise: no ROC-AUC/PR-AUC optimization happens in this phase.

## 2. Frozen model / probability source

**Raw, uncalibrated XGBoost** (Phase 7.8 decision, unchanged here). Predictions used for threshold selection are dev-only out-of-fold (OOF) scores from the frozen `train_xgboost.MODEL_PARAMS` configuration (246008 dev rows). `holdout.parquet` is not read anywhere in this phase.

## 3. Economic assumptions

- LGD (loss given default) = 0.6
- Margin (net profit if repaid) = 0.08
- Both expressed as a fraction of loan exposure; the same unit throughout.

## 4. Exact expected-cost formulation

**A naive formula was considered and rejected.** The initial brief suggested `expected_cost_approve = p*LGD - margin`, giving a break-even threshold of `margin/LGD = 0.13333`. This does **not** match the agreed theoretical threshold (0.11765), and the discrepancy is not a rounding issue -- it reflects a real economic assumption error: the naive formula treats `margin` as earned *unconditionally*, regardless of whether the loan is repaid.

**Corrected formulation.** Margin is only earned if the loan is repaid (probability `1-p`); LGD is only lost if the applicant defaults (probability `p`):

```
expected_cost_approve(p) = p * LGD - (1 - p) * margin
expected_cost_reject     = 0

approve if expected_cost_approve(p) < 0
        <=> p * LGD < (1 - p) * margin
        <=> p * (LGD + margin) < margin
        <=> p < margin / (margin + LGD)
```

This reproduces the agreed theoretical threshold exactly: `0.08/(0.08+0.6) = 0.11765`. **This corrected formula is what `evaluate_threshold()` implements.** The naive formula is retained in code (`naive_expected_cost_approve`, `naive_threshold`) for documentation and comparison only -- it is never used for the actual decision rule.

## 5. Threshold derivation

Theoretical break-even threshold: `margin/(margin+LGD)` = `0.08/0.68` = **0.11765**.

## 6. Threshold sweep methodology

Grid: `[0.01, 0.5]` step `0.001` (491 thresholds). For each threshold, applicants with `p_default >= threshold` are rejected; all others are approved. **"Positive" is defined as default/risky applicant (TARGET == 1)** throughout -- a REJECT decision corresponds to a positive prediction. Expected cost at each threshold is computed from *realized* dev outcomes (actual `TARGET` labels), not from re-applying the probability formula -- this is a historical backtest on dev, using dev's known labels, never holdout's.

## 7. Dev results

- Dev default prevalence: **8.0729%**
- "Approve everyone" mean expected cost: **-0.025104** (a net profit per applicant, since the population default rate is below the break-even threshold)
- "Reject everyone" mean expected cost: **0.000000** (by definition)
- Cost at theoretical break-even threshold (0.11765): **-0.038229**, approval rate 80.64%
- **Selected (minimum-cost, tie-broken) threshold: 0.110**, mean expected cost **-0.038287**, approval rate **78.86%**, rejection rate **21.14%**
- At the selected threshold: 10889.0 of 19860 dev defaults correctly rejected (54.83% capture rate); default rate among approved applicants: 4.6245% (vs. 8.0729% unconditional)
- Confusion matrix at selected threshold: TP=10889.0, FP=41129.0, TN=185019.0, FN=8971.0

**Tie-breaking rule applied:** among all grid thresholds within 1.0 standard error(s) of the minimum mean cost (26 candidate thresholds, tolerance 2.62e-04 -- a statistically grounded bar, not an arbitrary percentage; see DECISIONS.md), the one closest to a round hundredth (e.g. 0.10, 0.11, 0.12) was selected, for interpretability, since the cost difference among them is negligible.

See `reports/cost_vs_threshold.png` for the full cost curve.

## 8. Theoretical-vs-empirical threshold comparison

Theoretical: **0.11765**. Empirical (dev-minimum-cost, pre-tie-break): **0.114**. These are expected to be close but not necessarily identical: the theoretical threshold assumes the model's probability is exactly calibrated at that point, while the empirical minimum reflects whatever calibration the raw (uncalibrated) XGBoost score actually has there. Rough agreement is a consistency check on the model and the formula, not a claim that raw XGBoost is perfectly calibrated (it is close, per Phase 7's reliability diagram, but not exact).

## 9. Sensitivity analysis

LGD grid: [0.4, 0.5, 0.6, 0.7, 0.8]. Margin grid: [0.04, 0.06, 0.08, 0.1, 0.12]. For each combination, both the theoretical and empirical (dev) cost-minimizing threshold are computed. Full table: `reports/sensitivity_analysis.csv`; heatmap: `reports/sensitivity_heatmap.png`.

| LGD | Margin | Theoretical threshold | Empirical threshold | Approval rate |
|---|---|---|---|---|
| 0.40 | 0.04 | 0.09091 | 0.081 | 70.0% |
| 0.40 | 0.06 | 0.13043 | 0.114 | 79.8% |
| 0.40 | 0.08 | 0.16667 | 0.175 | 89.1% |
| 0.40 | 0.10 | 0.20000 | 0.205 | 91.7% |
| 0.40 | 0.12 | 0.23077 | 0.251 | 94.4% |
| 0.50 | 0.04 | 0.07407 | 0.066 | 63.3% |
| 0.50 | 0.06 | 0.10714 | 0.097 | 75.4% |
| 0.50 | 0.08 | 0.13793 | 0.117 | 80.5% |
| 0.50 | 0.10 | 0.16667 | 0.175 | 89.1% |
| 0.50 | 0.12 | 0.19355 | 0.193 | 90.8% |
| 0.60 | 0.04 | 0.06250 | 0.055 | 57.1% |
| 0.60 | 0.06 | 0.09091 | 0.081 | 70.0% |
| 0.60 | 0.08 | 0.11765 | 0.114 | 79.8% |
| 0.60 | 0.10 | 0.14286 | 0.137 | 84.1% |
| 0.60 | 0.12 | 0.16667 | 0.175 | 89.1% |
| 0.70 | 0.04 | 0.05405 | 0.047 | 51.4% |
| 0.70 | 0.06 | 0.07895 | 0.071 | 65.8% |
| 0.70 | 0.08 | 0.10256 | 0.095 | 74.8% |
| 0.70 | 0.10 | 0.12500 | 0.114 | 79.8% |
| 0.70 | 0.12 | 0.14634 | 0.140 | 84.6% |
| 0.80 | 0.04 | 0.04762 | 0.043 | 48.3% |
| 0.80 | 0.06 | 0.06977 | 0.066 | 63.3% |
| 0.80 | 0.08 | 0.09091 | 0.081 | 70.0% |
| 0.80 | 0.10 | 0.11111 | 0.108 | 78.4% |
| 0.80 | 0.12 | 0.13043 | 0.114 | 79.8% |

As LGD rises or margin falls, the break-even threshold drops (the lender must be more conservative — reject more applicants — since each default is costlier relative to the reward from a good loan). As margin rises or LGD falls, the threshold rises (more applicants are worth the risk).

## 10. Business interpretation

At the agreed economics (LGD=0.6, margin=0.08), the mathematically cost-minimizing policy approves 78.9% of dev applicants and rejects 21.1%. This is a *substantial* rejection volume relative to the raw default rate (8.07%) -- the cost-minimizing lender rejects far more applicants than actually default, because the cost of a missed default (LGD) is large relative to the reward from a good loan (margin), so the model errs toward caution near the threshold. **This is a real trade-off, not a modeling artifact**: a purely cost-minimizing threshold can imply a much lower approval volume than a real lender focused on growth or market share would accept. In practice, lenders operate under approval-volume or growth constraints that a pure expected-cost minimization does not represent. The expected-cost threshold derived here is a **business policy choice given the stated economics**, not a universally "correct" number independent of business context — a different risk appetite, funding cost, or growth target would justify a different threshold along the sensitivity curve in Section 9.

## 11. Limitations

- LGD and margin are treated as constants across all applicants; in reality both likely vary by loan size, term, and applicant segment.
- The expected-cost model ignores time value of money, funding cost, operational cost of processing an application, and any regulatory constraints on rejection rates or fair-lending requirements.
- The empirical sweep uses dev's realized outcomes as a backtest; it is not a forward-looking guarantee, and dev itself is one realization of the applicant population (same caveat as noted for holdout in `reports/final_model_card.md`).
- Raw XGBoost is close to, but not exactly, calibrated (Phase 7 finding) — the threshold sweep uses realized outcomes specifically so that residual miscalibration does not bias the *selected* threshold, but the theoretical break-even value in isolation does assume calibration at that point.
- **Reject inference**: this entire analysis (defaults captured, default rate among approved/rejected, expected cost) is computed against `dev.parquet`'s labels, which are only observed for applicants Home Credit's incumbent process actually approved. The threshold's expected-cost behavior is validated on the historically-approved population, not against how it would behave if applied to the incumbent process's historically-rejected applicants. See `reports/final_model_card.md`'s "Known limitations" for the full explanation.
- This analysis assumes **unit exposure** (every applicant weighted equally, independent of loan size). See `reports/exposure_weighted_analysis.md` for a secondary, dev-only sensitivity analysis weighting outcomes by `AMT_CREDIT` — it does not change the frozen 0.110 threshold.

## 12. Decision locked for final holdout evaluation

**Selected threshold: 0.110** (reject if `p_default >= 0.110`), locked from dev-only analysis above. Per Phase 8 methodology rules, holdout.parquet was not read at any point in this phase.

**Phase 8 threshold is now frozen; final holdout evaluation is the next step.**


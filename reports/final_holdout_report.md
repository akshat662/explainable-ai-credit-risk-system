# Final Holdout Evaluation Report (Phase 8.5)

## 1. Executive summary

The frozen model + frozen 0.110 decision threshold were evaluated exactly once on the untouched holdout set (61,503 applicants). Holdout ROC-AUC 0.7713, PR-AUC 0.2650, Brier score 0.066840. At the frozen threshold, 78.66% of applicants are approved with a mean expected cost of -0.038794 per applicant -- better than both the approve-all (-0.025105) and reject-all (0.000000) baselines. **This is a report of how a pre-specified policy generalized, not a search for a better one.**

## 2. Frozen model configuration

XGBoost, trained once on all of `dev.parquet` (`n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr", random_state=42`), imported unchanged from `train_xgboost.MODEL_PARAMS`. No cross-validation, no hyperparameter tuning, no calibration in this phase.

## 3. Frozen feature state

147 model features (120 raw `application_train` + 26 engineered bureau/previous-application/ratio features + 1 `days_employed_anomaly`), from the current, unmodified `src/features.py` / `src/data_quality.py`. Negative `AMT_CREDIT_SUM_DEBT` clipped to zero with `bureau_had_negative_debt` retained (Phase 7.5, unchanged). Dev/holdout split via the `ORDER BY SK_ID_CURR`-stabilized, deterministic 80/20 stratified split (Phase 7.5, unchanged).

## 4. Frozen probability choice

Raw, uncalibrated XGBoost probability (Phase 7.8 decision, statistically validated by bootstrap, unchanged). No isotonic calibration, no Platt scaling applied anywhere in this evaluation.

## 5. Frozen business threshold

**Threshold = 0.110** (reject if `p_default >= 0.110`), selected on dev-only data in Phase 8 under LGD=0.6, margin=0.08. Not re-derived, re-swept, or adjusted in this phase.

## 6. Why holdout evaluation happens only now

This is the first phase in the project permitted to read `holdout.parquet`. Model configuration (Phase 6), probability source (Phase 7.8, bootstrap-validated), and business threshold (Phase 8, dev-only sweep) were all frozen *before* this file's first `read_parquet('holdout.parquet')` call. Reading holdout any earlier would have let it silently influence a decision it exists to test honestly.

**DEV** was used for: XGBoost model comparison, calibration analysis, bootstrap significance testing, threshold sweep and selection, sensitivity analysis. **HOLDOUT** is used for: final evaluation only -- nothing computed on holdout in this phase feeds back into any prior decision.

## 7. Holdout predictive performance

- Holdout applicants: **61,503**
- Default prevalence: **8.0728%**
- ROC-AUC: **0.7713**
- PR-AUC: **0.2650**
- Brier score: **0.066840**
- Constant base-rate predictor Brier score (p = dev prevalence 8.0729%, frozen, not re-derived from holdout): **0.074211**

**Do not compare these directly to dev CV metrics as though they should match.** Dev CV metrics (ROC-AUC 0.7634 ± 0.0044, PR-AUC 0.2499 ± 0.0087) are averaged over 5 folds, each trained on 80% of dev and validated on a held-out 20% *of dev*. Holdout metrics come from a single model trained on 100% of dev, scored once on an entirely separate 61,503-applicant sample. Different training data, different evaluation data, and no folding/averaging on the holdout side -- some difference is expected by construction, not a sign of a problem either way.

## 8. Holdout business-decision performance

At threshold 0.110 (**"positive" = default/risky applicant**, TARGET == 1; a REJECT decision is a positive prediction):

- Approved: **48,380** (78.66%)
- Rejected: **13,123** (21.34%)
- Default rate among approved: **4.5122%**
- Default rate among rejected: **21.1994%**
- Defaults captured by rejection: **2,782** (56.03% of all holdout defaults)
- Confusion matrix: TP=2,782, FP=10,341, TN=46,197, FN=2,183
- Sanity check: TP+FP+TN+FN = 61,503 = holdout row count (61,503)

See `reports/final_holdout_confusion_matrix.png`.

## 9. Cost comparison

Same economic formulation locked in Phase 8 (`expected_cost_approve(p) = p*LGD - (1-p)*margin`), evaluated against *realized* holdout outcomes:

- Approve everyone: **-0.025105** per applicant
- Reject everyone: **0.000000** per applicant
- Frozen 0.110 threshold: **-0.038794** per applicant

See `reports/final_holdout_decision_summary.png`.

## 10. Dev vs. holdout comparison

| Metric | Dev (CV) | Holdout |
|---|---|---|
| ROC-AUC | 0.7634 ± 0.0044 | 0.7713 |
| PR-AUC | 0.2499 ± 0.0087 | 0.2650 |
| Default prevalence | 8.0729% | 8.0728% |

(Dev figures above are the Phase 7.6 authoritative CV benchmarks, restated for reference -- not recomputed in this phase.)

## 11. Generalization interpretation

The threshold of 0.110 was frozen using dev data before holdout evaluation. These holdout results measure how that pre-specified policy generalizes -- they are not evidence that 0.110 is optimal on holdout, and no alternative holdout threshold was computed or considered. Holdout's frozen-threshold cost is *better* than dev's cost at the same threshold, which is a reportable observation, not a reason to revisit the threshold.

## 12. Limitations

- This is a single holdout evaluation, not a repeated or bootstrapped one -- the reported numbers are one realization, same caveat as noted throughout this project for dev-based estimates.
- LGD and margin remain fixed, illustrative constants, not measured quantities; see `reports/threshold_analysis.md` Section 11 for the full list of economic-model limitations, which apply identically here.
- The constant-baseline Brier score uses dev's prevalence deliberately, not holdout's own realized prevalence, to avoid deriving any new quantity from holdout even for a contextual baseline.

## 13. Final frozen-system status

Model, probability source, feature pipeline, and business threshold all remain exactly as frozen entering this phase. No value in this report changes any of them.

**Phase 8.5 complete: the frozen 0.110 decision policy has been evaluated once on the untouched holdout; no further threshold optimization was performed.**


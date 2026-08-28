# Holdout Calibration Report

Honest, holdout-validated evaluation of the three calibration methods against `holdout.parquet` (61503 applicants, read only for scoring — never used to train XGBoost, generate OOF predictions, or fit Platt/isotonic). Development figures (246008 applicants) are from `reports/calibration_results.json`, regenerated in this phase against the corrected features and split (see DECISIONS.md). **This report's original Section 3 conclusion was superseded by a Phase 7.8 bootstrap test — see the note there and `reports/final_model_card.md` for the current final decision.**

## 1. Dev (OOF) vs. holdout (final evaluation)

| Method | Brier (dev OOF) | Brier (holdout) | ROC-AUC (dev OOF) | ROC-AUC (holdout) | PR-AUC (dev OOF) | PR-AUC (holdout) |
|---|---|---|---|---|---|---|
| raw_xgb | 0.06762 | 0.06684 | 0.7634 | 0.7713 | 0.2493 | 0.2650 |
| platt | 0.06860 | 0.06782 | 0.7634 | 0.7713 | 0.2493 | 0.2650 |
| isotonic | 0.06742 | 0.06683 | 0.7639 | 0.7713 | 0.2453 | 0.2585 |

## 2. Interpretation

Best method on holdout by Brier score: **isotonic** (0.06683).

ROC-AUC and PR-AUC are expected to move together across raw/Platt/isotonic on holdout, same as on dev OOF (Platt is a strictly monotonic transform of the same score; isotonic's step function only reorders through ties) -- Brier score remains the metric that actually differentiates calibration quality.

Dev-OOF and holdout numbers are close but not identical, as expected: they're different applicants, and dev OOF carries the small optimism noted in Phase 7 (calibrators evaluated on the same OOF sample used to fit them). The holdout numbers are the ones that should be trusted for reporting real-world performance, since holdout was never seen during training, OOF generation, or calibrator fitting.

**The isotonic-vs-raw gap narrows substantially on holdout.** On dev OOF, isotonic looked clearly ahead of raw (0.06742 vs. 0.06762, a 0.00020 gap). On holdout, the gap nearly vanishes (0.066826 vs. 0.066840, a 0.000014 gap) — consistent with the dev-OOF comparison carrying some optimism from calibrators being evaluated on the same sample used to fit them (flagged as a known limitation in Phase 7 and again in `src/evaluate_holdout_calibration.py`). The one finding that *is* robust across both evaluations: **Platt scaling underperforms raw XGBoost** in both dev OOF and holdout, by a similar margin each time (~0.001 Brier) — this is not a fitting artifact, it replicates on genuinely unseen data.

## 3. Final probability source

> **Superseded (Phase 7.8):** at the time this report was written, isotonic was recommended below as "a safe, non-inferior choice with a real, if small, edge." A 1,000-iteration bootstrap (`random_state=42`, `reports/calibration_bootstrap_results.json`) subsequently tested that edge directly: 95% CI for Brier(raw) − Brier(isotonic) is **[-6.81e-05, +1.03e-04]**, which includes 0. The "small edge" was not statistically distinguishable from noise. **Raw, uncalibrated XGBoost is the final production probability source** — see `reports/final_model_card.md` for the authoritative statement. The analysis and numbers below are preserved as the historical record of how the original (superseded) recommendation was reached; they are not being retracted or rewritten, only corrected going forward.

*Original conclusion, preserved for the record:* isotonic was recommended as the calibrated-probability source for the decision engine — it was never worse than raw on either evaluation and had holdout's best (if narrowly) Brier score. This should be read as "isotonic is a safe, non-inferior choice with a real, if small, edge," not as "isotonic decisively fixes a calibration problem" — the raw score turned out to already be close to calibrated. Platt scaling should not be used: it is the one method with a holdout-confirmed regression relative to the raw score. (This last point — Platt's exclusion — is unaffected by the Phase 7.8 supersession above and remains current.)


# Explainable AI Credit Risk Decision System

An end-to-end machine learning system for credit risk assessment with:

- SQL-based feature engineering
- Leakage auditing
- Probability calibration
- Cost-sensitive lending decisions
- SHAP explainability
- Streamlit deployment

## Project Status

Core pipeline complete and frozen (see `reports/final_model_card.md` for
the authoritative, versioned summary). Progression so far:

raw data → DuckDB feature engineering → data quality → leakage audit →
deterministic dev/holdout split → baseline comparison → calibration
investigation → statistical validation → **raw XGBoost selected** →
Phase 8 (business decision engine) next.

- ✅ DuckDB-based data profiling and SQL feature engineering
- ✅ Data quality processing (sentinel handling, negative-bureau-debt clipping)
- ✅ Leakage audit and temporal validity analysis
- ✅ Deterministic dev/holdout split
- ✅ Logistic regression baseline vs. XGBoost candidate comparison
- ✅ Calibration investigation (Platt, isotonic) with holdout and bootstrap
  statistical validation
- 🚧 Phase 8 — business decision thresholds, SHAP explanations, Streamlit
  deployment — not yet started

Engineered 30+ applicant-level features from 2 relational tables,
producing a 147-feature modelling matrix. Current benchmark (development
set, stratified 5-fold CV): XGBoost ROC-AUC 0.7634 ± 0.0044, PR-AUC 0.2499
± 0.0087, versus a logistic regression baseline of ROC-AUC 0.7543 ±
0.0039, PR-AUC 0.2301 ± 0.0076.

**Production probability source: raw, uncalibrated XGBoost.** Calibration
(Platt scaling, isotonic regression) was investigated, not assumed: Platt
consistently underperformed; isotonic's small holdout Brier-score edge
(0.066840 raw vs. 0.066826 isotonic) was tested with a 1,000-iteration
bootstrap and found not statistically distinguishable from zero (95% CI
[-6.81e-05, +1.03e-04]). The simpler, uncalibrated model was selected on
that basis — see `DECISIONS.md` for the full reasoning and
`reports/final_model_card.md` for the complete evidence trail.
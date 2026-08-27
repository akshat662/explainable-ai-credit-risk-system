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
the authoritative, versioned summary):

- ✅ DuckDB-based data profiling and SQL feature engineering
- ✅ Data quality processing and a deterministic dev/holdout split
- ✅ Leakage audit and temporal validity analysis
- ✅ Logistic regression baseline and XGBoost candidate model
- ✅ Probability calibration, bootstrap-validated against holdout
- 🚧 SHAP explanations, business decision thresholds, and Streamlit
  deployment — not yet started

Engineered 30+ applicant-level features from relational banking tables
using DuckDB SQL, producing a 147-feature modeling matrix. Final
benchmark (development set, stratified 5-fold CV): XGBoost ROC-AUC 0.7634
± 0.0044, PR-AUC 0.2499 ± 0.0087, versus a logistic regression baseline of
ROC-AUC 0.7543 ± 0.0039, PR-AUC 0.2301 ± 0.0076. The production
probability source is the raw XGBoost score (uncalibrated) — a bootstrap
significance test found no statistically meaningful improvement from
isotonic calibration; see `DECISIONS.md` for the full reasoning.
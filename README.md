# Explainable AI Credit Risk Decision System

An end-to-end machine learning system for credit risk assessment with:

- SQL-based feature engineering
- Leakage auditing
- Probability calibration
- Cost-sensitive lending decisions
- SHAP explainability
- Streamlit deployment

This is a portfolio project demonstrating an explainable, business-aware
credit-risk decision pipeline — not a production banking system.

## Project status

Core pipeline complete and frozen; a deployment demo layer is built on top
of it (see `reports/final_model_card.md` for the authoritative, versioned
pipeline summary, and `reports/phase10_deployment_report.md` for the
deployment layer). Progression so far:

raw data → DuckDB feature engineering → data quality → leakage audit →
deterministic dev/holdout split → baseline comparison → calibration
investigation → statistical validation → **raw XGBoost selected** →
cost-sensitive business threshold → untouched-holdout evaluation → SHAP
explainability → **Streamlit deployment demo**.

- ✅ DuckDB-based data profiling and SQL feature engineering
- ✅ Data quality processing (sentinel handling, negative-bureau-debt clipping)
- ✅ Leakage audit and temporal validity analysis
- ✅ Deterministic dev/holdout split
- ✅ Logistic regression baseline vs. XGBoost candidate comparison
- ✅ Calibration investigation (Platt, isotonic) with holdout and bootstrap
  statistical validation — raw XGBoost selected
- ✅ Cost-sensitive business decision threshold (0.110), frozen from
  dev-only analysis
- ✅ Final untouched-holdout evaluation of the frozen model + threshold
- ✅ SHAP explainability (global + local, log-odds space, empirically verified)
- ✅ Streamlit deployment demo (`app/app.py`) around the frozen system

Engineered 27 applicant-level features from 2 relational credit-history
tables, producing a 147-feature modelling matrix. Current benchmark
(development set, stratified 5-fold CV): XGBoost ROC-AUC 0.7634 ± 0.0044,
PR-AUC 0.2499 ± 0.0087, versus a logistic regression baseline of ROC-AUC
0.7543 ± 0.0039, PR-AUC 0.2301 ± 0.0076. Final untouched-holdout evaluation
(61,503 applicants, never used during development): ROC-AUC 0.7713, PR-AUC
0.2650.

**Production probability source: raw, uncalibrated XGBoost.** Calibration
(Platt scaling, isotonic regression) was investigated, not assumed: Platt
consistently underperformed; isotonic's small holdout Brier-score edge
(0.066840 raw vs. 0.066826 isotonic) was tested with a 1,000-iteration
bootstrap and found not statistically distinguishable from zero (95% CI
[-6.81e-05, +1.03e-04]). The simpler, uncalibrated model was selected on
that basis — see `DECISIONS.md` for the full reasoning and
`reports/final_model_card.md` for the complete evidence trail.

**Business decision**: applicants are rejected if `p_default >= 0.110`, a
threshold selected on development data only under an expected-cost model
(loss given default = 0.60, margin if repaid = 0.08) and evaluated exactly
once on the untouched holdout set — see `reports/threshold_analysis.md` and
`reports/final_holdout_report.md`.

**Explainability**: SHAP (`TreeExplainer`) global and local explanations for
the frozen model, with the output space (log-odds/raw-margin, not
probability) verified empirically rather than assumed — see
`reports/shap_explainability_report.md`.

## Running the deployment demo

```bash
pip install -r requirements.txt

# One-time: build the model artifact the app loads (requires dev.parquet;
# see "Data" below). Never reads holdout data, never re-tunes the model.
python src/build_model_artifact.py

streamlit run app/app.py
```

The app never trains, calibrates, sweeps a threshold, or reads holdout
data — it loads the artifact built above and reuses the frozen pipeline's
prediction, decision, and explanation logic unchanged (`src/inference.py`).
Three synthetic, hand-constructed demo applicants (clearly labeled as such,
not derived from dev or holdout data) are available in the app for a quick
walkthrough. See `reports/phase10_deployment_report.md` for the full
architecture and the honest account of how a 147-feature model input is
reconstructed from a curated subset of applicant fields.

## Data

This project uses the Kaggle "Home Credit Default Risk" dataset
(`application_train.csv`, `bureau.csv`, `previous_application.csv`),
expected under `data/raw/` (gitignored — not redistributed in this repo).
Run the pipeline in order to regenerate `data/processed/`:

```bash
python src/profile_data.py
python src/features.py
python src/data_quality.py
```

## Testing

```bash
pip install -r requirements.txt
python src/build_model_artifact.py   # tests below require the model artifact
python -m pytest tests/ -v
```

## Documentation

- `DECISIONS.md` — full phase-by-phase decision log, including why each
  major choice (raw XGBoost over calibration, the 0.110 threshold, the
  deployment architecture) was made.
- `reports/final_model_card.md` — authoritative, versioned model summary.
- `reports/threshold_analysis.md`, `reports/final_holdout_report.md`,
  `reports/shap_explainability_report.md`,
  `reports/phase10_deployment_report.md` — per-phase evidence and writeups.

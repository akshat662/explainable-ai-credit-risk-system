# Deployment Layer Report (Phase 10)

A Streamlit demo layer around the frozen Phase 6-9 system: applicant input ->
feature construction -> frozen XGBoost model -> raw default probability ->
0.110 business threshold -> APPROVE / REJECT -> decision-aware SHAP
explanation. This phase adds a deployment/inference layer only -- no model,
feature-engineering, threshold, or SHAP code from prior phases was changed.

## Architecture

```
src/build_model_artifact.py   (run once, offline)
  -> models/xgboost_frozen.json      (frozen booster, XGBoost native format)
  -> models/feature_schema.json      (147-column contract: order, categorical
                                       category sets, dev median/mode template)

src/inference.py              (imported by the app; never trains)
  -> load_model() / load_schema() / load_explainer()
  -> build_feature_row(user_inputs, schema)
  -> predict_probability() / decide() / explain()

app/app.py                    (Streamlit UI; imports inference.py only)
  -> curated applicant-input form
  -> risk result + threshold visualization
  -> decision-aware SHAP explanation
  -> model information panel
```

## Why a build-once artifact, not train-on-start

Every prior phase (holdout evaluation, SHAP) retrains the frozen XGBoost
configuration from scratch by calling
`evaluate_holdout_calibration.train_final_model()` -- correct for one-shot
report generation, but unsuitable for an interactive app that should respond
in under a second. `src/build_model_artifact.py` calls that exact same
function once, then persists the result via XGBoost's native
`model.save_model()`. The app (`src/inference.py`) only ever loads that
artifact; it never fits, cross-validates, sweeps a threshold, or reads
holdout data. This was verified both statically (source-scan tests for
forbidden identifiers/calls) and empirically (an app rerun after the first
completes in well under a second, inconsistent with retraining on 246,008
rows).

## The 147-feature reconstruction problem, and how it was resolved

The frozen model's 147 features are produced by a relational SQL aggregation
over `bureau.csv`/`previous_application.csv` (`src/features.py`) joined onto
`application_train`. There is no way for an interactive form to re-run that
aggregation for a hypothetical applicant who has no rows in those tables --
and exposing all 147 raw fields individually (dozens of them are
building/apartment descriptors or administrative document flags with little
individual signal) would make the form unusable, which the project brief
explicitly ruled out.

The resolution, in order of how much of the 147-column vector each part
covers:

1. **A curated set of ~35 raw fields** the UI actually collects, grouped as
   Applicant profile, Financial information, Employment, External credit
   scores, Credit history, and Previous application information. Bureau and
   previous-application fields are collected as the *same already-aggregated
   summary numbers* the model consumes (e.g. "total bureau loans"), not
   fabricated per-loan records -- the user supplies the aggregate directly,
   matching the model's real input contract.
2. **Six ratio features** (`credit_income_ratio`, `annuity_income_ratio`,
   `credit_annuity_ratio`, `goods_credit_ratio`, `income_per_person`,
   `debt_income_ratio`) recomputed in `inference.py` with the exact formulas
   `src/features.py`'s SQL uses, division-by-zero guarded the same way
   (`NULLIF(denominator, 0)` -> missing, never an error).
3. **`DAYS_EMPLOYED` sentinel handling** reuses
   `data_quality.DAYS_EMPLOYED_SENTINEL` and reproduces
   `clean_sentinel_values()`'s CASE WHEN logic exactly, rather than inventing
   a new rule.
4. **"No history" is preserved as missing, not zero.** Unchecking "has
   bureau-reported credit history" / "has previous Home Credit applications"
   sets those columns to `None` (NaN), not `0` -- the same distinction
   `src/features.py` and `reports/final_model_card.md` already document
   ("absence of history is a different fact from a zero-valued history").
5. **Every remaining feature** (mostly building/apartment descriptors,
   `FLAG_DOCUMENT_2`..`21`, and other low-signal administrative columns) is
   filled from a **frozen per-column dev-set median (numeric) / mode
   (categorical) template**, computed once in `build_model_artifact.py` from
   `dev.parquet` only. This is a disclosed demo simplification, not a claim
   that it reflects any individual applicant -- stated explicitly in the
   app's "Model information" panel and, where a template-filled field
   actually appears among an applicant's top SHAP contributors, inline next
   to that contributor.

The curated fields were deliberately chosen to cover every one of the top-15
global SHAP features from `reports/shap_explainability_report.md` (directly,
or as an input to one of the six ratio features) -- so the fields a user can
actually change are the fields that actually drive this model's predictions,
not a disconnected subset.

## A categorical-encoding pitfall this design had to get right

XGBoost's native categorical support (`enable_categorical=True`, used
throughout this project) recodes categorical inputs by *value* against the
training-time category set -- confirmed empirically, not order/position
dependent. However, a single-row DataFrame whose categorical value is
missing has **zero** locally-inferred pandas categories (pandas can't infer
categories from one `NaN`), and XGBoost hard-errors on a zero-category
column (`Categorical feature must have at least one category`). Every
categorical column in `inference.build_feature_row()` is therefore cast with
an explicit `categories=` list taken from the frozen schema
(`feature_schema.json`), which both avoids that failure mode and guarantees
every dropdown in the app only ever offers values the model actually saw
during training.

## What the app does and does not do

Does: load the frozen artifact once, construct one applicant's 147-feature
row, predict the raw default probability, apply the frozen 0.110 threshold,
and produce a decision-aware local SHAP explanation via the unchanged
`shap_explainability.explain_applicant()`.

Does not: train, cross-validate, calibrate, sweep or re-derive the
threshold, read the untouched final-evaluation set, or modify any frozen
model/feature/threshold code. Verified by static source-scan tests
(`tests/test_inference.py`) and a headless UI smoke test
(`tests/test_app.py`, via `streamlit.testing.v1.AppTest`).

## Demo data

The three "load a demo applicant" profiles in the app are synthetic and
hand-constructed -- not real applicants, and not derived from this project's
development or holdout data. They were tuned (external credit scores, bureau
debt, prior-application history) to illustrate a clear APPROVE case, a clear
REJECT case, and a near-threshold REJECT case (~1 percentage point past the
boundary), each verified against the frozen model and threshold.

## Limitations

- The dev-median/mode template means two applicants who differ only in
  fields this form doesn't collect will get identical predictions for those
  fields' contribution -- a disclosed simplification, not a claim of a
  complete applicant profile.
- As with every prior phase, reject inference applies: the underlying model
  was trained only on historically-approved applicants.
- This is a portfolio demonstration of an explainable, business-aware
  decision pipeline, not a production banking system -- no authentication,
  persistence, or regulatory compliance claims are made.

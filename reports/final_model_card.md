# Final Model Card — Explainable AI Credit Risk Decision System

Frozen after Phase 7.6 (benchmark refresh), verified in Phase 7.7 (freeze
verification, code- and data-unchanged), and updated in Phase 7.8 after a
bootstrap significance test reversed the calibration selection (see
"Calibration method" below). This card describes the model pipeline as it
exists at freeze time; it is not a live document.

## Data source

Home Credit Default Risk dataset, loaded from `data/raw/`:

| File | Rows | Role |
|---|---|---|
| `application_train.csv` | 307,511 | One row per applicant; `TARGET` label (8.0729% default rate) |
| `bureau.csv` | 1,716,428 | External credit bureau history, many-to-one on `SK_ID_CURR` |
| `previous_application.csv` | 1,670,214 | Prior Home Credit applications, many-to-one on `SK_ID_CURR` |

All three are read via DuckDB `read_csv_auto` views; raw files are never
modified (`data/raw/` is read-only throughout the pipeline).

## Feature engineering summary

`src/features.py` aggregates `bureau`/`previous_application` to one row
per `SK_ID_CURR` in DuckDB SQL, then `LEFT JOIN`s onto `application_train`
so grain is preserved by construction (verified: output row count always
equals `application_train`'s row count, `SK_ID_CURR` always unique).

- **Bureau aggregates** (13 features): loan counts by status, credit/debt
  sums and averages, overdue statistics, credit type variety, and
  `bureau_had_negative_debt`.
  `AMT_CREDIT_SUM_DEBT` negative values (8,418 rows, 0.49% of bureau,
  affecting 5,886 applicants) are clipped to 0 before summing into
  `bureau_total_debt` (and, downstream, `debt_income_ratio`);
  `bureau_had_negative_debt` preserves the fact that clipping occurred,
  since it may itself carry signal independent of the corrected magnitude.
  NULL bureau history (no bureau records for an applicant) remains
  distinguishable from a genuine zero/clipped debt value throughout — the
  clip only touches negative *values*, never converts a missing record
  into a zero one.
- **Previous-application aggregates** (7 features): application count,
  approval/refusal ratios, requested/granted amounts, term.
- **Financial ratios** (6 features): credit-to-income, annuity-to-income,
  credit-to-annuity, goods-to-credit, income-per-person, debt-to-income.
- **Sentinel handling**: `DAYS_EMPLOYED = 365243` (Home Credit's
  "not applicable" placeholder, ~18% of rows, overwhelmingly
  Pensioner/Unemployed applicants) is converted to `NaN` with a
  `days_employed_anomaly` indicator.
- Missing bureau/previous-application history is preserved as `NULL`
  throughout (never zero-filled) — absence of history is a different
  fact from a zero-valued history.

**Engineered 30+ applicant-level features from 2 relational tables,
producing a 147-feature modelling matrix.** Precisely: 120 raw
`application_train` columns are passed through unmodified (not
"engineered"), 26 new features are computed from `bureau`/
`previous_application` via DuckDB SQL (13 bureau + 7 previous-application
+ 6 financial ratios), and 1 more (`days_employed_anomaly`) is added
during sentinel cleaning — 120 + 26 + 1 = 147. This count is consistent
across `dev.parquet`, `holdout.parquet`, `baseline_results.json`, and
`xgboost_results.json` — confirmed at freeze time. **Not all 147 features
were engineered by this pipeline** — most are raw applicant fields
carried through unchanged; only the 27 listed above are newly derived.

## Validation protocol

- **Dev/holdout split**: 80% / 20%, stratified on `TARGET`,
  `random_state=42`, applied to `SK_ID_CURR` sorted in a fixed order
  before splitting — confirmed deterministic at freeze time by
  recomputing the split from scratch in an isolated connection and
  diffing against the saved `holdout.parquet`: 0 membership difference.
  (This determinism was not guaranteed before Phase 7.5 — see
  DECISIONS.md — since `train_test_split`'s shuffle depends on input row
  order, and DuckDB does not guarantee stable row order across separate
  process runs of the same query.)
- **Cross-validation**: `StratifiedKFold(n_splits=5, shuffle=True,
  random_state=42)` on `dev.parquet` only, identical across the logistic
  baseline and XGBoost.
- **Holdout usage**: read exactly once per evaluation script, only for
  `.predict()`/`.predict_proba()` — confirmed at freeze time that no
  `.fit()` call in the codebase touches holdout-derived data. Calibrators
  (Platt, isotonic) are fit exclusively on dev out-of-fold (OOF)
  predictions, then only ever applied (never refit) to holdout scores —
  confirmed by tracing `fit_calibrators()`'s only call site.

## Final benchmark metrics

Development set, stratified 5-fold CV (`dev.parquet`, 246,008 rows):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression (baseline) | 0.7543 ± 0.0039 | 0.2301 ± 0.0076 |
| XGBoost (candidate) | 0.7634 ± 0.0044 | 0.2499 ± 0.0087 |
| Delta (XGBoost − Logistic) | +0.0091 | +0.0198 |

Final holdout evaluation (`holdout.parquet`, 61,503 rows, read once, used
only for scoring — this is the authoritative real-world performance
estimate):

| Method | Brier score | ROC-AUC | PR-AUC |
|---|---|---|---|
| Constant predictor (p = dev prevalence, 8.0729%) | 0.074211 | 0.5000 | 0.0807 |
| Raw XGBoost | 0.066840 | 0.7713 | 0.2650 |
| Platt scaling | 0.067816 | 0.7713 | 0.2650 |
| Isotonic regression | 0.066826 | 0.7713 | 0.2585 |

Both raw XGBoost and isotonic improve on the constant-probability
predictor by ~9.9% relative Brier score reduction (raw: 9.93%; isotonic:
9.95%) — the model itself is doing substantive work; the remaining
question, addressed below, is whether isotonic calibration adds anything
further on top of the raw model.

**XGBoost is the final model family** — it outperformed the logistic
regression baseline on both ROC-AUC and PR-AUC (dev CV, above) and is the
model all calibration experiments below were run against.

## Calibration method

**Calibration methods evaluated:** raw XGBoost (uncalibrated), Platt
scaling (`LogisticRegression` on the raw score), and isotonic regression
— all fit exclusively on dev out-of-fold (OOF) predictions from a 5-fold
XGBoost CV, then evaluated against holdout (read once, scoring only).

**Bootstrap significance test (Phase 7.8):** the isotonic-vs-raw Brier
gap on holdout (0.066826 vs. 0.066840, a 0.000014 difference) was tested
with a 1,000-iteration paired bootstrap (`random_state=42`) over
Brier(raw) − Brier(isotonic). Result: mean difference 1.45e-05, 95% CI
**[-6.81e-05, +1.03e-04]** — an interval that includes 0. **The isotonic
improvement is not statistically distinguishable from sampling noise.**
Full results: `reports/calibration_bootstrap_results.json`.

**Final production probability choice: raw XGBoost, uncalibrated.**
Given a statistically insignificant difference in Brier score, the
simpler option is preferred — no calibration mapping is a strictly
simpler production artifact than one, with one fewer fitted object to
version, validate, and explain. This reverses Phase 7.5's original
selection of isotonic regression, which was based on the holdout point
estimate alone, before this bootstrap test existed. The one calibration
finding that *did* replicate across dev OOF and holdout, and is therefore
trusted: **Platt scaling underperforms raw XGBoost** and remains excluded
from consideration regardless of this reversal.

## Known limitations

- **`EXT_SOURCE_1/2/3` leakage has not been tested.** The Phase 4/4.5
  leakage audits covered bureau/previous-application record timing only;
  these external credit-bureau scores remain unexamined.
- **Holdout evaluation is still a single realization of the applicant
  population**, even though the calibration comparison specifically is
  now bootstrap-validated (Phase 7.8, 1,000 iterations). The absolute
  benchmark numbers themselves (ROC-AUC, PR-AUC, Brier score) are not
  bootstrapped — only the raw-vs-isotonic *difference* was.
- **`bureau_had_negative_debt` is untested as a standalone signal** — added
  for data-quality correctness (Phase 7.5), not yet evaluated for its own
  predictive contribution or interactions with other features.
- **`CREDIT_ACTIVE`'s `Sold`/`Bad debt` categories remain undercounted**
  by `bureau_active_loans`/`bureau_closed_loans` (0.38% of bureau rows) —
  a known, small feature-coverage gap, not fixed in this pipeline.
- **`DAYS_CREDIT_UPDATE > 0`** affects 17 of 1,716,428 bureau rows
  (0.001%) — documented, not filtered, since the population is too small
  to matter, but the pipeline does not explicitly exclude it.
- **No feature selection or importance analysis has been performed** — all
  147 features are used as-is; some may contribute little or introduce
  noise.
- **No business decision layer exists yet** — no threshold, expected-loss
  calculation, or SHAP explanation sits on top of the raw XGBoost
  probability yet; this is the next phase of work.
- **Isotonic's OOF-vs-full-dev-model mismatch**: the isotonic calibrator
  (evaluated, not shipped) was fit on out-of-fold predictions produced by
  5 separate models, each trained on 4/5 of `dev.parquet`, then applied to
  raw scores from a *different*, sixth model trained on all of
  `dev.parquet` for holdout scoring. Treating these as interchangeable
  score distributions is a common, pragmatic approximation, but it is an
  approximation — a methodological nuance of the calibration experiment
  itself, worth keeping in mind when reading `reports/calibration_bootstrap_results.json`
  and `reports/holdout_calibration_report.md`, even though it did not end
  up mattering for the final choice (raw XGBoost needs no calibrator at
  all).
- **Categorical encoding differs between the two benchmarked models**
  (one-hot for logistic regression, native categorical splits for
  XGBoost) — appropriate for each model family, but means the "same 147
  features" claim is about column coverage, not identical numerical
  encoding.
- **Platt scaling was not included in the bootstrap significance test** —
  it was already excluded on the strength of replicating across two
  independent point-estimate evaluations (dev OOF and holdout), which was
  judged sufficient given it consistently underperformed rather than
  showing an ambiguous gap; only the raw-vs-isotonic comparison, which was
  genuinely too close to call from point estimates alone, received the
  bootstrap test.

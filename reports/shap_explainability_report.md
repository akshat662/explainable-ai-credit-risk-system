# SHAP Explainability Report (Phase 9)

## Model used

Frozen XGBoost (`train_xgboost.MODEL_PARAMS`, imported unchanged), trained once on all of `dev.parquet` -- the same deterministic training call used and validated in Phase 8.5. No tuning, no new model family. Raw, uncalibrated probability output (Phase 7.8 decision, unchanged).

## Data used

**Global**: a deterministic sample of 2,000 development applicants (`random_state=42`), sampled from the 246,008-row dev set used throughout this project. Sampling is used because (a) SHAP importance rankings stabilize well before the full dev set is used, and (b) a beeswarm plot over 246,008 points is unreadable and slow to render -- 2,000 is a standard SHAP sample size. `holdout.parquet` is not read anywhere in this phase, and no holdout label is used for feature ranking.

**Local**: a single development applicant (`SK_ID_CURR=136133`), for demonstration purposes only. The `explain_applicant()` function itself is data-source-agnostic (it accepts any correctly-prepared single-row DataFrame) and is intended for direct reuse in Phase 10.

## SHAP method

`shap.TreeExplainer` (exact, tree-structure-based SHAP for gradient-boosted trees) — `shap==0.52.0`, `xgboost` matching `train_xgboost.MODEL_PARAMS`.

## SHAP output space

**Verified empirically against the installed shap/xgboost versions, not assumed**: `TreeExplainer`'s default `model_output` is `"raw"` for `XGBClassifier`, meaning SHAP values are additive in **log-odds (raw margin) space**:

```
base_value + sum(shap_values) == model.predict(X, output_margin=True)
sigmoid(base_value + sum(shap_values)) == model.predict_proba(X)[:, 1]
```

Additivity check: `max(|base_value + sum(SHAP) - raw_margin_output|)` = **7.15e-06** over the 2,000-applicant sample (tolerance 1e-04) — **passed**.

Individual SHAP values are reported as log-odds contributions throughout this project's outputs — never rescaled or described as a literal percentage-point probability change, since `sigmoid` is nonlinear and a log-odds decomposition does not translate into an additive probability-space decomposition. The predicted probability, its distance to the 0.110 threshold, and the APPROVE/REJECT decision are computed directly via `model.predict_proba` / `model.predict(..., output_margin=True)` — never by summing or transforming SHAP values.

## Global importance results

Top 15 features by mean |SHAP value| (log-odds space, dev sample, n=2,000):

| Rank | Feature | Display name | Mean \|SHAP\| (log-odds) |
|---|---|---|---|
| 1 | `EXT_SOURCE_2` | External credit score 2 | 0.3278 |
| 2 | `EXT_SOURCE_3` | External credit score 3 | 0.3217 |
| 3 | `EXT_SOURCE_1` | External credit score 1 | 0.1548 |
| 4 | `credit_annuity_ratio` | Credit-to-annuity ratio | 0.1533 |
| 5 | `ORGANIZATION_TYPE` | Employer organization type | 0.1439 |
| 6 | `goods_credit_ratio` | Goods price-to-credit ratio | 0.1095 |
| 7 | `prev_avg_term` | Average previous loan term | 0.1025 |
| 8 | `CODE_GENDER` | Gender | 0.0971 |
| 9 | `DAYS_EMPLOYED` | Days employed (negative = past) | 0.0924 |
| 10 | `prev_grant_ratio` | Previous grant-to-request ratio | 0.0915 |
| 11 | `DAYS_BIRTH` | Applicant age (days, negative = past) | 0.0871 |
| 12 | `OWN_CAR_AGE` | Car age (years) | 0.0862 |
| 13 | `AMT_ANNUITY` | Loan annuity (installment amount) | 0.0823 |
| 14 | `debt_income_ratio` | Bureau debt-to-income ratio | 0.0730 |
| 15 | `NAME_EDUCATION_TYPE` | Education level | 0.0685 |

Full top-20 table: `reports/shap_global_importance.csv`. Bar chart: `reports/shap_global_importance.png`. Beeswarm/summary plot (shows direction and per-applicant spread, not just magnitude): `reports/shap_summary.png`.

These features are associated with higher or lower model risk scores. This ranking describes the model's learned behavior, not a causal claim about real-world default risk.

## Local explanation example

```
Applicant: 136133

Predicted default probability: 16.0%
Decision threshold: 11.0%
Distance from threshold: +5.0 percentage points
Decision: REJECT

Main risk contributors (higher model risk score, log-odds space):
  1. External credit score 2 (EXT_SOURCE_2 = 0.06522930099879802), SHAP=+0.7576
  2. Days since ID document issued (DAYS_ID_PUBLISH = -228.0), SHAP=+0.1677
  3. Employer organization type (ORGANIZATION_TYPE = XNA), SHAP=+0.1330
  4. Average previous loan term (prev_avg_term = 24.0), SHAP=+0.0972
  5. Price of goods being financed (AMT_GOODS_PRICE = 450000.0), SHAP=+0.0931

Main protective contributors (lower model risk score, log-odds space):
  1. External credit score 3 (EXT_SOURCE_3 = 0.5495965024956946), SHAP=-0.1622
  2. External credit score 1 (EXT_SOURCE_1 = 0.4345178938966018), SHAP=-0.1234
  3. Goods price-to-credit ratio (goods_credit_ratio = 1.0), SHAP=-0.0978
  4. Occupation (OCCUPATION_TYPE = None), SHAP=-0.0890
  5. Gender (CODE_GENDER = F), SHAP=-0.0832
```

Full structured output: `reports/shap_local_example.json`. The `explain_applicant()` function producing this is reusable directly (same signature, same return structure) for Phase 10's Streamlit interface.

## Relationship to the 0.110 decision threshold

SHAP explains *why the model produced a given prediction*. The 0.110 threshold — frozen independently in Phase 8 from dev-only expected-cost analysis, unrelated to and unaffected by SHAP — is what converts that prediction into a business decision. These are two separate steps and are kept conceptually separate throughout this module: `explain_applicant()` computes the probability and decision first (via the frozen model and threshold, with no SHAP involvement), and only then computes SHAP contributions to explain *why* the model arrived at that probability. SHAP does not determine, adjust, or influence the threshold in any way.

## Limitations

- SHAP explains the model's learned behavior, not a real-world causal mechanism — a feature being a "risk contributor" means the model's score moved in the risk direction when that feature took its observed value, not that the feature *causes* default.
- SHAP does not establish or guarantee fairness, absence of bias, or regulatory compliance; no such claim is made here.
- Global importance is computed on a 2,000-applicant deterministic sample, not the full 246,008-row dev set — chosen for plot readability and runtime, and unlikely to materially change a top-15/20 ranking, but not verified against the full-dev ranking in this phase.
- Missing values are passed through to the model exactly as the frozen pipeline produces them (e.g. `NaN` for bureau features when an applicant has no bureau history) — `TreeExplainer` attributes these via the tree's learned default-direction split behavior, not any artificial zero-fill introduced for explainability.
- Reject inference (documented in `reports/final_model_card.md`) applies here too: explanations, like the model itself, are only demonstrated against the historically-approved applicant population.


# Engineering Decisions Log

This document records important architectural and technical decisions made during development.

---

## Decision Template

### Decision:
 

### Context:


### Choice:


### Reasoning:


### Alternatives considered:


### Impact:

---

## Decision: Use DuckDB for raw data profiling

### Context:
Phase 1 needs to load `application_train.csv`, `bureau.csv`, and
`previous_application.csv` and compute basic profiling metrics
(row counts, default rate) without yet doing feature engineering or
modeling.

### Choice:
Use DuckDB to register each CSV as a SQL view (`read_csv_auto`) and
compute metrics via SQL aggregation, rather than loading full CSVs into
pandas DataFrames.

### Reasoning:
- DuckDB views are lazy — a view is registered instantly and the file is
  only scanned when queried, so profiling large Home Credit tables (some
  with millions of rows) doesn't require loading them fully into memory.
- SQL is the natural fit for later feature engineering phases (joins across
  `bureau`/`previous_application`), so starting the pipeline in DuckDB now
  keeps a consistent access pattern across phases instead of mixing pandas
  and SQL approaches.
- DuckDB requires no external server/process (unlike Postgres), which keeps
  the pipeline simple and fully reproducible from a single Python script.

### Alternatives considered:
- **pandas `read_csv` + `.describe()`**: simpler for a single table, but
  doesn't scale as well to multi-table joins planned for the feature
  engineering phase, and duplicates logic that SQL already expresses well.
- **Loading into a persistent DuckDB file (`.duckdb`)**: unnecessary for
  profiling since no state needs to persist between runs; an in-memory
  connection is sufficient and avoids managing a stray database file.

### Impact:
Establishes DuckDB as the primary data-access layer for this project.
Future phases (feature engineering, leakage audits) should build on the
same view-based pattern for consistency.

---

## Decision: Separate profiling from feature engineering

### Context:
It would be possible to compute richer statistics (per-column nulls,
distributions, cross-table join checks) in this same script.

### Choice:
Phase 1 (`src/profile_data.py`) is scoped strictly to: load raw CSVs as
views, and compute high-level metrics for `application_train`
(row count, default count, default percentage). No joins, no derived
features, no leakage checks yet.

### Reasoning:
- Keeps each pipeline stage auditable and independently testable — a
  reviewer can verify raw data profiling is leakage-free before any
  feature engineering logic is introduced.
- Matches the project's phased structure (profiling → feature engineering →
  leakage audit → modeling), avoiding one script that tries to do
  everything and is harder to reason about or test in isolation.
- Metrics are persisted to `reports/metrics.json` so later phases (and CI)
  can reference the raw baseline without recomputing it or re-deriving it
  from a monolithic script.

### Alternatives considered:
- **One combined profiling + feature engineering script**: rejected —
  harder to isolate bugs, and blurs the line between "what the raw data
  looks like" and "what we derived from it," which matters for leakage
  auditing later.

### Impact:
Establishes a per-phase script convention (`src/profile_data.py`, later
`src/feature_engineering.py`, `src/audit_leakage.py`, etc.), each with a
narrow, single-purpose responsibility.

---

## Decision: DuckDB SQL aggregation over pandas groupby for feature engineering

### Context:
`bureau.csv` (1.7M rows) and `previous_application.csv` (1.67M rows) both
need to be collapsed from one-row-per-loan to one-row-per-applicant before
joining onto `application_train` (307K rows). This aggregation could be
done with pandas `.groupby().agg()` after loading each CSV fully, or with
DuckDB `GROUP BY` views over the same views already used in Phase 1.

### Choice:
All aggregation (`bureau_agg`, `prev_app_agg`) and the final join
(`features_naive`) are expressed as DuckDB SQL views. `src/features.py`
never loads a full raw CSV into a pandas DataFrame, and the final feature
table is written to Parquet directly from DuckDB (`COPY ... TO ... (FORMAT
PARQUET)`), so pandas is not imported at all in this script.

### Reasoning:
- Keeps the same DuckDB-as-primary-data-access pattern established in
  Phase 1 — one consistent way of touching raw data across the pipeline.
- DuckDB's `GROUP BY`/`FILTER` push aggregation down to the file scan, so
  the two largest tables are never fully materialized in Python memory,
  which matters as the multi-table joins in later phases grow.
- SQL `SUM`/`AVG`/`COUNT` ignore NULLs per-group by default, which is
  exactly the behavior needed to keep bureau history genuinely missing
  (see the LEFT JOIN decision below) rather than accidentally coercing it
  to 0 the way a naive pandas `.sum()` on an empty group could.

### Alternatives considered:
- **pandas `groupby` after `read_csv`**: rejected — requires loading both
  multi-million-row tables fully into memory before aggregating, and
  reintroduces a second data-access pattern alongside DuckDB for no
  benefit at this stage.

### Impact:
`src/features.py` has no pandas dependency. Future feature-engineering
phases (e.g. `POS_CASH_balance`, `installments_payments`) should follow
the same aggregate-as-a-DuckDB-view pattern before joining.

---

## Decision: Feature engineering output is one row per SK_ID_CURR

### Context:
`application_train` is the modeling grain (one row = one credit decision).
`bureau` and `previous_application` are one-to-many against it (5-6x more
rows). The Phase 2 output needs to be directly usable as a model training
table.

### Choice:
`bureau` and `previous_application` are aggregated to one row per
`SK_ID_CURR` *before* joining, and `validate_features()` asserts that the
final `features_naive` row count equals `application_train`'s row count
and that `SK_ID_CURR` is unique in the output — hard failures, not
warnings, if either check fails.

### Reasoning:
- Matches the actual prediction grain: a credit decision is made once per
  application, so the feature table must have exactly one row per
  applicant to be a valid model input.
- Joining pre-aggregated (already unique-keyed) tables via LEFT JOIN is
  structurally 1:1 — it cannot fan out `application_train`'s row count,
  so the assertion is a correctness backstop against a future join being
  added incorrectly (e.g. joining raw `bureau` directly instead of
  `bureau_agg`).

### Alternatives considered:
- **Row-level (loan-level) table with a separate rollup step later**:
  rejected — adds an extra transformation stage for no benefit, since
  every downstream consumer (model training, SHAP explanations) needs the
  applicant grain, not the loan grain.

### Impact:
`data/processed/features_naive.parquet` is directly model-ready in shape
(one row per applicant) once leakage auditing (a separate later phase)
has run over it.

---

## Decision: Preserve LEFT JOIN missing values instead of zero-filling

### Context:
`bureau_agg` and `prev_app_agg` only contain `SK_ID_CURR` values that
appear in `bureau.csv` / `previous_application.csv`. An applicant with no
bureau history (44,020 of 307,511 in the current data) or no previous
applications (16,454) will not have a matching row in those aggregates.

### Choice:
The join is a plain `LEFT JOIN` with no `COALESCE(..., 0)` anywhere in
`build_applicant_features()`. An applicant absent from bureau/previous
history gets `NULL` for every corresponding engineered feature
(`bureau_total_loans`, `prev_app_count`, etc.), not `0`.

### Reasoning:
- "No bureau history" and "bureau history that sums to zero debt" are
  different facts about an applicant and must remain distinguishable — the
  current data has both (7,360 applicants have bureau records but all-NULL
  debt fields, vs. 69,689 with a real, non-null `0` total debt). Zero-filling
  at this stage would silently collapse that distinction.
- Imputation is a modeling decision (what value best represents "unknown"
  for a given model), not a data engineering one, and belongs in a later,
  explicit phase where it can be tuned and audited — not hardcoded here.
- `NULLIF` is used in ratio calculations (e.g. `NULLIF(AMT_INCOME_TOTAL,
  0)`) purely to prevent divide-by-zero turning into `inf`/error; it does
  not fill in any missing source value, so it doesn't conflict with this
  decision.

### Alternatives considered:
- **Zero-fill bureau/previous-application aggregates at join time**:
  rejected — makes "no history" indistinguishable from "zero history,"
  which is a leakage-adjacent correctness bug a downstream model could
  learn from spuriously.

### Impact:
`features_naive.parquet` still contains NULLs by design. Any model trained
directly on it must handle missing values (native NaN support, e.g.
XGBoost, or an explicit imputation step) — this is intentionally deferred
to a later phase.

---

## Decision: Convert the DAYS_EMPLOYED sentinel to NaN with an explicit indicator

### Context:
`DAYS_EMPLOYED` contains the value `365243` (~1000 years) for 55,374 of
307,511 applicants (~18%) — confirmed to be almost entirely `Pensioner`
(55,352) and `Unemployed` (22) applicants. This is Home Credit's known
placeholder for "employment duration not applicable," not a measured value.

### Choice:
`clean_sentinel_values()` in `src/data_quality.py` replaces `365243` with
`NULL` and adds a `days_employed_anomaly` (1/0) column recording where the
sentinel was found, before any downstream ratio or aggregate touches
`DAYS_EMPLOYED`.

### Reasoning:
- `365243` is not a real employment duration — treating it as a numeric
  value would let it dominate any average, ratio, or model split involving
  `DAYS_EMPLOYED` (e.g. it would appear as a multi-century outlier).
  Converting it to `NULL` is the only way to prevent that distortion.
- The fact that an applicant *had* the sentinel is itself informative (it
  flags retirees/unemployed applicants, a segment with a different risk
  profile) independent of the now-missing value — hence the indicator
  column rather than silently dropping the information.
- This must happen before any future employment-related ratio feature
  (e.g. income per year employed), otherwise such a ratio would compute a
  meaningless near-zero value against a 1000-year denominator instead of
  correctly propagating `NULL`.

### Alternatives considered:
- **Leave the sentinel as a numeric value**: rejected — silently corrupts
  any statistic or ratio computed over `DAYS_EMPLOYED`.
- **Drop the indicator, keep only the NULL**: rejected — would discard a
  segment-identifying signal (Pensioner/Unemployed) that costs nothing to
  preserve.

### Impact:
Any feature or model built on `DAYS_EMPLOYED` after this phase must treat
it as having genuine missingness, and can use `days_employed_anomaly` as a
free, leakage-safe segment flag.

---

## Decision: Preserve missing values instead of imputing in Phase 3

### Context:
`reports/missing_values.md` profiles missingness across all ~148 features
in `features_clean`. Missingness ranges from ~0% to ~70% (building/
apartment descriptive columns are the most missing; `bureau_*`/`prev_*`
engineered features are missing at 5-17% by construction, per
[the Phase 2 LEFT JOIN decision](#decision-preserve-left-join-missing-values-instead-of-zero-filling)).

### Choice:
No imputation is performed anywhere in `src/data_quality.py`. The missing
value report documents an intended *future* treatment per feature category
(`EXT_SOURCE_*` → keep NaN, consider a missingness indicator; `bureau_*`/
`prev_*` → keep NaN, since NaN means "no history" not "zero"; everything
else → left unchanged for now) but does not act on it yet.

### Reasoning:
- The correct imputation strategy (mean/median fill, model-based, or
  leaving NaN for a NaN-native model like XGBoost) is a modeling decision
  that depends on which model family is chosen — deciding it now, before
  Phase 4 modeling exists, would be premature and likely wrong for at
  least one candidate model.
- Some missingness here is not "missing data" in the usual sense at all —
  `bureau_*`/`prev_*` NaNs encode "this applicant has no such history,"
  a fact a model may want to use directly (e.g. via a NaN-aware split)
  rather than have overwritten by an imputed value.
- Committing to imputation before the holdout split existed would also
  risk fitting an imputation statistic (e.g. a column mean) that
  implicitly used holdout rows, contaminating the holdout before it's
  even created — deferring imputation past the split avoids this entirely.

### Alternatives considered:
- **Simple mean/median imputation now**: rejected — irreversible
  information loss (real vs. imputed values become indistinguishable) made
  before any model has stated what it actually needs.

### Impact:
`dev.parquet`/`holdout.parquet` both still contain NaNs. Any Phase 4
training code must either use a NaN-native model or add an explicit,
dev-only-fitted imputation step.

---

## Decision: Create the dev/holdout split before any modeling, and lock it

### Context:
Phase 4 will involve model training, calibration, and threshold tuning —
all of which can overfit to a dataset if the same data is used to both
develop and evaluate a model. The project's goal is a *trustworthy*
decision system, which requires an honest, untouched estimate of
real-world performance.

### Choice:
`create_dev_holdout_split()` splits applicants 80/20 into
`data/processed/dev.parquet` / `data/processed/holdout.parquet` using
`sklearn.model_selection.train_test_split(stratify=TARGET, random_state=42)`,
run once, in this phase — before any model exists. `validate_split()` hard-
asserts the split is disjoint (`SK_ID_CURR` never in both), row-complete
(`dev + holdout == total`), and target-balanced (default rate within 1
percentage point between the two). Verified: dev default rate 8.0729%,
holdout 8.0728%, and confirmed reproducible under the fixed `random_state`
(identical `SK_ID_CURR` membership across repeated runs, independent of
Parquet file byte-level metadata).

### Reasoning:
- Splitting before any model or feature-selection work exists is the only
  way to guarantee the holdout wasn't (even implicitly) used to inform a
  decision — the project's own rule ("never use holdout data for
  decisions") is only enforceable if the holdout is locked before there's
  anything to decide.
- Only `SK_ID_CURR` + `TARGET` (2 of ~148 columns) are materialized into
  pandas for `train_test_split`, since that's the minimum sklearn's API
  needs; the full feature table is filtered and written by DuckDB directly,
  keeping the DuckDB-first architecture for the actual data movement.
- Stratifying on `TARGET` keeps the already-rare default class (~8%)
  represented proportionally in both splits — an unstratified split risks
  a holdout default rate different enough from dev to make evaluation
  results misleading purely from sampling variance.

### Alternatives considered:
- **K-fold cross-validation only, no fixed holdout**: rejected — the
  project needs one final, untouched number for reporting business
  decisions, not just a cross-validation estimate that could still be
  indirectly tuned against across many modeling iterations.

### Impact:
From this point forward, `holdout.parquet` must not be read by any
exploratory, feature-selection, calibration, or threshold-tuning code —
only by a final evaluation step once a model is fully decided. `dev.parquet`
is the only file later phases should touch during development.

---

## Decision: Leakage audit runs before any model is trained for real

### Context:
Phase 4 introduces the first model training in this project — an
XGBoost classifier — but its stated purpose is to test the *feature
pipeline*, not to produce a model anyone should keep.

### Choice:
Before any "real" model development happens, `src/leakage_experiment.py`
trains the same lightweight, untuned model on two feature variants
(`features_naive.parquet` vs `features_filtered.parquet`) built from the
identical development set, and compares them.

### Reasoning:
- Any leakage baked into the SQL feature pipeline (Phases 2-3) would
  silently inflate every subsequent model's reported performance,
  including on the holdout — by the time a "real" model is built and its
  holdout score looks good, it would be too late to tell whether that
  score reflects genuine predictive signal or leaked information.
  Auditing the pipeline first, with a throwaway model, catches this
  before it can contaminate a decision anyone would act on.
- Using an untuned, fixed-hyperparameter model for the audit is
  deliberate: any performance gap between naive and filtered must be
  attributable to the data, not to incidentally different tuning — this
  is a reliability test, not a benchmark.

### Alternatives considered:
- **Train the final model first, audit leakage only if performance looks
  suspiciously high**: rejected — "looks suspiciously high" is not a
  reliable signal (leakage can produce plausible, not just obviously
  inflated, scores), and it reverses the burden of proof for a project
  whose stated goal is trustworthiness, not just accuracy.

### Impact:
No model trained before this audit is treated as a candidate for
deployment. Phase 5 modeling work can build on `features_naive.parquet`
(or a future revision) only once its leakage status is understood, not
assumed.

---

## Decision: Naive vs. temporally-filtered comparison as the leakage test

### Context:
"Leakage" is not directly observable — the audit needs some form of
controlled comparison to make it detectable at all.

### Choice:
`build_features(time_filtered=True)` reproduces the exact same
aggregation pipeline as the default, with one difference: bureau is
restricted to `DAYS_CREDIT <= 0` and previous_application to
`DAYS_DECISION <= 0` before aggregating. Comparing model performance on
this table against the naive one isolates the effect of that one
variable.

### Reasoning:
- Holding the join structure, feature list, model, and CV procedure
  identical between the two runs means any performance gap can only come
  from the rows excluded by the temporal filter — a clean, single-variable
  test rather than a confounded before/after comparison.
- This experiment turned out to be a null test in the current data: both
  filter conditions already hold for 100% of raw bureau/previous_application
  rows (verified directly — 0 rows excluded by either filter), so
  `features_naive.parquet` and `features_filtered.parquet` are
  value-identical up to ~1e-16 floating-point noise from DuckDB's
  parallel aggregation (confirmed by rebuilding the naive pipeline twice
  and diffing it against itself, which showed the same magnitude of
  noise). The CV results are consequently identical to reported precision.
  This is documented as an inconclusive result, not a clean bill of
  health — see `reports/leakage_report.md` Section 4.
- `DAYS_CREDIT_UPDATE` (bureau's last-refresh offset, distinct from
  `DAYS_CREDIT`'s origination offset) was deliberately *not* used as a
  filter: only 17 of 1,716,428 rows (0.001%) have a positive value, too
  few to move any aggregate, so it's reported
  (`reports/days_credit_update_analysis.json`) rather than filtered on
  without justification.

### Alternatives considered:
- **Only report DAYS_CREDIT/DAYS_DECISION ranges, skip building a second
  parquet file**: rejected — a written-down range check doesn't reveal
  whether a violation would actually change *model* performance, which is
  the thing that matters for trustworthiness.

### Impact:
The audit conclusively rules out one specific leakage vector (future-
dated bureau/previous-application records) and explicitly flags what it
does not rule out (bureau snapshot staleness beyond `DAYS_CREDIT_UPDATE`,
`EXT_SOURCE_*` external scores) as needing a different test.

---

## Decision: Holdout remains untouched throughout the leakage audit

### Context:
The project's standing rule is "never use holdout data for decisions."
Phase 4 is exactly the kind of exploratory, judgment-forming work that
rule exists to protect against.

### Choice:
`src/leakage_experiment.py` reads only `data/processed/dev.parquet` (for
the naive arm) and `features_filtered.parquet` restricted to
`dev.parquet`'s `SK_ID_CURR` set (for the filtered arm).
`holdout.parquet` is never opened by any file created or modified in this
phase.

### Reasoning:
- A leakage audit's entire purpose is to inform a judgment call about
  the feature pipeline. If that judgment were formed using holdout data,
  the holdout would no longer be a valid, untouched estimate of
  real-world performance for whatever model is eventually built — the
  same failure mode the Phase 3 holdout lock was created to prevent in
  the first place.
  ([the Phase 3 holdout decision](#decision-create-the-devholdout-split-before-any-modeling-and-lock-it))
- Restricting the filtered arm to `dev.parquet`'s applicant IDs (rather
  than all of `features_filtered.parquet`) keeps both arms of the
  comparison on the exact same rows, which is necessary for the
  comparison to be valid — and, as a side effect, guarantees the filtered
  arm never touches a holdout applicant either.

### Alternatives considered:
- **Evaluate both arms on holdout for a "final" leakage check**: rejected
  — a leakage audit is inherently exploratory (multiple metrics, multiple
  candidate filters could be tried), which is precisely the kind of
  repeated peeking the holdout must be protected from.

### Impact:
`holdout.parquet` remains valid for a true final evaluation later.
Any future leakage-audit iteration (e.g. testing the `DAYS_CREDIT_UPDATE`
population specifically) must continue to use `dev.parquet` only.

---

## Decision: No additional temporal filtering required before baseline modeling

### Context:
`reports/leakage_report.md` (Phase 4) left one item explicitly open:
whether bureau's status-snapshot fields (`CREDIT_DAY_OVERDUE`,
`AMT_CREDIT_SUM_DEBT`, `CREDIT_ACTIVE`) could reflect information from
after the application decision even on rows where `DAYS_CREDIT <= 0`
holds. `reports/bureau_temporal_analysis.md` was produced specifically to
answer this before Phase 5 modeling begins.

### Choice:
Based on that analysis, no further temporal filter is added to
`build_bureau_aggregates()`. The existing `DAYS_CREDIT <= 0`
(`time_filtered=True`) option from Phase 4 remains as-is; bureau's
snapshot fields are treated as safe to use for Phase 5 baseline modeling
without further temporal restriction.

### Reasoning:
- `CREDIT_DAY_OVERDUE` and `AMT_CREDIT_SUM_DEBT` are refreshed as of
  `DAYS_CREDIT_UPDATE`, and 99.999% of bureau rows have
  `DAYS_CREDIT_UPDATE <= 0` — meaning their values were genuinely known as
  of the application decision for essentially the entire dataset. Only 17
  of 1,716,428 rows (0.000990%) violate this, too few to distort any
  aggregate or justify a new filter.
- Two data-quality observations surfaced by the same analysis (undercounted
  `Sold`/`Bad debt` loan statuses in `bureau_active_loans`/
  `bureau_closed_loans`; 8,418 unexplained negative `AMT_CREDIT_SUM_DEBT`
  values) are feature-quality concerns, not temporal-leakage concerns —
  conflating the two would misclassify a data-quality fix as a leakage fix
  and risks the wrong remediation being applied.

### Alternatives considered:
- **Add a `DAYS_CREDIT_UPDATE <= 0` filter alongside the existing
  `DAYS_CREDIT <= 0` filter**: rejected for now — would exclude only 17
  rows dataset-wide, with no measurable effect on any aggregate, so it
  adds pipeline complexity without a corresponding reliability benefit.
  Revisit if a future, larger bureau extract shows a materially different
  `DAYS_CREDIT_UPDATE` distribution.

### Impact:
Closes the temporal-leakage side of the Phase 4 audit. `EXT_SOURCE_*`
leakage and the two data-quality observations above remain open items for
a future phase, separate from this decision.

---

## Decision: Logistic regression as the Phase 5 baseline model

### Context:
Phase 5 needs a first trained model to serve as a performance floor that
later, more complex models (e.g. the XGBoost used only for the Phase 4
leakage audit) are measured against.

### Choice:
`src/train_baseline.py` trains a single, un-tuned
`LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)`
via stratified 5-fold CV on `dev.parquet`, with no hyperparameter search.

### Reasoning:
- A linear model's coefficients are directly interpretable in sign and
  relative magnitude, which matters for a project whose stated goal is a
  *trustworthy*, explainable decision system — a baseline should be
  something a reviewer can reason about without SHAP before more complex,
  less directly-interpretable models are introduced.
- `class_weight="balanced"` addresses the ~8% default rate without
  resampling the data or hand-picking a class weight, keeping the
  baseline a genuine off-the-shelf reference point rather than an already-
  tuned result.
- Explicitly *not* tuning hyperparameters here is intentional: a tuned
  baseline would blur the line between "how good is a simple model
  before any effort" and "how good is a simple model after effort,"
  making it a less honest floor for later comparisons.

### Alternatives considered:
- **Start with a tree-based model as the baseline**: rejected — a tree
  ensemble was already used in Phase 4 for the leakage audit specifically
  *because* it doesn't need scaling/encoding; using it again as "the
  baseline" would conflate the leakage-audit model with the first real
  benchmarking result.

### Impact:
Every subsequent model in this project should be compared against this
result (`reports/baseline_results.json`) before being considered an
improvement, not against an arbitrary absolute threshold.

---

## Decision: Preprocessing lives inside a single sklearn Pipeline

### Context:
`dev.parquet` has missing values (per Phase 3) and both numeric and
categorical (string / nullable-boolean) columns. Preprocessing statistics
— imputation medians, scaling mean/variance, and known one-hot categories
— must come from somewhere, and cross-validation makes "somewhere" a real
question: computed once on all of `dev.parquet`, or per fold?

### Choice:
`SimpleImputer`, `StandardScaler`, and `OneHotEncoder` are composed via a
`ColumnTransformer` and wrapped together with `LogisticRegression` in one
`sklearn.pipeline.Pipeline`, which is what gets passed to
`cross_validate()` — not fit once beforehand on the full dataset.

### Reasoning:
- If the median/scaler/category-set were fit on all of `dev.parquet`
  before splitting into folds, every fold's "held-out" validation rows
  would have quietly influenced the very preprocessing statistics used to
  transform them — a classic, easy-to-miss form of leakage that inflates
  CV scores without any obviously wrong step in the code.
  `cross_validate(pipeline, X, y, cv=...)` instead refits the entire
  Pipeline — imputer, scaler, encoder, and model — from scratch on each
  fold's training split alone.
- Column classification is done by dtype (`number` → numeric branch,
  everything else → categorical branch) rather than an explicit
  bool/object allow-list, since `EMERGENCYSTATE_MODE` loads from Parquet
  as pandas' nullable `boolean` dtype (not numpy `bool`) — an allow-list
  keyed on `["number", "bool"]` would have silently misrouted it into
  `SimpleImputer`+`StandardScaler`, untested against that dtype; the
  complement-based split was verified to route it to `OneHotEncoder`
  instead, which was directly tested against its True/False/NA values.
- No imputer is used ahead of `OneHotEncoder` (per the task's spec) —
  verified directly (see code comment in `build_pipeline`) that sklearn
  1.9's `OneHotEncoder` treats a missing categorical value as its own
  category rather than erroring, so this omission is a verified fact
  about the library, not an unchecked assumption.

### Alternatives considered:
- **Fit preprocessing once on all of `dev.parquet`, then cross-validate
  only the model**: rejected — exactly the leakage pattern this decision
  exists to avoid.

### Impact:
Any future model added to this project should follow the same pattern —
preprocessing and model fit together inside one Pipeline passed to
`cross_validate`, never fit on the full dataset ahead of splitting.

---

## Decision: PR-AUC as the primary metric, ROC-AUC reported alongside

### Context:
The dev set's default rate is ~8% (confirmed in Phase 3) — a substantially
imbalanced target. ROC-AUC and PR-AUC can disagree in how favorably they
represent a model under class imbalance.

### Choice:
Both metrics are computed and saved for every experiment
(`reports/baseline_results.json`, and earlier `reports/leakage_results.json`),
but PR-AUC (`average_precision`) is treated as the primary metric for
comparing models going forward; ROC-AUC is reported for context and
comparability with the Phase 4 leakage-audit numbers.

### Reasoning:
- ROC-AUC is computed against the true-negative rate, and with ~92% of
  applicants being non-defaulters, a model can achieve a deceptively high
  ROC-AUC while still performing poorly at identifying the minority
  default class specifically — which is the class this system actually
  needs to act on (a credit decision).
- PR-AUC is sensitive to exactly this: it's computed from precision and
  recall on the positive (default) class only, so it more directly
  reflects how useful the model's default-risk ranking would be for the
  business decision this whole project exists to support.
- Reporting ROC-AUC alongside (not instead of) PR-AUC keeps the two
  audit/benchmark artifacts (`leakage_results.json`,
  `baseline_results.json`) comparable to each other on both metrics.

### Alternatives considered:
- **ROC-AUC only**: rejected — for an ~8% base rate, it is the more
  optimistic and less business-relevant of the two metrics; using it
  alone risks a false sense of confidence in a model that ranks the
  minority class poorly.

### Impact:
Model selection and "is this an improvement" judgments in later phases
should be argued primarily from PR-AUC movement, with ROC-AUC as
supporting context — not the reverse.

---

## Decision: XGBoost as the nonlinear candidate model

### Context:
Phase 6 needs a nonlinear model to test whether the logistic baseline's
linear decision boundary is leaving real, learnable signal on the table —
credit risk features commonly interact (e.g. income-to-credit ratio
matters differently across income brackets) in ways a linear model can't
represent without manual feature crosses.

### Choice:
`src/train_xgboost.py` trains a single `XGBClassifier` with the exact
8-parameter configuration specified (`n_estimators=300, learning_rate=0.1,
max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
eval_metric="aucpr", random_state=42`), plus `enable_categorical=True`.

### Reasoning:
- Gradient-boosted trees capture nonlinearities and feature interactions
  natively, without requiring the modeler to hand-specify which
  interactions to test — a natural next step after a linear baseline.
- `tree_method="hist"` natively handles missing values (consistent with
  Phase 3's decision not to impute) and, combined with
  `enable_categorical=True`, natively handles categorical splits — so all
  146 raw features are used unmodified, identical to what the logistic
  baseline saw (via one-hot encoding instead). This keeps the model
  comparison about the *model*, not about one model seeing more
  information than the other.
- `enable_categorical=True` required one data-fix: `EMERGENCYSTATE_MODE`
  loads as pandas' nullable `boolean` dtype, which XGBoost's categorical
  encoder rejects outright (verified directly — it requires string or int
  category values). Mapping `True`/`False` to strings before casting to
  `category` (via `.map()`, which leaves genuine nulls as NaN rather than
  a literal `"<NA>"` category) resolved this without dropping the column
  or changing any model hyperparameter.
- This is the same untuned model configuration already used for the
  Phase 4 leakage audit, reused here as a benchmarking candidate — not
  re-derived — keeping one canonical "reference nonlinear model" in the
  project rather than two subtly different ones.

### Alternatives considered:
- **Drop the 16 categorical columns and train on numeric features only**
  (as Phase 4's leakage-audit script did): rejected for this comparison
  specifically — it would understate XGBoost's potential relative to the
  baseline, which had access to those columns via one-hot encoding.

### Impact:
`reports/xgboost_results.json` and `reports/model_comparison.json` now
reflect a same-information comparison between the two model families.
Any future model added to this project should also use the full 146-
feature set unless there's a specific reason to restrict it.

> **Superseded note (Phase 7.8):** the feature count above (146) was
> accurate as of Phase 6. Phase 7.5 added `bureau_had_negative_debt`,
> bringing the count to 147 — the current, authoritative figure is in
> `reports/final_model_card.md`. The guidance to use the full feature set
> still applies; just read "147" wherever "146" appears above and below.

---

## Decision: Hyperparameter tuning deferred past Phase 6

### Context:
`XGBClassifier`'s configuration (n_estimators, learning_rate, max_depth,
subsample, colsample_bytree) has many plausible values; none were swept
in this phase.

### Choice:
The exact hyperparameter values specified for this phase are used as
given, with no grid/random search, Bayesian optimization, or manual
tuning performed anywhere in `src/train_xgboost.py`.

### Reasoning:
- Phase 6's purpose is to answer one question — "does a nonlinear model
  meaningfully outperform the linear baseline at all" — which an untuned
  model answers more honestly than a tuned one: a tuned XGBoost beating an
  untuned logistic regression conflates "nonlinearity helps" with "more
  optimization effort helps," making the comparison uninterpretable.
- Any tuning search implicitly requires a validation signal to optimize
  against. Performing that search now, before this project has decided
  which of several models it's committing to, risks either overfitting to
  the dev-set CV folds or creating pressure to peek at holdout — the
  project's standing rule against exactly that
  ([[decision-create-the-devholdout-split-before-any-modeling-and-lock-it]]).
- Deferring tuning until a model family is actually selected also avoids
  wasted tuning effort on a candidate that might not even be chosen.

### Alternatives considered:
- **Light tuning now, to see XGBoost's "real" ceiling**: rejected — scope
  creep for a phase whose explicit goal is comparison, not optimization;
  the task instructions for this phase state this directly.

### Impact:
The Phase 6 result (PR-AUC 0.2527 vs. baseline's 0.2334) should be read as
"a nonlinear model, untuned, already outperforms the untuned linear
baseline" — a lower bound on what XGBoost can do, not its ceiling. A
future tuning phase should be scoped explicitly once a model family is
chosen to move forward with.

> **Superseded note (Phase 7.8):** 0.2527 was XGBoost's PR-AUC as measured
> in Phase 6, before the Phase 7.5 pipeline fixes (negative-debt clipping,
> split determinism). The refreshed, current figure is PR-AUC 0.2499 ±
> 0.0087 (ROC-AUC 0.7634 ± 0.0044) — see Phase 7.6's decision below and
> `reports/final_model_card.md`. The qualitative conclusion ("untuned
> nonlinear beats untuned linear") is unchanged.

---

## Decision: Every candidate model is compared against the baseline, not evaluated in isolation

### Context:
`reports/xgboost_results.json` alone reports XGBoost's CV metrics but says
nothing about whether that's actually *better* than the alternative this
project already has.

### Choice:
`build_comparison()` in `src/train_xgboost.py` always reads
`reports/baseline_results.json` and writes `reports/model_comparison.json`
— a required side-by-side artifact, not an optional one — hard-failing
with a clear error if the baseline results don't exist yet rather than
silently comparing against nothing.

### Reasoning:
- A model's absolute PR-AUC/ROC-AUC number is not, by itself,
  interpretable as "good" or "bad" for this dataset — it only becomes
  meaningful relative to a known reference point, which is exactly what
  the Phase 5 baseline exists to be.
- Making the comparison an artifact (JSON on disk), not just a printed
  number, means the delta is available to any later phase or reviewer
  without re-running both experiments.
- Failing loudly if `baseline_results.json` is missing (rather than
  skipping the comparison) prevents this phase from silently shipping an
  incomplete evaluation.

### Alternatives considered:
- **Report XGBoost's metrics standalone, compare manually later**:
  rejected — defers a comparison that's cheap to automate now and easy to
  forget to do consistently later.

### Impact:
Every future candidate model in this project should follow the same
pattern: write its own `*_results.json`, then extend or regenerate
`model_comparison.json` against the current best-known baseline, so the
comparison history accumulates rather than requiring manual reconciliation.

---

## Decision: Probability calibration is required before this system makes decisions

### Context:
Phase 6 established that XGBoost *ranks* applicants by risk better than
the logistic baseline (higher ROC-AUC/PR-AUC). Ranking quality alone does
not mean the model's output can be read as an actual default probability.

### Choice:
Phase 7 explicitly separates "does the model rank well" (already answered)
from "can the model's raw score be trusted as a probability" (this
phase's question), via `src/calibrate_model.py`, before any business
decision logic is built on top of it.

### Reasoning:
- This system's entire purpose is to convert a probability into a
  business decision (a future phase). That conversion needs the number
  itself to be meaningful — e.g. "this applicant has an 8% chance of
  default" must actually mean roughly 8 in 100 similar applicants
  default, not just "this applicant ranks higher-risk than that one."
  ROC-AUC/PR-AUC are rank-based and provably invariant to any monotonic
  transformation of the score (confirmed empirically below), so they
  cannot detect whether the raw score is off by a consistent multiplicative
  or additive factor.
- Gradient-boosted trees are commonly, though not universally,
  under/over-confident at the extremes even when their ranking is strong,
  because the training objective optimizes ranking-adjacent loss, not
  calibration directly.
- In this project's case, the raw XGBoost score turned out to already be
  reasonably close to calibrated (Brier 0.06745, visually near the
  diagonal in `reports/calibration_curve.png`) — but this was verified,
  not assumed, and isotonic regression still improved on it (Brier
  0.06725). Skipping this phase would have meant shipping an assumption
  instead of a measurement.

### Alternatives considered:
- **Skip calibration and use the raw XGBoost score directly for business
  decisions**: rejected — even a small calibration gap compounds when the
  probability feeds directly into an expected-loss calculation, which is
  exactly what a future phase will do with this number.

### Impact:
Later phases (threshold/decision logic) should read probabilities from
the calibrated model (isotonic — see `reports/calibration_results.json`),
not the raw XGBoost score, unless a documented reason says otherwise.

---

## Decision: Calibration is fit and evaluated on out-of-fold predictions

### Context:
Calibration needs a training signal (raw score → observed outcome) to
learn from. Fitting XGBoost once on all of `dev.parquet` and calibrating
against its own training predictions was one option; generating
out-of-fold (OOF) predictions via 5-fold CV was the other.

### Choice:
`generate_oof_predictions()` retrains Phase 6's exact XGBoost
configuration once per fold and predicts only on that fold's held-out
rows, so every row in `reports/oof_predictions.csv` was scored by a model
that never saw it during training. Platt and isotonic are then fit on
these OOF scores, not on in-sample predictions from a single
fit-on-everything model.

### Reasoning:
- A model's predictions on its own training data are systematically
  overconfident (it has partially memorized those rows), which would make
  any calibration fit against them learn the wrong correction — a
  calibrator trained to fix overconfidence that isn't actually present in
  genuine unseen-data behavior.
- OOF predictions are the same tool this project already trusts CV
  metrics to come from (Phases 5-6) — using anything else here would mean
  Phase 7 measures calibration against a different, less honest notion of
  "unseen data" than the rest of the project uses for performance.
- Reusing Phase 6's `MODEL_PARAMS`/`prepare_categoricals` unmodified (per
  this phase's rule) keeps the OOF scores representative of the exact
  model already benchmarked, rather than a subtly different one.

### Alternatives considered:
- **Calibrate on the same model's in-sample training predictions**:
  rejected — would understate the true calibration gap by fitting against
  optimistic, overconfident scores.

### Impact:
`reports/oof_predictions.csv` (SK_ID_CURR, TARGET, xgb_probability) is now
the canonical honest-score artifact for this model; any future
recalibration attempt should regenerate or reuse this file rather than
scoring `dev.parquet` with a single fit-on-everything model.

---

## Decision: Brier score reported alongside ROC-AUC/PR-AUC for calibration

### Context:
Step 3 needed a way to actually tell the three methods (raw, Platt,
isotonic) apart. Empirically here, ROC-AUC and PR-AUC came back
effectively identical for raw (0.7664 / 0.2524) and Platt (0.7664 /
0.2524) — expected, since Platt scaling is a strictly monotonic
transform of a single input score and cannot change how it ranks
applicants relative to each other. Isotonic showed only a marginal
difference (0.7669 / 0.2479), attributable to tie-breaking from its
step-function output rather than a real ranking change.

> **Superseded note (Phase 7.8):** these dev-OOF figures were measured in
> Phase 7, before the Phase 7.5 pipeline fixes. The pattern they illustrate
> (ROC-AUC/PR-AUC ~flat across methods; Brier score is what differentiates
> them) still holds on the corrected pipeline and on holdout — see
> `reports/calibration_results.json` (refreshed) and
> `reports/holdout_calibration_results.json` for current numbers.

### Choice:
Brier score (`brier_score_loss`) is computed and reported for every
method, alongside ROC-AUC/PR-AUC, in `reports/calibration_results.json`.

### Reasoning:
- Brier score is the mean squared error between predicted probability and
  actual outcome — it is sensitive to the actual magnitude of the
  predicted probability, not just its rank, which is precisely the
  property ROC-AUC/PR-AUC lack for this phase's purpose.
- This wasn't a hypothetical concern: it's the metric that revealed Platt
  scaling actually made calibration slightly *worse* than the raw score in
  this experiment (Brier 0.06850 vs. raw 0.06745), a finding ROC-AUC/PR-AUC
  were structurally incapable of surfacing since they came back identical.
  Isotonic was the only method that improved on raw (Brier 0.06725).

### Alternatives considered:
- **Rely on the calibration curve plot alone**: rejected as the sole
  measure — a visual reliability diagram is useful context (kept, as
  `reports/calibration_curve.png`) but doesn't give a single comparable
  number for ranking the three methods against each other the way Brier
  score does.

### Impact:
Method selection for calibration in this project is driven by Brier
score, not ROC-AUC/PR-AUC (which are expected, by construction, to stay
roughly flat across recalibration methods). Isotonic is the current best
result on this basis and is the calibrated-probability source for future
phases per the decision above.

---

## Decision: Negative bureau debt is clipped to 0, with a preserved indicator

### Context:
`reports/bureau_temporal_analysis.md` (Phase 4 follow-up) flagged 8,418
negative `AMT_CREDIT_SUM_DEBT` values (0.49% of bureau rows) as an
unexplained data-quality anomaly, deferred at the time rather than fixed.
Re-verified here: 8,418 rows, affecting 5,886 of 305,811 bureau-covered
applicants (1.92%), concentrated in `Active` loans (6,222 of 8,418), range
-0.045 to -4,705,600.32.

### Choice:
`build_bureau_aggregates()` in `src/features.py` now clips negative
`AMT_CREDIT_SUM_DEBT` values to 0 before summing into `bureau_total_debt`
(via `CASE WHEN AMT_CREDIT_SUM_DEBT < 0 THEN 0 ELSE ... END`, which leaves
genuine NULLs as NULL — only actual negative numbers are touched), and
adds `bureau_had_negative_debt` (1 if any of an applicant's bureau records
had negative debt, else 0; NULL only for applicants with no bureau
history at all, matching every other `bureau_*` column's missingness
convention).

### Reasoning:
- A negative outstanding debt has no documented meaning in the Home
  Credit data dictionary, and the extreme tail (-4.7M) is far too large to
  be plausibly explained as a simple overpayment/credit-balance case.
  Left unclipped, `SUM(AMT_CREDIT_SUM_DEBT)` and `debt_income_ratio` would
  be silently pulled downward by an unexplained anomaly for the 1.92% of
  applicants affected.
- Clipping to 0 (rather than, say, taking absolute value) is the more
  conservative correction: it assumes "no evidence of negative debt below
  zero" rather than actively asserting a specific alternate value, which
  would be fabricating information the data doesn't support.
- `bureau_had_negative_debt` exists because clipping is a lossy operation
  — without a flag, the fact that a correction happened would disappear
  entirely from the feature set, even though it may itself carry a
  correlation-with-something-else signal (e.g. a particular reporting
  source's data quality) independent of the corrected magnitude.

### Alternatives considered:
- **Leave negative debt as-is (status quo from Phase 4)**: rejected — a
  "Model Reliability Cleanup" phase is exactly the point at which a known,
  previously-deferred anomaly should be resolved rather than carried
  forward again.
- **Drop rows with negative debt**: rejected — would violate applicant
  grain (one row per `SK_ID_CURR` is required) and discard the rest of an
  otherwise-valid applicant's data over one anomalous bureau record.

### Impact:
`features_naive.parquet`/`features_filtered.parquet` now have 26
engineered features (was 25). `bureau_total_debt` and `debt_income_ratio`
values changed for the 5,886 affected applicants; every other engineered
feature is untouched. This required rebuilding `dev.parquet`/
`holdout.parquet` — see the split-stability fix below, which was
discovered during that rebuild.

---

## Decision: Dev/holdout split must sort by SK_ID_CURR before splitting

### Context:
Rebuilding `features_naive.parquet` for the negative-debt fix, then
rerunning `src/data_quality.py` unchanged, was expected to reproduce the
same dev/holdout partition (same `random_state=42`, same applicant
universe). Instead, 49,151 of 61,503 holdout applicants (80%) turned out
to be *different* from the previous holdout — even though the same
`SK_ID_CURR`'s `TARGET` value never changed (0 label mismatches, verified
directly). A back-to-back rerun of `data_quality.py` against a single,
unchanged `features_naive.parquet` reproduced its own split exactly (0
diff), isolating the cause to something that changes when the *upstream*
Parquet file is rewritten.

### Choice:
`create_dev_holdout_split()` now runs
`SELECT SK_ID_CURR, TARGET FROM features_clean ORDER BY SK_ID_CURR`
before calling `train_test_split`, instead of an unordered `SELECT`.

### Reasoning:
- `train_test_split`'s shuffle (even with a fixed `random_state`) operates
  on row *position* in its input array, not on the `SK_ID_CURR` values
  themselves. DuckDB does not guarantee a stable row-materialization order
  for the same query across separate process runs — confirmed directly:
  rebuilding `features_naive.parquet` after an unrelated column addition
  changed the order `COPY ... TO parquet` happened to write rows in, which
  silently fed `train_test_split` a differently-ordered input and produced
  a different partition, despite every "reproducibility" input
  (`random_state`, the SQL query text, the applicant universe) being
  unchanged.
  - Verified the fix: rebuilt `features_naive.parquet` a second time
    (forcing another row-order change) and reran the now-`ORDER BY`'d
    split — 0 holdout membership difference from the run immediately
    before that rebuild.
- This means the *original* Phase 3 dev/holdout split was never actually
  a stable function of applicant identity — it was, unknowingly, a
  function of an incidental Parquet-writing detail. Any future rebuild of
  the upstream feature pipeline (for any reason, not just this one) could
  have silently reshuffled which applicants are in holdout, without
  changing a single line of `data_quality.py` or any label.
- This is precisely the kind of bug a project claiming to keep holdout
  "locked" and "untouched" cannot afford to have unexamined — the whole
  point of a holdout is that it names a fixed, known set of applicants
  over time; a set that silently changes on unrelated upstream rebuilds
  defeats that purpose even if no single run looks wrong in isolation.

### Alternatives considered:
- **Sort by row content other than SK_ID_CURR (e.g. a hash)**: rejected —
  `SK_ID_CURR` is already the canonical, immutable applicant identifier;
  sorting by it is simpler and self-explanatory.
- **Leave as-is, treat this rebuild's new split as the "real" one going
  forward**: rejected — doesn't fix the underlying fragility, only
  papers over this one instance of it.

### Impact:
`dev.parquet`/`holdout.parquet` membership changed as a side effect of
this fix (in addition to the negative-debt feature correction) — this is
a new, now-actually-stable partition, not the same one as before Phase
7.5. Every downstream artifact evaluated against the old holdout
(`baseline_results.json`, `xgboost_results.json`, `model_comparison.json`,
Phase 7's original `calibration_results.json`) reflects the old,
no-longer-current split and should be treated as superseded, not
authoritative, until rerun (see Risks in the Phase 7.5 completion summary).

---

## Decision: Isotonic regression selected as the calibrated-probability source, validated on holdout

### Context:
Phase 7 selected isotonic regression based on dev-set out-of-fold (OOF)
metrics alone. `src/evaluate_holdout_calibration.py` re-evaluates all
three methods (raw XGBoost, Platt, isotonic) on `holdout.parquet` — read
only for scoring, never for fitting anything — to check whether that
selection actually holds on data the pipeline has never touched.

### Choice:
Isotonic regression remains the selected calibration method: holdout
Brier score 0.066826 vs. raw 0.066840 vs. Platt 0.067816. Platt scaling is
explicitly rejected for use in the decision engine.

### Reasoning:
- The isotonic-vs-raw advantage that looked clear on dev OOF (0.06742 vs.
  0.06762, a 0.00020 gap) nearly disappeared on holdout (0.000014 gap).
  This is reported honestly as a real finding, not smoothed over: it means
  Phase 7's original conclusion was more confident than the evidence
  actually supported, and the raw XGBoost score was already close to
  calibrated on this dataset. Isotonic is still selected because it is
  never worse than raw on either evaluation, not because it dramatically
  outperforms it.
- Platt scaling underperforming raw XGBoost is the one pattern that *does*
  replicate: worse Brier score on both dev OOF (0.06860 vs. 0.06762) and
  holdout (0.06782 vs. 0.06684), by a similar margin each time. A
  regression that shows up consistently across two independent
  evaluations is trustworthy in a way a single-evaluation result is not;
  Platt scaling is excluded from the decision engine on this basis.
- ROC-AUC/PR-AUC stayed effectively flat across all three methods on
  holdout too (0.7713/0.2650 for raw and Platt, 0.7713/0.2585 for
  isotonic) — reconfirming, on new data, that ranking metrics can't
  distinguish these methods and Brier score is the metric that matters
  for this decision.

### Alternatives considered:
- **Keep Phase 7's dev-OOF-only conclusion, skip holdout validation**:
  rejected — this is precisely the gap Phase 7.5 exists to close; a
  calibration choice that only looks good on the same sample it was
  fit/compared on is not yet a validated choice.

### Impact:
**Isotonic-calibrated XGBoost probabilities are the final probability
source for the decision engine.** Any future phase building business
decision logic (thresholds, expected-loss calculations) on top of a
default probability should read it from the isotonic calibrator applied
to the full-dev-trained XGBoost model — the exact artifacts produced by
`src/evaluate_holdout_calibration.py` — not from the raw XGBoost score,
and not from Platt scaling.

> **Superseded (Phase 7.8):** the "never worse than raw" observation above
> was correct but incomplete — it didn't test whether isotonic's holdout
> edge (0.066826 vs. 0.066840) was distinguishable from noise. A
> 1,000-iteration bootstrap found the 95% CI for the difference spans 0,
> i.e. it isn't. **Raw XGBoost, uncalibrated, is now the final probability
> source** — see the two Phase 7.8 decisions below. Platt scaling's
> exclusion is unaffected by this reversal.

---

## Decision: Baseline/XGBoost benchmarks refreshed post-Phase-7.5, superseding prior numbers

### Context:
`reports/baseline_results.json`, `reports/xgboost_results.json`, and
`reports/model_comparison.json` were flagged as stale in Phase 7.5's Risks
section — they were computed against the pre-fix `dev.parquet` (negative
bureau debt unclipped, and a dev/holdout split that wasn't actually a
stable function of applicant identity — see the two Phase 7.5 decisions
above). Phase 7.6 reruns `src/train_baseline.py` and `src/train_xgboost.py`
unchanged against the corrected `dev.parquet` to close that gap.

### Choice:
All three files are regenerated with no code or hyperparameter changes to
either training script. New results: logistic ROC-AUC 0.7543 / PR-AUC
0.2301; XGBoost ROC-AUC 0.7634 / PR-AUC 0.2499; XGBoost still leads on
both metrics (+0.0091 ROC-AUC, +0.0198 PR-AUC).

### Reasoning:
- These are benchmark numbers this project's future model-selection
  decisions are meant to be measured against
  ([[decision-every-candidate-model-is-compared-against-the-baseline-not-evaluated-in-isolation]]).
  Leaving them computed against a since-corrected, since-reshuffled
  dataset would make every future "is this an improvement" comparison
  reference the wrong ground truth.
- No model or preprocessing logic changed — only the input data
  (`dev.parquet`) did, via Phase 7.5's fixes. Rerunning the exact same
  scripts is sufficient; no new engineering decision was required here
  beyond acknowledging which numbers are now authoritative.

### Alternatives considered:
- **Keep the old numbers, note the discrepancy in prose only**: rejected
  — the files are the artifacts other phases and reviewers actually read;
  leaving them stale invites exactly the kind of silent staleness this
  project's phased, documented approach exists to avoid.

### Impact:
The pre-Phase-7.5 versions of these three files are superseded and should
not be cited going forward. Any phase referencing "the baseline" or "the
XGBoost benchmark" from this point on means these Phase 7.6 numbers.

---

## Decision: Model pipeline frozen and captured in reports/final_model_card.md

### Context:
By Phase 7.6, every stage — feature engineering (with the negative-debt
fix), the dev/holdout split (with the determinism fix), both benchmarked
models, and holdout-validated calibration — had been independently
verified. Phase 7.7 re-verified all five consistency properties one more
time (feature count, split determinism, XGBoost parameters, calibrator
fitting scope, holdout-usage scope) with no code changes, and all five
held.

### Choice:
The pipeline as it exists at this point — 147 features, the
`SK_ID_CURR`-ordered deterministic 80/20 split, `XGBClassifier` with the
Phase 6 hyperparameters, isotonic-calibrated probabilities from a
full-dev-trained model — is declared frozen and documented in
`reports/final_model_card.md`.

### Reasoning:
- A "final model card" is only meaningful if it describes something that
  isn't still shifting under it. Freezing gives every future phase
  (SHAP explanations, business-decision thresholds, Streamlit deployment)
  a fixed, named reference to build on, rather than an implicit
  assumption that "the current pipeline" means whatever it happens to be
  when that phase starts.
- Re-verifying all five properties immediately before freezing (rather
  than trusting Phase 7.5/7.6's prior verifications alone) is what makes
  this freeze trustworthy rather than aspirational — each check was
  re-run against the actual current code and data, not recalled from
  memory of having checked before.

### Alternatives considered:
- **No formal freeze, treat the pipeline as "done for now" informally**:
  rejected — this project's whole approach has been to make decisions
  explicit and re-checkable; an informal, undocumented freeze point would
  break that pattern right before the phases that most need a stable
  foundation (explainability, business decisions) begin.

### Impact:
Any future change to feature engineering, the split, model
hyperparameters, or the calibration method is now a deviation from a
named, documented baseline (`reports/final_model_card.md`) and should be
justified and re-recorded in DECISIONS.md as such, not made silently.

---

## Decision: Bootstrap significance test added before finalizing calibration choice

### Context:
Phase 7.5 selected isotonic regression over raw XGBoost based on a single
holdout Brier score comparison (0.066826 vs. 0.066840 — a 0.000014 gap).
A single point estimate cannot distinguish a real effect from sampling
noise at that magnitude.

### Choice:
`src/bootstrap_calibration_comparison.py` reproduces the frozen holdout
predictions (same `MODEL_PARAMS`, same dev-fit final model, same dev-OOF-
fit isotonic calibrator — verified byte-identical to Phase 7.5's saved
`brier_score` values before proceeding, diff = 0.00e+00 for both methods)
and runs a 1,000-iteration paired bootstrap (`random_state=42`) over
Brier(raw) − Brier(isotonic), reporting the mean difference and a 95% CI.

### Reasoning:
- Paired resampling (the same bootstrap row indices applied to both raw
  and isotonic each iteration) isolates the *difference* between the two
  methods on identical resampled applicants, rather than treating two
  independently-resampled Brier scores as if their difference were
  itself directly comparable — the right design for "is A better than B,"
  as opposed to "what is A's Brier score."
- Verifying the reproduced predictions against Phase 7.5's saved results
  before bootstrapping was necessary, not optional: this project's
  earlier discovery of unstable row ordering (the split-determinism bug)
  is exactly the kind of failure mode that would silently corrupt a
  bootstrap built on freshly reproduced data — checking first turns
  "assume reproducibility" into "confirm reproducibility."
- 1,000 iterations at a fixed seed makes the CI itself reproducible run to
  run, consistent with every other stochastic step in this project.

### Alternatives considered:
- **Keep the Phase 7.5 point-estimate decision as final**: rejected — a
  0.000014 Brier gap on 61,503 holdout rows is exactly the kind of result
  that demands a significance check before being used to justify adding a
  calibration layer to production.

### Impact:
`reports/calibration_bootstrap_results.json` is now the evidentiary basis
for the calibration decision below, superseding Phase 7.5's point-estimate
justification for isotonic.

---

## Decision: Raw XGBoost selected as the final production probability, superseding isotonic

### Context:
The Phase 7.8 bootstrap (above) found the Brier(raw) − Brier(isotonic)
95% CI to be **[-6.81e-05, +1.03e-04]** — an interval spanning 0. The
isotonic advantage Phase 7.5 selected on is not statistically
distinguishable from noise at this sample size.

### Choice:
**Raw, uncalibrated XGBoost probabilities are the final production
probability source**, reversing Phase 7.5's selection of isotonic
regression. Platt scaling remains excluded (unaffected by this reversal —
see below).

### Reasoning:
- When two options perform statistically indistinguishably, the simpler
  one should win: raw XGBoost requires no additional fitted object (no
  isotonic calibrator to version, validate, or explain alongside the
  model), which matters directly for a system whose stated goal is
  trustworthiness and explainability, not marginal metric optimization.
  This is the same "don't add unnecessary complexity" principle this
  project has applied to code throughout, now applied to a modeling
  choice.
- This is not a rejection of calibration as a concept — it's a finding
  specific to this dataset/model: the raw XGBoost score here already
  turned out to be close to calibrated (established in Phase 7's
  reliability diagram), so there wasn't a real calibration gap for
  isotonic to close, and the bootstrap confirms that directly rather than
  leaving it as a visual impression.
- Platt scaling remains excluded independent of this reversal: it
  underperformed raw XGBoost in *both* the dev-OOF and holdout point-
  estimate evaluations, a pattern that replicated across two independent
  looks at the data even without a bootstrap test — a materially
  different evidentiary situation than the isotonic-vs-raw comparison,
  which only ever had one point estimate per evaluation and turned out to
  be within noise.

### Alternatives considered:
- **Keep isotonic anyway, since it's "never worse"**: rejected — "never
  worse, statistically indistinguishable" is not a reason to prefer the
  more complex option; it's precisely the condition under which the
  simpler option should be preferred.

### Impact:
`reports/final_model_card.md`'s "Calibration method" section now names
raw XGBoost as the production probability source. Any future phase
building business-decision logic (thresholds, expected-loss calculations,
SHAP explanations) on top of a default probability should use the raw
XGBoost score directly — not an isotonic-calibrated one.

---

## Decision: Phase 7.9 documentation-consistency freeze

### Context:
Phase 7.8 established raw XGBoost as the final probability source and
refreshed most current-facing documentation, but two artifacts were
missed: `reports/holdout_calibration_report.md`'s Section 3 still asserted
isotonic as the choice "going forward," and `reports/final_model_card.md`
mislabeled the bureau feature count (12, pre-negative-debt-fix) and
reintroduced the "146 engineered features" framing this project has
otherwise avoided. Phase 7.9 is a documentation-only pass to close these
gaps before Phase 8 begins.

### Choice:
No code, model, feature engineering, or split logic changed. Confirmed by
re-reading the code directly, not from memory: (1) the negative-bureau-
debt treatment in `src/features.py` — clip-to-zero on `bureau_total_debt`,
`bureau_had_negative_debt` indicator, NULL-history preservation — matches
the established Phase 7.5 decision exactly, no drift found. (2) Raw
XGBoost (uncalibrated) is confirmed as the final probability source;
`reports/holdout_calibration_report.md` Section 3 and
`reports/final_model_card.md` are corrected to state this without
deleting or rewriting the historical calibration analysis that led there.
(3) Current benchmark numbers (XGBoost ROC-AUC 0.7634 ± 0.0044 / PR-AUC
0.2499 ± 0.0087; Logistic ROC-AUC 0.7543 ± 0.0039 / PR-AUC 0.2301 ±
0.0076) remain as established in Phase 7.6 — this phase only confirmed
consistency, not new values.

### Reasoning:
- A report titled "Final ... Report" that still recommends a superseded
  choice is worse than no report at all — a reader landing on it directly
  (not via DECISIONS.md) would be actively misled, which is a different
  and more urgent problem than a decision log entry being chronologically
  ordered.
- The bureau-feature-count and "146 engineered" issues were introduced by
  this project's own prior documentation pass (Phase 7.8), not inherited
  from further back — worth noting plainly rather than glossing over,
  since this project's own standard is to document mistakes honestly.

### Alternatives considered:
- **Leave `holdout_calibration_report.md` as pure historical record,
  unedited**: rejected — unlike DECISIONS.md, it is not framed as a
  dated log entry; its title and structure present it as describing
  current reality, so leaving a superseded conclusion in place would
  misrepresent the present, not just preserve the past.

### Impact:
Before Phase 8: production probability = raw XGBoost (confirmed,
consistent everywhere); negative-debt treatment = clip-to-zero + indicator
(confirmed unchanged in code); feature count = 147 (120 raw +
26 engineered + 1 sentinel indicator), consistently described as such
rather than as "147 engineered features" anywhere current-facing.

---

## Decision: Business thresholding comes after model and probability selection, not before or alongside

### Context:
Phase 8 needed a default probability to threshold. That probability had to
already be settled — model family, hyperparameters, and calibration
status — before a business decision rule could be built on top of it.

### Choice:
Threshold selection (Phase 8) strictly follows model selection (Phase 6),
calibration investigation (Phase 7), and its bootstrap validation
(Phase 7.8) — all frozen and unchanged in this phase. `src/decision_threshold.py`
consumes the existing dev-only OOF `xgb_probability` column as a fixed
input; it does not retrain, re-tune, or re-calibrate anything.

### Reasoning:
- A business decision threshold is only as trustworthy as the probability
  it's built on. Deriving a threshold against a probability that might
  still change (a different model, different hyperparameters, or a
  different calibration choice) would mean re-deriving the threshold
  every time an upstream modeling decision shifted — exactly the kind of
  rework this project's phase-gated, frozen-artifact structure exists to
  avoid.
- Keeping the two concerns separate also keeps the evidence separate: the
  model/probability's quality is judged by ROC-AUC/PR-AUC/Brier score
  (ranking and calibration quality); the threshold's quality is judged by
  expected business cost (a distinct question). Conflating them risks
  optimizing the threshold search *for* a ranking metric, which this phase
  explicitly avoids by construction (the sweep never touches AUC).

### Alternatives considered:
- **Choose the threshold and the model together (joint optimization)**:
  rejected — would reopen Phase 6/7's already-frozen, already-verified
  decisions for a phase whose explicit goal is downstream of them, and
  would conflate a ranking-quality objective with a business-cost
  objective in one search.

### Impact:
Any future change to the model or probability source is now grounds to
explicitly *re-run* Phase 8's threshold sweep, not to silently invalidate
it — the dependency is one-directional and documented.

---

## Decision: Corrected expected-cost formulation, not the naive brief formula

### Context:
The Phase 8 brief proposed `expected_cost_approve = p*LGD - margin`
(giving break-even threshold `margin/LGD ~= 0.13333`) alongside an
independently agreed theoretical threshold of `margin/(margin+LGD) ~=
0.11765`. These are genuinely different numbers, not a rounding
discrepancy, and the brief explicitly required resolving the conflict
rather than silently picking one.

### Choice:
The implemented decision rule uses
`expected_cost_approve(p) = p*LGD - (1-p)*margin`, which algebraically
reduces to `approve if p < margin/(margin+LGD)` — reproducing the agreed
theoretical threshold exactly (`0.08/0.68 = 0.11765`). The naive formula
is retained in code (`naive_expected_cost_approve`, `naive_threshold`)
for comparison only and is never used for the actual decision.

### Reasoning:
- The naive formula implicitly treats `margin` as earned *unconditionally*
  — as if the lender collects the interest margin regardless of whether
  the loan is ever repaid. That is not how lending economics work: margin
  is only realized if the loan is repaid (probability `1-p`); if the
  borrower defaults (probability `p`), the lender loses `LGD` and earns no
  margin on that loan at all.
- Making margin conditional on `(1-p)` is the standard expected-profit
  framing for a binary lend/don't-lend decision
  (`expected_profit = (1-p)*margin - p*LGD`), and its break-even point is
  exactly `margin/(margin+LGD)` — matching the independently-agreed
  theoretical value is a strong signal this is the formulation the project
  actually intended, not a coincidence to be forced by fitting the code to
  the number.
- `validate_threshold_logic()` hard-asserts
  `expected_cost_approve(theoretical_threshold) == 0` analytically, so any
  future edit that breaks this consistency fails loudly rather than
  silently drifting from the agreed economics again.

### Alternatives considered:
- **Use the naive formula as given, treat 0.11765 as approximate**:
  rejected — explicitly forbidden by the brief ("do not silently assume
  these are equivalent... do not simply force the code to produce
  0.118"), and would have shipped an economically incomplete cost model.

### Impact:
`reports/threshold_analysis.md` Section 4 documents this reconciliation
in full so a future reader (or interviewer) sees the resolution, not just
the final formula.

---

## Decision: Threshold selection uses dev only; holdout is never read in Phase 8

### Context:
`src/decision_threshold.py` needed labeled data to backtest expected cost
per threshold. Both `dev.parquet` (via its OOF predictions) and
`holdout.parquet` have labels.

### Choice:
Only dev-only OOF predictions (`reports/oof_predictions.csv`, regenerated
via the frozen pipeline if missing/stale) are used for the entire sweep,
sensitivity analysis, and threshold selection.
`evaluate_holdout_calibration.load_holdout_data` /
`data_quality.HOLDOUT_PATH` are not imported anywhere in this module —
verified by inspection, not just asserted.

### Reasoning:
- This is the same holdout discipline the project has held since Phase 3:
  holdout exists to produce one honest, never-peeked-at final number.
  Every decision that *shapes* the eventual production behavior — model
  choice, calibration choice, and now the decision threshold — must be
  made without it, or the final holdout evaluation stops being a genuine
  test of anything.
- OOF predictions specifically (not a full-dev-trained model's in-sample
  predictions on dev) were used for the same reason Phase 7 used them for
  calibration fitting: in-sample predictions are optimistic, and an
  optimistic probability distribution would bias the sweep's picture of
  where costs actually cross zero.

### Alternatives considered:
- **Use holdout now "just to see", without acting on it**: rejected per
  the phase's explicit instruction and the project's standing rule — even
  looking, without formally using the result, erodes the honesty of a
  later "final" evaluation in ways that are hard to fully undo.

### Impact:
Holdout remains valid for a genuine final evaluation of the Phase 8
threshold, which is explicitly deferred, not performed in this phase (see
`reports/threshold_analysis.md` Section 12).

---

## Decision: Sensitivity analysis is required, not optional polish

### Context:
LGD and margin were supplied as a single illustrative pair (0.60, 0.08).
A threshold derived from one fixed cost pair looks more authoritative
than it should.

### Choice:
`run_sensitivity_analysis()` sweeps a 5x5 grid of LGD (0.40-0.80) and
margin (0.04-0.12), reporting the theoretical and empirical
cost-minimizing threshold for every combination
(`reports/sensitivity_analysis.csv`, `reports/sensitivity_heatmap.png`).

### Reasoning:
- LGD and margin are business assumptions, not measured quantities in this
  dataset — they were never fit from data and can't be validated the way
  a model metric can. Presenting the resulting threshold without showing
  how much it moves under different plausible assumptions would overstate
  the certainty of a number that is, at its core, a policy input.
- The sensitivity table makes the threshold's *sensitivity itself* legible
  as a finding: the heatmap shows the threshold moving smoothly and
  monotonically with both parameters (higher LGD -> lower threshold;
  higher margin -> higher threshold), which is itself a sanity check on
  the formula's correctness, independent of any single chosen threshold.

### Alternatives considered:
- **Report only the single (0.60, 0.08) result**: rejected — matches the
  brief's explicit requirement to not "stop at a single arbitrary cost
  pair," and would leave a reader unable to judge how much the operating
  point depends on assumptions versus data.

### Impact:
Any future change to the assumed LGD/margin (e.g. a different loan
product or funding environment) can be read directly off the existing
sensitivity table/heatmap without rerunning the sweep from scratch.

---

## Decision: A cost-minimizing threshold is a business policy choice, not the final answer

### Context:
The dev sweep's mathematically cost-minimizing threshold (0.110, tie-broken
from a true empirical minimum at 0.114) approves ~78.9% of applicants —
rejecting far more applicants (21.1%) than the raw default rate (8.07%)
would suggest, because the LGD/margin asymmetry makes the model cautious
near the threshold.

### Choice:
The threshold selected for the next (holdout) evaluation is **0.110**
(reject if `p_default >= 0.110`), but the report explicitly frames this as
a policy choice under the stated economics, not a universally "correct"
number — Section 10 of `reports/threshold_analysis.md` states directly
that a real lender operating under growth or approval-volume constraints
would reasonably choose a different point on the sensitivity curve.

### Reasoning:
- A cost-minimizing threshold answers "what minimizes expected loss given
  these exact LGD/margin assumptions" — it does not answer "what should a
  lender actually do," which also depends on constraints (volume targets,
  regulatory requirements, competitive positioning) entirely outside this
  model's scope. Presenting the number without this framing would
  misrepresent a narrow, well-defined optimization as a general business
  recommendation.
- The tie-break rule itself (documented in `select_threshold()`'s
  docstring) matters here: a naive fixed relative-percent tolerance first
  produced a threshold (0.100) meaningfully worse than the true minimum,
  because the cost curve is genuinely flat near its optimum and a loose
  tolerance swept in points far from it. Replacing this with a
  standard-error-based tolerance (same statistical logic as the Phase 7.8
  bootstrap CI), then tie-breaking toward round numbers *closest to the
  true minimum* rather than the smallest value in the band, produced 0.110
  — close to both the true empirical minimum (0.114) and the theoretical
  break-even (0.1176).

### Alternatives considered:
- **Report the exact, non-rounded empirical minimum (0.114) as the
  threshold**: rejected per the brief's explicit tie-breaking instruction
  — when cost differences are statistically negligible, the simpler,
  more communicable number is preferred and documented as such.

### Impact:
**Threshold locked at 0.110 for the next phase's holdout evaluation.**
`reports/threshold_analysis.md` is the complete record of how this number
was derived, including the two formula candidates, the sweep, the
tie-break correction, and the sensitivity context. Holdout evaluation of
this locked threshold is explicitly deferred to a subsequent phase, per
the Phase 8 methodology rules.

---

## Decision: Final holdout evaluation performed once, after full policy freeze — results are not further tuning inputs

### Context:
By the end of Phase 8, model configuration (Phase 6), probability source
(Phase 7.8, bootstrap-validated), and business threshold (0.110, Phase 8
dev-only sweep) were all frozen. `holdout.parquet` had not been read by
any script in this project up to this point.

### Choice:
`src/evaluate_final_holdout.py` reads `holdout.parquet` for the first
time in the project, trains exactly one XGBoost model on all of
`dev.parquet` (frozen `MODEL_PARAMS`, no CV, no tuning, no calibration),
scores holdout once, and applies the frozen threshold 0.110 exactly once
via the same `evaluate_threshold()` function Phase 8 already uses — no
sweep, no re-selection. A static self-check
(`verify_no_holdout_threshold_search()`) asserts the script's own source
contains no `sweep_thresholds`/`select_threshold` calls and exactly one
`evaluate_threshold()` call against holdout data, so a future edit that
accidentally reintroduces a sweep fails loudly rather than silently.
Result: holdout ROC-AUC 0.7713, PR-AUC 0.2650, Brier 0.066840 (identical
to Phase 7.5/7.8's numbers, confirming reproducibility); at threshold
0.110, mean expected cost -0.038794/applicant, better than dev's cost at
the same threshold (-0.038287) — reported honestly as an observation, not
acted on.

### Reasoning:
- Holdout's entire value depends on it never having influenced the
  decisions it's meant to test. Every decision that shapes production
  behavior — which model, which probability source, which threshold — was
  locked before this file's first `read_parquet('holdout.parquet')` call,
  consistent with every prior phase's holdout discipline
  ([[decision-create-the-devholdout-split-before-any-modeling-and-lock-it]],
  [[decision-holdout-remains-untouched-throughout-the-leakage-audit]]).
- These results are reported as a one-time generalization check, not as
  new evidence to justify changing the model, calibration, or threshold —
  doing so would retroactively make this "final" evaluation just another
  round of dev-style tuning with extra steps.

### Alternatives considered:
- **Re-tune the threshold if holdout suggested a better one**: explicitly
  rejected per this phase's own instructions — holdout performing
  differently from dev (better, in this case) is reported, not acted on.

### Impact:
This is the project's final evaluation artifact
(`reports/final_holdout_results.json`, `reports/final_holdout_report.md`).
`holdout.parquet` should not be read again in this project except for a
genuinely new, separately-justified final check — not for iterating on
Phase 6-8's now-evaluated decisions.

---

## Decision: Pre-Phase-9 hardening pass — wording fixes, reject-inference limitation, exposure-weighted secondary analysis

### Context:
A reviewer pass ahead of Phase 9 (SHAP explainability) identified four
additive gaps in current-facing documentation and one open question about
the Phase 8 economics: (1) "30+ engineered features" was imprecise —
the exact count is 27; (2) reject inference (the model is trained/
evaluated only on Home Credit's historically-approved population) was
never documented as a limitation; (3) the dev-CV-vs-holdout ROC-AUC/
PR-AUC gap had no explanation; (4) Phase 8's threshold assumes unit
exposure per applicant — whether `AMT_CREDIT` was available and suitable
for a secondary exposure-weighted check was unresolved.

### Choice:
(1) "30+" replaced with the exact wording "Engineered 27 applicant-level
features from 2 relational credit-history tables, producing a 147-feature
modelling matrix" in `README.md` and `reports/final_model_card.md`. (2)
A full reject-inference limitation added to `reports/final_model_card.md`
("Known limitations"), cross-referenced (not duplicated) from
`reports/threshold_analysis.md` and (transitively, via its existing
cross-reference) `reports/final_holdout_report.md`. (3) A concise
dev-CV-vs-holdout explanation added to `reports/final_model_card.md`
(full-dev training set size vs. per-fold ~197K; holdout ROC-AUC is
~1.8 SD above the CV mean, not statistically extreme). (4) `AMT_CREDIT`
was confirmed available, complete, and strictly positive in `dev.parquet`
— `src/exposure_weighted_analysis.py` was implemented as a **separate**
script (imports `LGD`/`MARGIN`/the threshold grid from
`decision_threshold.py` unchanged; does not modify it) that reuses the
existing dev-only OOF predictions, joins `AMT_CREDIT`, and sweeps the
same grid weighting realized cost by loan size.

### Reasoning:
- All four items are documentation/analysis additions on top of an
  already-frozen system, not re-openings of Phase 6-8's decisions — the
  distinction matters because this project's credibility rests on
  "frozen means frozen," and every edit here was scoped to stay strictly
  additive.
- The exposure-weighted analysis found the cost-minimizing threshold
  under exposure weighting (0.114) differs from the frozen 0.110 by only
  +0.004 — a small, non-material difference. This is reported as
  *evidence the unit-exposure simplification was reasonable*, not as
  grounds to change production policy; changing the frozen threshold
  based on a newly-run secondary analysis would violate the same
  dev-only, no-post-hoc-changes discipline Phase 8 itself established.
- Reject inference is documented as a limitation of the *data*, not
  reframed as a flaw in this project's modeling choices — the standard,
  honest framing for this well-known credit-scoring issue, and
  explicitly out of scope to solve here (reject-inference modeling is a
  substantial undertaking in its own right).

### Alternatives considered:
- **Skip the exposure-weighted analysis and just document unit-exposure
  as a limitation**: considered, but rejected once `AMT_CREDIT` was
  confirmed clean and available — per the task's own instruction, a
  clean, scope-bounded implementation was preferred over a documentation-
  only placeholder when the data supported it.
- **Fold the exposure-weighted result into the primary threshold
  analysis**: rejected — would blur which result is the frozen
  production policy (unit-exposure, 0.110) and which is supplementary
  context, exactly the ambiguity the task asked to avoid.

### Impact:
`reports/exposure_weighted_analysis.json`/`.md` are new, clearly-labeled
secondary artifacts. The frozen production threshold, model, and
probability source are unchanged. Phase 9 can proceed against the same
frozen artifacts as before this pass.

---

## Decision: SHAP TreeExplainer for global + local explainability, in verified log-odds space

### Context:
Phase 9 needed to explain the frozen XGBoost model's predictions — both
which features generally drive risk (global) and why a specific
applicant received their score (local) — and connect that explanation to
the frozen 0.110 decision threshold, without retraining, tuning, or
introducing a new predictive model.

### Choice:
`src/shap_explainability.py` uses `shap.TreeExplainer` against the same
frozen model training call already validated in Phase 8.5
(`evaluate_holdout_calibration.train_final_model`, imported unchanged).
Global importance is computed on a deterministic 2,000-applicant dev
sample (`random_state=42`); local explanation is a reusable
`explain_applicant()` function returning a structured dict, designed for
direct reuse in Phase 10 (Streamlit). SHAP's output space was verified
empirically against the actual installed `shap==0.52.0`/`xgboost==3.4.1`
— not assumed from memory — before writing any interpretation code:
`TreeExplainer.model_output` defaults to `"raw"`, so SHAP values are
additive in log-odds (raw margin) space
(`base_value + sum(shap_values) == model.predict(X, output_margin=True)`,
confirmed to ~7e-6 over the sample). All prose describing individual SHAP
values uses direction/magnitude language ("risk contributor",
"protective contributor") — never a claimed percentage-point probability
change, since that would require decomposing a nonlinear sigmoid
additively, which log-odds SHAP values do not do. The predicted
probability, threshold distance, and APPROVE/REJECT decision are computed
directly via `model.predict_proba`, entirely independent of SHAP.

### Reasoning:
- Global + local coverage answers both questions this phase needed to
  answer ("what drives the model in general" and "why did this applicant
  get this score") without requiring interaction values, dependence
  plots for dozens of features, counterfactuals, LIME, or any other
  heavier XAI method — all explicitly out of scope for a project whose
  stated goal is a credible, interview-defensible explainability layer,
  not a research contribution.
- Verifying the output space empirically (rather than assuming SHAP
  values are probability-space, a common mistake) was necessary to avoid
  shipping a mathematically incorrect claim like "this feature increased
  default probability by 5%" — the kind of imprecision that would
  undermine the project's credibility in exactly the audience (technical
  interviewers) it's aimed at.
- Reusing `train_final_model` and `MODEL_PARAMS` rather than redefining
  them keeps this phase strictly an explanation layer on top of the
  already-frozen model, not a parallel implementation that could
  silently drift from it.
- The local-explanation demo applicant is deliberately chosen (not
  arbitrary or random) as a REJECT case ~5 percentage points above the
  threshold — the most illustrative margin for demonstrating
  threshold-relative reasoning, deterministically selected by minimizing
  distance to a target probability, not by chance.

### Alternatives considered:
- **SHAP with `model_output="probability"`**: available in SHAP but not
  used — it changes the computation path (interventional feature
  perturbation against a background dataset) and was not necessary once
  the default "raw" output was verified consistent and additive; adding
  a second SHAP configuration would have expanded scope without a clear
  benefit for this project's needs.
- **LIME or counterfactual explanations**: rejected per this phase's
  explicit scope boundaries — SHAP's exactness for tree models (no
  surrogate-model approximation error) and its now-standard status in
  applied ML make it the more defensible, more interview-relevant choice
  for this project.

### Impact:
`explain_applicant()` is the reusable interface Phase 10's Streamlit
layer should call directly — same signature, same structured return
value. Global importance results should be read as a description of the
model's learned behavior on the historically-approved (reject-inference-
limited) population, not a causal or fairness claim.

---

## Decision: Streamlit deployment layer with a build-once model artifact, curated inputs, and a dev-median/mode template for unexposed features

### Context:
Phase 10 needed a deployment/demo layer around the frozen model, threshold,
and Phase 9 SHAP explainer — without retraining, without reading holdout
data, and without exposing all 147 model features as manual UI fields
(explicitly ruled out as unusable). The 147 features are produced by a
relational SQL aggregation over `bureau.csv`/`previous_application.csv`
joined onto `application_train` (`src/features.py`) — there is no way for an
interactive form to re-run that aggregation for a hypothetical applicant who
has no rows in those tables, so a direct "form maps 1:1 onto the SQL
pipeline" design was never on the table. See
`reports/phase10_deployment_report.md` for the full architecture writeup.

### Choice:
- **`src/build_model_artifact.py`** (new, run once, offline): calls
  `evaluate_holdout_calibration.train_final_model()` unchanged — the same
  training call Phase 8.5 and Phase 9 already use — and persists the result
  via XGBoost's native `model.save_model()`, plus a `feature_schema.json`
  capturing the frozen 147-column order, each categorical column's exact
  training-time category set, and a per-column dev median/mode template.
- **`src/inference.py`** (new): the only model-facing module the app
  imports. Loads the artifact once (never trains), reconstructs a
  147-column applicant row from a curated ~35-field input plus the six
  ratio features (recomputed with `src/features.py`'s exact formulas) plus
  the frozen template for every unexposed feature, and calls
  `predict_proba` / the frozen 0.110 threshold (imported from
  `evaluate_final_holdout.FROZEN_THRESHOLD`) / Phase 9's unchanged
  `explain_applicant()`.
- **`app/app.py`** (new): UI only. Grouped applicant-input sections,
  threshold-relative risk result, decision-aware SHAP explanation, a model
  information panel, and three hand-constructed, verified synthetic demo
  profiles (APPROVE / near-threshold REJECT / clear REJECT) — explicitly
  labeled synthetic, not derived from dev or holdout data.
- **"No history" stays missing, not zero**: unchecking "has bureau credit
  history" / "has previous applications" in the UI sets those columns to
  `None` (NaN), preserving the same distinction `src/features.py` already
  documents, rather than letting an unset numeric field silently mean "zero
  loans."

### Reasoning:
- Retraining on every app run was rejected outright: Phase 6-9 already
  established a single, validated training call: persisting its output once
  is a packaging concern, not a modeling change, and keeps the interactive
  app fast without touching MODEL_PARAMS or the training methodology.
- The curated input fields were deliberately chosen to cover every one of
  the top-15 global SHAP features from `reports/shap_explainability_report.md`
  (directly, or as an input to a derived ratio) — so a user can actually
  move the fields that drive this model's predictions, not a disconnected
  subset that happens to look like a form.
- A single-row categorical DataFrame with a missing value has zero
  locally-inferred pandas categories, and XGBoost hard-errors on that
  (verified empirically, not assumed) — `build_feature_row()` therefore
  always casts categorical columns with an explicit `categories=` list from
  the frozen schema, both preventing that crash and guaranteeing every UI
  dropdown only offers values the model actually saw in training.
- The dev-median/mode template for unexposed features (rather than
  imputation logic invented for this phase) was chosen because it is the
  simplest defensible way to fill ~112 fields nobody would reasonably type
  into a form, and it is disclosed explicitly in the app rather than
  presented as a complete applicant profile.

### Alternatives considered:
- **Re-running the full relational feature pipeline per submitted
  applicant**: rejected — there are no bureau/previous-application rows for
  a hypothetical applicant to aggregate; this was the core constraint the
  whole design works around, not an oversight.
- **Exposing all 147 features as manual fields**: rejected per the explicit
  project brief — unusable, and most of the excluded fields (building/
  apartment descriptors, `FLAG_DOCUMENT_2`..`21`) carry little individual
  signal.
- **Zero-filling unexposed/missing fields**: rejected — contradicts this
  project's own established "missing ≠ zero" convention
  (`src/features.py`, `reports/final_model_card.md`) and would silently
  misrepresent "no bureau history" as "zero debt," a materially different
  signal to a model trained to distinguish the two.
- **A FastAPI/REST serving layer instead of Streamlit**: rejected — no
  concrete requirement calls for a separate client/server split for a
  single-demo portfolio project; `streamlit` was already in
  `requirements.txt` from earlier planning, and a REST layer would add
  surface area without adding interview-relevant capability.

### Impact:
The frozen model, threshold, and Phase 9 SHAP logic are unchanged — Phase 10
only adds a consumption layer on top. `models/xgboost_frozen.json` and
`models/feature_schema.json` are build artifacts checked into the repo (not
`data/processed/`-style raw-data derivatives) so the app runs without
requiring `data/raw/` to be present. Rebuilding the artifact after any
future change to `src/features.py`, `src/train_xgboost.py`, or the dev/
holdout split requires re-running `src/build_model_artifact.py`.

---
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
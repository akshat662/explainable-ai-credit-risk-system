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
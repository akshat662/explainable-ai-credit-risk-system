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
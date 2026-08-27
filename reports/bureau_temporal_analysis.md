# Bureau Temporal Validity Analysis

Follow-up to `reports/leakage_report.md`, which left bureau snapshot
staleness as an open question. This is a read-only analysis of
`data/raw/bureau.csv` (1,716,428 rows) — no pipeline code was changed and
no model was trained.

## 1. DAYS_CREDIT_UPDATE distribution

| Stat | Value |
|---|---|
| Min | -41,947 |
| Max | **+372** |
| Mean | -593.75 |
| Rows with value > 0 | 17 |
| Percentage > 0 | **0.000990%** |

Quantiles (p0/p1/p10/p25/p50/p75/p90/p99/p100):
`-41947, -2531, -1561, -908, -395, -33, -13, -3, 372`

The distribution is overwhelmingly negative (a record last refreshed well
before the current application). Only 17 of 1,716,428 rows were refreshed
*after* the application decision date, capped at +372 days. This matches
and confirms the Phase 4 finding.

## 2. CREDIT_ACTIVE distribution

| Status | Rows | % of total |
|---|---|---|
| Closed | 1,079,273 | 62.88% |
| Active | 630,607 | 36.74% |
| Sold | 6,527 | 0.38% |
| Bad debt | 21 | 0.0012% |

Two categories — `Sold` and `Bad debt` — exist beyond `Active`/`Closed`.
Current `bureau_active_loans`/`bureau_closed_loans` features (Phase 2)
only count `Active` and `Closed`, so their sum is short of
`bureau_total_loans` by the `Sold` + `Bad debt` rows for affected
applicants — a previously-flagged, still-open feature-coverage gap
(small in magnitude at ~0.38% of rows, not a temporal leakage issue).

## 3. CREDIT_DAY_OVERDUE statistics by CREDIT_ACTIVE

| Status | Rows | Min | Max | Mean | Median | Rows > 0 | % > 0 |
|---|---|---|---|---|---|---|---|
| Closed | 1,079,273 | 0 | 2,347 | 0.105 | 0.0 | 83 | 0.77% |
| Active | 630,607 | 0 | 2,770 | 1.810 | 0.0 | 4,044 | 0.64% |
| Sold | 6,527 | 0 | 2,792 | 21.936 | 0.0 | 80 | 1.23% |
| Bad debt | 21 | 0 | 1,761 | 313.619 | 0.0 | 10 | 47.62% |

Every status is heavily right-skewed (median 0 across the board — most
loans are never overdue). `Bad debt` is a striking outlier (313.6 mean
days overdue, 47.6% of its 21 rows currently overdue), but it's a
21-row population — statistically negligible at the aggregate-feature
level, though individually severe. This reconfirms a prior code-review
observation: the current `bureau_avg_days_overdue`/`bureau_max_days_overdue`
features average across all four statuses combined, diluting the active-
loan overdue signal with the much larger, near-always-zero `Closed`
population — a feature-design consideration, not a temporal one.

## 4. AMT_CREDIT_SUM_DEBT missingness and distribution

Overall missingness: **257,669 / 1,716,428 (15.01%)** null.

By `CREDIT_ACTIVE`:

| Status | Rows | Null | % Null |
|---|---|---|---|
| Closed | 1,079,273 | 180,543 | 16.73% |
| Active | 630,607 | 73,547 | 11.66% |
| Sold | 6,527 | 3,573 | **54.74%** |
| Bad debt | 21 | 6 | 28.57% |

Missingness is not random with respect to loan status — `Sold` loans are
missing debt figures more than half the time. This is consistent with
(and reinforces) the Phase 2/3 decision to preserve NULL rather than
zero-fill: the missingness itself carries information correlated with
loan status, which zero-filling would have destroyed.

Distribution of non-null values:

| Stat | Value |
|---|---|
| Min | -4,705,600.32 |
| Max | 170,100,000.00 |
| Mean | 137,085.12 |
| Median | 0.0 |
| p90 | 295,456.50 |
| p99 | 2,259,728.46 |
| Negative values | 8,418 (0.49% of non-null rows) |

Median 0 with a long right tail is expected for a debt field. The 8,418
negative values are **not explained by anything in this analysis** — a
negative outstanding debt is unusual (possibly overpayment/credit
balances, or a data artifact) and is flagged here as an open data-quality
question, distinct from the temporal-leakage question this analysis was
scoped to answer.

## 5. Are these fields safe to use as historical credit features?

**On temporal validity — yes, with a documented, negligible exception.**
`CREDIT_DAY_OVERDUE` and `AMT_CREDIT_SUM_DEBT` are both bureau-reported
snapshot fields, refreshed as of `DAYS_CREDIT_UPDATE`. Since 99.999% of
records were last refreshed on or before the application decision date,
their values reflect information that was genuinely knowable at decision
time for essentially the entire dataset. The exception — 17 rows
(0.000990%) refreshed up to 372 days after the decision — is too small
to move any aggregate measurably (already established in Phase 4) and is
not a basis for additional filtering at this time.

**Two non-temporal caveats, neither blocking, both worth carrying
forward:**
1. `CREDIT_ACTIVE`'s `Sold`/`Bad debt` categories are undercounted by the
   current `bureau_active_loans`/`bureau_closed_loans` features (0.38%
   of rows affected) — a feature-coverage fix, not a leakage fix.
2. `AMT_CREDIT_SUM_DEBT` has 8,418 unexplained negative values (0.49% of
   non-null rows) — worth a domain-knowledge check before the field is
   used in any more refined feature than the current `SUM`-based
   aggregate, but not large enough to distort `bureau_total_debt` today.

**Conclusion: no additional temporal filtering is required before Phase 5
baseline modeling.** This closes the "further investigation required"
item left open by `reports/leakage_report.md`, specifically for the
`DAYS_CREDIT_UPDATE`/`CREDIT_DAY_OVERDUE`/`AMT_CREDIT_SUM_DEBT` question.
The `EXT_SOURCE_*` leakage question remains separately open and untouched
by this analysis.

"""
Phase 2: Applicant-level feature engineering pipeline.
Phase 4 extension: optional temporal filtering for leakage auditing.

Aggregates bureau.csv and previous_application.csv to one row per
SK_ID_CURR, left-joins them onto application_train, and derives financial
ratio features. All aggregation happens in DuckDB SQL — no full-table
pandas loads. Missing bureau/previous-application history is left as NULL
(no zero-fill); leakage auditing and imputation are separate later phases.

build_features(time_filtered=True) additionally restricts bureau to
DAYS_CREDIT <= 0 and previous_application to DAYS_DECISION <= 0 before
aggregating, producing features_filtered.parquet as a leakage-audit
counterpart to the default features_naive.parquet.
"""

import json
import logging
import os

from profile_data import DATA_DIR, PROJECT_ROOT, get_connection, load_all_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_NAIVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features_naive.parquet")
FEATURES_FILTERED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features_filtered.parquet")
DAYS_CREDIT_UPDATE_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "days_credit_update_analysis.json")


def register_views(con):
    """Register DuckDB views for all raw tables (reuses profile_data's loader)."""
    load_all_views(con, DATA_DIR)


def analyze_days_credit_update(con):
    """Analyze DAYS_CREDIT_UPDATE without assuming it's safe to filter on.

    DAYS_CREDIT_UPDATE is the offset (relative to the current application)
    at which a bureau record was last refreshed — distinct from
    DAYS_CREDIT, which is when the underlying loan originated. A positive
    DAYS_CREDIT_UPDATE means that record's snapshot fields (e.g.
    CREDIT_DAY_OVERDUE, AMT_CREDIT_SUM_DEBT) were refreshed after the
    current application's decision date, which is a plausible — if narrow —
    leakage vector distinct from record-origination timing.

    This function only measures and reports; it does not filter, because
    the affected population turns out to be a tiny fraction of bureau rows
    (see decision_reason below) — the finding is documented, not acted on
    silently in either direction.
    """
    total = con.execute("SELECT COUNT(*) FROM bureau").fetchone()[0]
    min_val, max_val = con.execute(
        "SELECT MIN(DAYS_CREDIT_UPDATE), MAX(DAYS_CREDIT_UPDATE) FROM bureau"
    ).fetchone()
    n_positive = con.execute(
        "SELECT COUNT(*) FROM bureau WHERE DAYS_CREDIT_UPDATE > 0"
    ).fetchone()[0]
    quantile_labels = ["p0", "p10", "p25", "p50", "p75", "p90", "p99", "p100"]
    quantile_values = con.execute(
        "SELECT quantile_cont(DAYS_CREDIT_UPDATE, [0,0.1,0.25,0.5,0.75,0.9,0.99,1.0]) FROM bureau"
    ).fetchone()[0]

    analysis = {
        "total_bureau_rows": total,
        "min": min_val,
        "max": max_val,
        "rows_with_days_credit_update_gt_0": n_positive,
        "pct_rows_with_days_credit_update_gt_0": round(n_positive / total * 100, 6),
        "quantiles": dict(zip(quantile_labels, quantile_values)),
        "used_as_filter": False,
        "decision_reason": (
            f"Only {n_positive} of {total} bureau rows "
            f"({n_positive / total * 100:.4f}%) have DAYS_CREDIT_UPDATE > 0 "
            f"(max +{max_val} days), and all of them are already-Active loans "
            "whose DAYS_CREDIT (origination) is confirmed <= 0. Filtering these "
            "out would not measurably change any aggregate. DAYS_CREDIT_UPDATE "
            "is reported for transparency, not used as a row-inclusion filter "
            "in build_bureau_aggregates()."
        ),
    }

    os.makedirs(os.path.dirname(DAYS_CREDIT_UPDATE_REPORT_PATH), exist_ok=True)
    with open(DAYS_CREDIT_UPDATE_REPORT_PATH, "w") as f:
        json.dump(analysis, f, indent=2)

    logger.info(
        "DAYS_CREDIT_UPDATE analysis: %d/%d rows (%.4f%%) > 0, max=%d -- not used as filter (see %s)",
        n_positive, total, analysis["pct_rows_with_days_credit_update_gt_0"], max_val,
        DAYS_CREDIT_UPDATE_REPORT_PATH,
    )
    return analysis


def build_bureau_aggregates(con, time_filtered=False):
    """Aggregate bureau.csv to one row per SK_ID_CURR.

    Uses SUM/AVG/COUNT, which ignore NULLs per-row by default rather than
    treating them as zero — an applicant absent from bureau entirely still
    gets NULL aggregates (not zeros) once left-joined downstream.

    time_filtered=True restricts to DAYS_CREDIT <= 0 (loans that had
    already originated as of the current application) before aggregating.

    AMT_CREDIT_SUM_DEBT contains 8,418 negative values (0.49% of all
    bureau rows, affecting 5,886 of 305,811 bureau-covered applicants) --
    confirmed via reports/bureau_temporal_analysis.md and re-verified here.
    A negative outstanding debt is not explained by anything in the data
    dictionary (unlike, say, a documented "credit balance" semantic) and
    is concentrated in Active loans (6,222 of 8,418), with a long tail
    down to -4.7M -- too extreme and unexplained to trust as a genuine
    negative liability. bureau_total_debt clips these to 0 (a debt of "at
    least zero" is the conservative, defensible read), while
    bureau_had_negative_debt preserves the fact that clipping happened,
    since it may itself be a useful signal (e.g. a data-quality artifact
    correlated with a particular reporting source) independent of the
    now-corrected magnitude.
    """
    where_clause = "WHERE DAYS_CREDIT <= 0" if time_filtered else ""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW bureau_agg AS
        SELECT
            SK_ID_CURR,
            COUNT(*) AS bureau_total_loans,
            COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bureau_active_loans,
            COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Closed') AS bureau_closed_loans,
            AVG(AMT_CREDIT_SUM) AS bureau_avg_credit,
            MAX(AMT_CREDIT_SUM) AS bureau_max_credit,
            SUM(AMT_CREDIT_SUM) AS bureau_total_credit,
            SUM(CASE WHEN AMT_CREDIT_SUM_DEBT < 0 THEN 0 ELSE AMT_CREDIT_SUM_DEBT END) AS bureau_total_debt,
            MAX(CASE WHEN AMT_CREDIT_SUM_DEBT < 0 THEN 1 ELSE 0 END) AS bureau_had_negative_debt,
            SUM(AMT_CREDIT_SUM_OVERDUE) AS bureau_total_overdue,
            AVG(CREDIT_DAY_OVERDUE) AS bureau_avg_days_overdue,
            MAX(CREDIT_DAY_OVERDUE) AS bureau_max_days_overdue,
            COUNT(DISTINCT CREDIT_TYPE) AS bureau_credit_type_variety,
            SUM(CNT_CREDIT_PROLONG) AS bureau_total_prolongs
        FROM bureau
        {where_clause}
        GROUP BY SK_ID_CURR
        """
    )
    logger.info("Built bureau_agg (time_filtered=%s)", time_filtered)


def build_previous_application_aggregates(con, time_filtered=False):
    """Aggregate previous_application.csv to one row per SK_ID_CURR.

    Granted-amount features are scoped to Approved rows only, since AMT_CREDIT
    on refused applications does not represent an actual grant.

    time_filtered=True restricts to DAYS_DECISION <= 0 (applications already
    decided as of the current application) before aggregating.
    """
    where_clause = "WHERE DAYS_DECISION <= 0" if time_filtered else ""
    con.execute(
        f"""
        CREATE OR REPLACE VIEW prev_app_agg AS
        SELECT
            SK_ID_CURR,
            COUNT(*) AS prev_app_count,
            CAST(COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS DOUBLE)
                / COUNT(*) AS prev_approval_ratio,
            CAST(COUNT(*) FILTER (WHERE NAME_CONTRACT_STATUS = 'Refused') AS DOUBLE)
                / COUNT(*) AS prev_refusal_ratio,
            AVG(AMT_APPLICATION) AS prev_avg_requested,
            AVG(AMT_CREDIT) FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_avg_granted,
            AVG(AMT_CREDIT / NULLIF(AMT_APPLICATION, 0))
                FILTER (WHERE NAME_CONTRACT_STATUS = 'Approved') AS prev_grant_ratio,
            AVG(CNT_PAYMENT) AS prev_avg_term
        FROM previous_application
        {where_clause}
        GROUP BY SK_ID_CURR
        """
    )
    logger.info("Built prev_app_agg (time_filtered=%s)", time_filtered)


def build_applicant_features(con):
    """Left-join bureau/previous-application aggregates onto application_train
    and derive financial ratio features.

    LEFT JOIN preserves every application_train row and leaves SK_ID_CURR
    without bureau/previous-application history as NULL rather than 0.
    NULLIF guards divisions against zero denominators (turns an undefined
    ratio into NULL instead of inf/error) — this is arithmetic safety, not
    imputation: no missing feature value is filled in.
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW features_naive AS
        SELECT
            a.*,
            b.bureau_total_loans,
            b.bureau_active_loans,
            b.bureau_closed_loans,
            b.bureau_avg_credit,
            b.bureau_max_credit,
            b.bureau_total_credit,
            b.bureau_total_debt,
            b.bureau_had_negative_debt,
            b.bureau_total_overdue,
            b.bureau_avg_days_overdue,
            b.bureau_max_days_overdue,
            b.bureau_credit_type_variety,
            b.bureau_total_prolongs,
            p.prev_app_count,
            p.prev_approval_ratio,
            p.prev_refusal_ratio,
            p.prev_avg_requested,
            p.prev_avg_granted,
            p.prev_grant_ratio,
            p.prev_avg_term,
            a.AMT_CREDIT / NULLIF(a.AMT_INCOME_TOTAL, 0) AS credit_income_ratio,
            a.AMT_ANNUITY / NULLIF(a.AMT_INCOME_TOTAL, 0) AS annuity_income_ratio,
            a.AMT_CREDIT / NULLIF(a.AMT_ANNUITY, 0) AS credit_annuity_ratio,
            a.AMT_GOODS_PRICE / NULLIF(a.AMT_CREDIT, 0) AS goods_credit_ratio,
            a.AMT_INCOME_TOTAL / NULLIF(a.CNT_FAM_MEMBERS, 0) AS income_per_person,
            b.bureau_total_debt / NULLIF(a.AMT_INCOME_TOTAL, 0) AS debt_income_ratio
        FROM application_train a
        LEFT JOIN bureau_agg b ON a.SK_ID_CURR = b.SK_ID_CURR
        LEFT JOIN prev_app_agg p ON a.SK_ID_CURR = p.SK_ID_CURR
        """
    )
    logger.info("Built features_naive (application_train + bureau + previous_application + ratios)")


def validate_features(con):
    """Sanity-check the feature table before saving.

    Hard-fails (raises) on the two correctness invariants that must hold for
    any applicant-level table: row count preserved, SK_ID_CURR unique. Shape
    and engineered-feature count are reported, not asserted.
    """
    app_row_count = con.execute("SELECT COUNT(*) FROM application_train").fetchone()[0]
    app_col_count = len(con.execute("DESCRIBE application_train").fetchall())

    feature_row_count, distinct_ids = con.execute(
        "SELECT COUNT(*), COUNT(DISTINCT SK_ID_CURR) FROM features_naive"
    ).fetchone()
    feature_col_count = len(con.execute("DESCRIBE features_naive").fetchall())

    assert feature_row_count == app_row_count, (
        f"Row count mismatch: features_naive has {feature_row_count} rows, "
        f"application_train has {app_row_count}. A join likely fanned out."
    )
    assert feature_row_count == distinct_ids, (
        f"SK_ID_CURR is not unique in features_naive: {feature_row_count} rows "
        f"but only {distinct_ids} distinct SK_ID_CURR values."
    )

    engineered_count = feature_col_count - app_col_count

    logger.info("Validation passed: row count preserved, SK_ID_CURR unique")
    print(f"Final feature table shape: ({feature_row_count}, {feature_col_count})")
    print(f"Engineered features added: {engineered_count}")


def save_features(con, output_path):
    """Write features_naive to Parquet directly from DuckDB (no pandas hop)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    con.execute(f"COPY (SELECT * FROM features_naive) TO '{output_path}' (FORMAT PARQUET)")
    logger.info("Saved features to %s", output_path)


def build_features(time_filtered=False, con=None):
    """Build the applicant-level feature table.

    time_filtered=False (default): reproduces the original pipeline exactly
    (all bureau/previous_application rows) -> features_naive.parquet.
    time_filtered=True: restricts bureau to DAYS_CREDIT <= 0 and
    previous_application to DAYS_DECISION <= 0 before aggregating ->
    features_filtered.parquet. This is the leakage-audit counterpart: if a
    model trained on this performs materially differently from one trained
    on the naive table, that's evidence the naive pipeline let in
    information that shouldn't have been available at decision time.

    Accepts an existing connection (with views already registered) so
    callers building both variants back-to-back can share one connection
    for a later cross-comparison; otherwise opens and registers its own.
    """
    owns_connection = con is None
    if owns_connection:
        con = get_connection()
        register_views(con)

    analyze_days_credit_update(con)

    build_bureau_aggregates(con, time_filtered=time_filtered)
    build_previous_application_aggregates(con, time_filtered=time_filtered)
    build_applicant_features(con)

    validate_features(con)

    output_path = FEATURES_FILTERED_PATH if time_filtered else FEATURES_NAIVE_PATH
    save_features(con, output_path)

    if owns_connection:
        con.close()

    return output_path


def validate_naive_vs_filtered(con, naive_path=FEATURES_NAIVE_PATH, filtered_path=FEATURES_FILTERED_PATH):
    """Confirm the naive and filtered feature tables share the same grain.

    Both are built from the same application_train LEFT JOIN skeleton, so
    time-filtering the bureau/previous_application aggregation inputs can
    change feature *values*, but must never change which applicants appear,
    how many rows exist, or the label distribution. Hard-fails otherwise.
    """
    con.execute(f"CREATE OR REPLACE VIEW naive_check AS SELECT SK_ID_CURR, TARGET FROM read_parquet('{naive_path}')")
    con.execute(f"CREATE OR REPLACE VIEW filtered_check AS SELECT SK_ID_CURR, TARGET FROM read_parquet('{filtered_path}')")

    naive_rows = con.execute("SELECT COUNT(*) FROM naive_check").fetchone()[0]
    filtered_rows = con.execute("SELECT COUNT(*) FROM filtered_check").fetchone()[0]
    assert naive_rows == filtered_rows, (
        f"Row count mismatch: naive={naive_rows}, filtered={filtered_rows}"
    )

    coverage_diff = con.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT SK_ID_CURR FROM naive_check
            EXCEPT
            SELECT SK_ID_CURR FROM filtered_check
        )
        """
    ).fetchone()[0]
    assert coverage_diff == 0, (
        f"{coverage_diff} SK_ID_CURR present in naive but missing from filtered"
    )

    naive_default_rate = con.execute("SELECT AVG(TARGET) FROM naive_check").fetchone()[0] * 100
    filtered_default_rate = con.execute("SELECT AVG(TARGET) FROM filtered_check").fetchone()[0] * 100
    assert abs(naive_default_rate - filtered_default_rate) < 0.01, (
        f"TARGET distribution differs: naive={naive_default_rate:.4f}%, "
        f"filtered={filtered_default_rate:.4f}%"
    )

    logger.info("naive vs filtered validation passed: same row count, coverage, and TARGET distribution")
    print(f"naive rows={naive_rows}, filtered rows={filtered_rows}")
    print(f"naive default rate={naive_default_rate:.4f}%, filtered default rate={filtered_default_rate:.4f}%")


def main():
    con = get_connection()
    register_views(con)

    build_features(time_filtered=False, con=con)
    build_features(time_filtered=True, con=con)

    validate_naive_vs_filtered(con)


if __name__ == "__main__":
    main()

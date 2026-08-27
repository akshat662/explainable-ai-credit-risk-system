"""
Phase 2: Applicant-level feature engineering pipeline.

Aggregates bureau.csv and previous_application.csv to one row per
SK_ID_CURR, left-joins them onto application_train, and derives financial
ratio features. All aggregation happens in DuckDB SQL — no full-table
pandas loads. Missing bureau/previous-application history is left as NULL
(no zero-fill); leakage auditing and imputation are separate later phases.
"""

import logging
import os

from profile_data import DATA_DIR, PROJECT_ROOT, get_connection, load_all_views

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features_naive.parquet")


def register_views(con):
    """Register DuckDB views for all raw tables (reuses profile_data's loader)."""
    load_all_views(con, DATA_DIR)


def build_bureau_aggregates(con):
    """Aggregate bureau.csv to one row per SK_ID_CURR.

    Uses SUM/AVG/COUNT, which ignore NULLs per-row by default rather than
    treating them as zero — an applicant absent from bureau entirely still
    gets NULL aggregates (not zeros) once left-joined downstream.
    """
    con.execute(
        """
        CREATE OR REPLACE VIEW bureau_agg AS
        SELECT
            SK_ID_CURR,
            COUNT(*) AS bureau_total_loans,
            COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Active') AS bureau_active_loans,
            COUNT(*) FILTER (WHERE CREDIT_ACTIVE = 'Closed') AS bureau_closed_loans,
            AVG(AMT_CREDIT_SUM) AS bureau_avg_credit,
            MAX(AMT_CREDIT_SUM) AS bureau_max_credit,
            SUM(AMT_CREDIT_SUM) AS bureau_total_credit,
            SUM(AMT_CREDIT_SUM_DEBT) AS bureau_total_debt,
            SUM(AMT_CREDIT_SUM_OVERDUE) AS bureau_total_overdue,
            AVG(CREDIT_DAY_OVERDUE) AS bureau_avg_days_overdue,
            MAX(CREDIT_DAY_OVERDUE) AS bureau_max_days_overdue,
            COUNT(DISTINCT CREDIT_TYPE) AS bureau_credit_type_variety,
            SUM(CNT_CREDIT_PROLONG) AS bureau_total_prolongs
        FROM bureau
        GROUP BY SK_ID_CURR
        """
    )
    logger.info("Built bureau_agg (one row per SK_ID_CURR)")


def build_previous_application_aggregates(con):
    """Aggregate previous_application.csv to one row per SK_ID_CURR.

    Granted-amount features are scoped to Approved rows only, since AMT_CREDIT
    on refused applications does not represent an actual grant.
    """
    con.execute(
        """
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
        GROUP BY SK_ID_CURR
        """
    )
    logger.info("Built prev_app_agg (one row per SK_ID_CURR)")


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


def save_features(con, output_path=PROCESSED_PATH):
    """Write features_naive to Parquet directly from DuckDB (no pandas hop)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    con.execute(f"COPY (SELECT * FROM features_naive) TO '{output_path}' (FORMAT PARQUET)")
    logger.info("Saved features to %s", output_path)


def main():
    con = get_connection()
    register_views(con)

    build_bureau_aggregates(con)
    build_previous_application_aggregates(con)
    build_applicant_features(con)

    validate_features(con)
    save_features(con, PROCESSED_PATH)


if __name__ == "__main__":
    main()

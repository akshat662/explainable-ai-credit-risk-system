"""
Phase 3: Data quality fixes and holdout creation.

Cleans the known DAYS_EMPLOYED sentinel, profiles missingness across the
engineered feature table, and produces a locked dev/holdout split. No
imputation, feature selection, or modeling happens here — this phase only
prepares the dataset so later phases can experiment on `dev` while
`holdout` stays untouched until final evaluation.
"""

import logging
import os

from sklearn.model_selection import train_test_split

from profile_data import PROJECT_ROOT, get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features_naive.parquet")
DEV_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "dev.parquet")
HOLDOUT_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "holdout.parquet")
MISSING_REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "missing_values.md")

# 365243 days (~1000 years) is Home Credit's placeholder for "not
# applicable" on DAYS_EMPLOYED — used for Pensioner/Unemployed applicants
# who have no current employment duration. It is not a real value and must
# never be aggregated or ratio'd against as if it were.
DAYS_EMPLOYED_SENTINEL = 365243

HOLDOUT_FRACTION = 0.2
RANDOM_STATE = 42


def load_features(con, features_path=FEATURES_PATH):
    """Register a view over the Phase 2 feature table."""
    if not os.path.isfile(features_path):
        raise FileNotFoundError(
            f"Expected engineered feature table at '{features_path}' but it was not found. "
            f"Run src/features.py first to generate it."
        )
    con.execute(f"CREATE OR REPLACE VIEW features_naive AS SELECT * FROM read_parquet('{features_path}')")
    logger.info("Loaded features_naive from %s", features_path)


def clean_sentinel_values(con):
    """Flag and null out the DAYS_EMPLOYED sentinel.

    Must run before any employment-related ratio feature is computed
    (e.g. an income-per-year-employed feature), since dividing by 365243
    would silently produce a meaningless near-zero ratio instead of NaN.
    `days_employed_anomaly` preserves the fact that the value was a
    sentinel, since that itself carries signal (it flags Pensioner/
    Unemployed applicants) independent of the now-missing raw value.
    """
    con.execute(
        f"""
        CREATE OR REPLACE VIEW features_clean AS
        SELECT
            * EXCLUDE (DAYS_EMPLOYED),
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN 1 ELSE 0 END
                AS days_employed_anomaly,
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN NULL ELSE DAYS_EMPLOYED END
                AS DAYS_EMPLOYED
        FROM features_naive
        """
    )
    n_sentinel = con.execute(
        f"SELECT COUNT(*) FROM features_clean WHERE days_employed_anomaly = 1"
    ).fetchone()[0]
    logger.info("Flagged %d rows with DAYS_EMPLOYED sentinel (365243 -> NaN)", n_sentinel)


def profile_missing_values(con, table_name="features_clean"):
    """Compute missing percentage per column in a single table scan.

    Returns a list of (column, missing_pct) sorted descending. Uses one
    aggregate query with COUNT(col) per column (COUNT ignores NULLs) rather
    than a query-per-column, so the whole table is scanned once.
    """
    columns = [row[0] for row in con.execute(f"DESCRIBE {table_name}").fetchall()]

    select_exprs = ", ".join(f'COUNT("{c}") AS "{c}"' for c in columns)
    total, *present_counts = con.execute(
        f"SELECT COUNT(*) AS total, {select_exprs} FROM {table_name}"
    ).fetchone()

    profile = [
        (col, round((total - present) / total * 100, 4))
        for col, present in zip(columns, present_counts)
    ]
    profile.sort(key=lambda pair: pair[1], reverse=True)
    return profile


def categorize_treatment(feature_name):
    """Map a feature to its documented missing-value treatment and reason.

    Reflects Phase 2's design: bureau_*/prev_* NULLs mean "no history," not
    zero, and must stay NaN. No imputation happens in this phase regardless
    of category — this only records the intended future treatment.
    """
    if feature_name.startswith("EXT_SOURCE_"):
        return (
            "Keep NaN; consider missingness indicator",
            "External bureau score; absence may itself be predictive, but "
            "must not be imputed before leakage review.",
        )
    if feature_name.startswith("bureau_"):
        return (
            "Keep NaN",
            "Missing means the applicant has no bureau-reported credit "
            "history, not zero history.",
        )
    if feature_name.startswith("prev_"):
        return (
            "Keep NaN",
            "Missing means the applicant has no previous Home Credit "
            "applications, not a zero-value application.",
        )
    return (
        "Leave unchanged for now",
        "Not yet reviewed; deferred to a later data-quality/imputation phase.",
    )


def write_missing_value_report(profile, output_path=MISSING_REPORT_PATH, top_n=15):
    """Write the full missingness profile plus manual treatment notes.

    Two treatment sections are included: the literal top N by missing %
    (as specified), and a supplementary section for the EXT_SOURCE_*/
    bureau_*/prev_* categories — since those categories have documented
    treatment rules but, in the current data, do not happen to fall inside
    the top N by raw percentage (they're outranked by the many building/
    apartment descriptive columns, which are ~48-70% missing).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# Missing Value Report",
        "",
        "Generated from `features_clean` (post sentinel-cleaning, pre-imputation) "
        "in `src/data_quality.py`. No imputation has been applied.",
        "",
        "## All Features",
        "",
        "| Feature | Missing % |",
        "|---|---|",
    ]
    lines += [f"| {col} | {pct} |" for col, pct in profile]

    lines += [
        "",
        f"## Top {top_n} Most Missing Features — Treatment Decisions",
        "",
        "| Rank | Feature | Missing % | Treatment | Reason |",
        "|---|---|---|---|---|",
    ]
    for rank, (col, pct) in enumerate(profile[:top_n], start=1):
        treatment, reason = categorize_treatment(col)
        lines.append(f"| {rank} | {col} | {pct} | {treatment} | {reason} |")

    key_categories = [
        (col, pct) for col, pct in profile
        if col.startswith("EXT_SOURCE_") or col.startswith("bureau_") or col.startswith("prev_")
    ]
    lines += [
        "",
        "## EXT_SOURCE / bureau_ / prev_ Feature Treatments",
        "",
        "These categories have explicit treatment rules regardless of top-N rank "
        "(none currently fall in the top "
        f"{top_n} by raw percentage — they're outranked by sparse building/apartment columns).",
        "",
        "| Feature | Missing % | Treatment | Reason |",
        "|---|---|---|---|",
    ]
    for col, pct in key_categories:
        treatment, reason = categorize_treatment(col)
        lines.append(f"| {col} | {pct} | {treatment} | {reason} |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote missing value report to %s", output_path)


def create_dev_holdout_split(con, holdout_fraction=HOLDOUT_FRACTION, random_state=RANDOM_STATE):
    """Split SK_ID_CURR into dev/holdout via sklearn's train_test_split.

    Only SK_ID_CURR and TARGET (2 of ~148 columns) are materialized into
    pandas — just enough for train_test_split's API — then registered back
    with DuckDB so the full feature table is filtered and written to
    Parquet without ever loading it into Python memory.

    ORDER BY SK_ID_CURR is required, not cosmetic: train_test_split's
    shuffle operates on row *position*, not on SK_ID_CURR values. DuckDB
    does not guarantee a stable row order for the same query across
    separate process runs (verified directly -- rebuilding
    features_naive.parquet after an unrelated column addition changed the
    materialized row order), so without an explicit deterministic sort
    here, re-running the upstream feature pipeline could silently reshuffle
    which applicants land in dev vs. holdout even with random_state fixed.
    """
    ids = con.execute("SELECT SK_ID_CURR, TARGET FROM features_clean ORDER BY SK_ID_CURR").fetchdf()

    dev_ids, holdout_ids = train_test_split(
        ids,
        test_size=holdout_fraction,
        stratify=ids["TARGET"],
        random_state=random_state,
    )

    con.register("dev_ids_tbl", dev_ids[["SK_ID_CURR"]])
    con.register("holdout_ids_tbl", holdout_ids[["SK_ID_CURR"]])

    logger.info(
        "Split %d applicants into %d dev / %d holdout (holdout_fraction=%.2f, random_state=%d)",
        len(ids), len(dev_ids), len(holdout_ids), holdout_fraction, random_state,
    )


def save_split(con, dev_path=DEV_PATH, holdout_path=HOLDOUT_PATH):
    """Write dev/holdout Parquet files directly from DuckDB via the registered ID tables."""
    os.makedirs(os.path.dirname(dev_path), exist_ok=True)

    con.execute(
        f"""
        COPY (
            SELECT f.* FROM features_clean f
            JOIN dev_ids_tbl d ON f.SK_ID_CURR = d.SK_ID_CURR
        ) TO '{dev_path}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT f.* FROM features_clean f
            JOIN holdout_ids_tbl h ON f.SK_ID_CURR = h.SK_ID_CURR
        ) TO '{holdout_path}' (FORMAT PARQUET)
        """
    )
    logger.info("Saved dev split to %s", dev_path)
    logger.info("Saved holdout split to %s", holdout_path)


def validate_split(con, dev_path=DEV_PATH, holdout_path=HOLDOUT_PATH):
    """Confirm the split is disjoint, complete, and target-balanced. Hard-fails otherwise."""
    con.execute(f"CREATE OR REPLACE VIEW dev AS SELECT * FROM read_parquet('{dev_path}')")
    con.execute(f"CREATE OR REPLACE VIEW holdout AS SELECT * FROM read_parquet('{holdout_path}')")

    total = con.execute("SELECT COUNT(*) FROM features_clean").fetchone()[0]
    dev_rows, dev_defaults = con.execute(
        "SELECT COUNT(*), SUM(TARGET) FROM dev"
    ).fetchone()
    holdout_rows, holdout_defaults = con.execute(
        "SELECT COUNT(*), SUM(TARGET) FROM holdout"
    ).fetchone()
    overlap = con.execute(
        "SELECT COUNT(*) FROM dev d JOIN holdout h ON d.SK_ID_CURR = h.SK_ID_CURR"
    ).fetchone()[0]

    assert dev_rows + holdout_rows == total, (
        f"Row counts don't sum: dev={dev_rows} + holdout={holdout_rows} != total={total}"
    )
    assert overlap == 0, f"{overlap} applicants appear in both dev and holdout"

    dev_default_rate = dev_defaults / dev_rows * 100
    holdout_default_rate = holdout_defaults / holdout_rows * 100
    assert abs(dev_default_rate - holdout_default_rate) < 1.0, (
        f"Default rate diverges beyond tolerance: dev={dev_default_rate:.2f}%, "
        f"holdout={holdout_default_rate:.2f}%"
    )

    dev_cols = len(con.execute("DESCRIBE dev").fetchall())
    holdout_cols = len(con.execute("DESCRIBE holdout").fetchall())

    logger.info("Validation passed: disjoint, complete, target-balanced")
    print(f"Dev shape: ({dev_rows}, {dev_cols})")
    print(f"Holdout shape: ({holdout_rows}, {holdout_cols})")
    print(f"Dev default rate: {dev_default_rate:.4f}%")
    print(f"Holdout default rate: {holdout_default_rate:.4f}%")


def main():
    con = get_connection()

    load_features(con)
    clean_sentinel_values(con)

    profile = profile_missing_values(con)
    write_missing_value_report(profile)

    create_dev_holdout_split(con)
    save_split(con)
    validate_split(con)


if __name__ == "__main__":
    main()

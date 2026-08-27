"""
Phase 1: Data profiling pipeline.

Loads raw Home Credit CSVs as DuckDB views and computes basic profiling
metrics for application_train. No feature engineering or modeling here —
this phase only answers "what does the raw data look like?".
"""

import json
import logging
import os

import duckdb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Resolved from this file's location (not cwd) so the script works the same
# whether it's run as `python src/profile_data.py` or from any other directory.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Centralized so paths are consistent across functions and easy to override in tests.
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
REPORTS_PATH = os.path.join(PROJECT_ROOT, "reports", "metrics.json")

# Table name -> source CSV filename, in one place so adding a new raw table
# later only requires one edit instead of touching every function.
TABLE_FILES = {
    "application_train": "application_train.csv",
    "bureau": "bureau.csv",
    "previous_application": "previous_application.csv",
}


def get_connection():
    """Create an in-memory DuckDB connection.

    In-memory is sufficient for profiling: we never need the database to
    persist between runs, and views over CSVs are cheap to recreate.
    """
    return duckdb.connect(database=":memory:")


def register_view(con, table_name, csv_path):
    """Register a DuckDB view over a CSV file.

    Fails fast with a clear message if the file is missing, rather than
    letting DuckDB raise a generic IO error later when the view is queried.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Expected raw data file for '{table_name}' at '{csv_path}' but it was not found. "
            f"Place the Home Credit CSV files under '{DATA_DIR}/' before running this script."
        )

    # read_csv_auto infers schema and only reads the file when the view is
    # queried, so registering a view has no meaningful load cost up front.
    con.execute(
        f"CREATE OR REPLACE VIEW {table_name} AS "
        f"SELECT * FROM read_csv_auto('{csv_path}')"
    )
    logger.info("Registered view '%s' from %s", table_name, csv_path)


def load_all_views(con, data_dir=DATA_DIR):
    """Register a view for every raw table in TABLE_FILES."""
    for table_name, filename in TABLE_FILES.items():
        csv_path = os.path.join(data_dir, filename)
        register_view(con, table_name, csv_path)


def profile_application_train(con):
    """Compute total rows, default count, and default percentage.

    TARGET is the Home Credit label column: 1 = defaulted, 0 = repaid.
    Aggregation is done in SQL so DuckDB does the scan, not pandas.
    """
    result = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            SUM(CASE WHEN TARGET = 1 THEN 1 ELSE 0 END) AS num_defaults
        FROM application_train
        """
    ).fetchone()

    total_rows, num_defaults = result
    default_pct = (num_defaults / total_rows * 100) if total_rows else 0.0

    return {
        "total_rows": total_rows,
        "num_defaults": num_defaults,
        "default_pct": round(default_pct, 4),
    }


def save_metrics(metrics, output_path=REPORTS_PATH):
    """Write metrics to a JSON file, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to %s", output_path)


def main():
    con = get_connection()
    load_all_views(con, DATA_DIR)

    logger.info("Profiling application_train...")
    metrics = profile_application_train(con)
    save_metrics(metrics, REPORTS_PATH)

    print("Data profiling complete. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

"""
Phase 4: Leakage audit experiment.

Trains an identically-configured XGBoost model on two feature tables built
from the same development set:

- "naive"    -> data/processed/dev.parquet (Phase 3's locked dev split,
               built from features_naive.parquet: bureau/previous_application
               aggregated with no temporal filtering)
- "filtered" -> data/processed/features_filtered.parquet, restricted to the
               same dev SK_ID_CURR set and given the same DAYS_EMPLOYED
               sentinel cleaning, so the only structural difference from
               "naive" is that bureau/previous_application rows were
               restricted to DAYS_CREDIT <= 0 / DAYS_DECISION <= 0 before
               aggregating.

This is a reliability audit, not a performance optimization: hyperparameters
are fixed and identical across both runs, and holdout.parquet is never
touched. If "naive" scores materially higher than "filtered", that is
evidence of temporal leakage in the naive feature pipeline.
"""

import json
import logging
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from data_quality import DAYS_EMPLOYED_SENTINEL, DEV_PATH
from features import FEATURES_FILTERED_PATH
from profile_data import PROJECT_ROOT, get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "leakage_results.json")

MODEL_PARAMS = dict(
    tree_method="hist",
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="aucpr",
    random_state=42,
)
N_SPLITS = 3
CV_RANDOM_STATE = 42


def load_naive_dev(con, dev_path=DEV_PATH):
    """Load the Phase 3 locked dev split as-is (already sentinel-cleaned)."""
    return con.execute(
        f"SELECT * FROM read_parquet('{dev_path}') ORDER BY SK_ID_CURR"
    ).fetchdf()


def load_filtered_dev(con, filtered_path=FEATURES_FILTERED_PATH, dev_path=DEV_PATH):
    """Load features_filtered.parquet restricted to the same dev applicants.

    features_filtered.parquet has not been through Phase 3's cleaning or
    split, so both are applied here: the same DAYS_EMPLOYED sentinel fix
    (so the two arms differ only in the temporal filter, not in whether
    the sentinel was cleaned), and an inner join against dev.parquet's
    SK_ID_CURR (never holdout.parquet) to restrict to the identical
    applicant set used for the naive arm.
    """
    con.execute(f"CREATE OR REPLACE VIEW dev_ids AS SELECT SK_ID_CURR FROM read_parquet('{dev_path}')")
    con.execute(f"CREATE OR REPLACE VIEW features_filtered_raw AS SELECT * FROM read_parquet('{filtered_path}')")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW features_filtered_clean AS
        SELECT
            * EXCLUDE (DAYS_EMPLOYED),
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN 1 ELSE 0 END
                AS days_employed_anomaly,
            CASE WHEN DAYS_EMPLOYED = {DAYS_EMPLOYED_SENTINEL} THEN NULL ELSE DAYS_EMPLOYED END
                AS DAYS_EMPLOYED
        FROM features_filtered_raw
        """
    )
    return con.execute(
        """
        SELECT f.* FROM features_filtered_clean f
        JOIN dev_ids d ON f.SK_ID_CURR = d.SK_ID_CURR
        ORDER BY f.SK_ID_CURR
        """
    ).fetchdf()


def select_model_matrix(df, feature_columns=None):
    """Split a loaded dev DataFrame into (X, y, feature_columns).

    X is restricted to numeric/boolean columns, excluding SK_ID_CURR and
    TARGET. String columns (raw application_train categoricals) are
    dropped rather than encoded -- this is a lightweight leakage-audit
    model, not a performance-tuned one, and categorical encoding choice
    would be identical noise across both arms regardless. NaN values are
    left as-is; tree_method="hist" handles them natively, consistent with
    Phase 3's decision not to impute yet.
    """
    if feature_columns is None:
        feature_columns = [
            c for c in df.select_dtypes(include=["number", "bool"]).columns
            if c not in ("SK_ID_CURR", "TARGET")
        ]
    X = df[feature_columns]
    y = df["TARGET"].astype(int)
    return X, y, feature_columns


def evaluate_model(X, y):
    """Run stratified 3-fold CV with the fixed model config. Returns ROC-AUC/PR-AUC mean+std."""
    model = XGBClassifier(**MODEL_PARAMS)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CV_RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "pr_auc_mean": float(np.mean(scores["test_average_precision"])),
        "pr_auc_std": float(np.std(scores["test_average_precision"])),
    }


def main():
    con = get_connection()

    logger.info("Loading naive dev set (data/processed/dev.parquet)")
    naive_df = load_naive_dev(con)
    logger.info("Loading filtered dev set (features_filtered.parquet, restricted to dev IDs)")
    filtered_df = load_filtered_dev(con)

    assert len(naive_df) == len(filtered_df), (
        f"Dev row count mismatch: naive={len(naive_df)}, filtered={len(filtered_df)}"
    )
    assert list(naive_df["SK_ID_CURR"]) == list(filtered_df["SK_ID_CURR"]), (
        "naive/filtered dev sets are not row-aligned by SK_ID_CURR -- CV folds would not be comparable"
    )

    X_naive, y_naive, feature_columns = select_model_matrix(naive_df)
    X_filtered, y_filtered, _ = select_model_matrix(filtered_df, feature_columns=feature_columns)

    logger.info("Evaluating naive (%d rows, %d features)", len(X_naive), len(feature_columns))
    naive_results = evaluate_model(X_naive, y_naive)
    logger.info("Evaluating filtered (%d rows, %d features)", len(X_filtered), len(feature_columns))
    filtered_results = evaluate_model(X_filtered, y_filtered)

    results = {
        "naive": naive_results,
        "filtered": filtered_results,
        "config": {
            "model_params": MODEL_PARAMS,
            "cv_splits": N_SPLITS,
            "cv_random_state": CV_RANDOM_STATE,
            "n_features": len(feature_columns),
            "n_rows": len(X_naive),
            "data_source": "development set only (data/processed/dev.parquet applicant IDs); holdout.parquet not used",
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", RESULTS_PATH)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

"""
Phase 5: Logistic regression baseline.

Trains an un-tuned logistic regression on development data only, as a
trustworthy performance floor later models are measured against. All
preprocessing (imputation, scaling, one-hot encoding) lives inside a
single sklearn Pipeline, so cross_validate fits each fold's imputation
medians, scaling statistics, and known categories on that fold's training
split only -- never on its own held-out rows, and never on holdout.parquet.
"""

import json
import logging
import os

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_quality import DEV_PATH
from profile_data import PROJECT_ROOT, get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "baseline_results.json")

N_SPLITS = 5
RANDOM_STATE = 42


def load_dev_data(dev_path=DEV_PATH):
    """Load dev.parquet only -- holdout.parquet must never be read here."""
    con = get_connection()
    df = con.execute(f"SELECT * FROM read_parquet('{dev_path}')").fetchdf()
    con.close()
    return df


def split_features_target(df):
    """TARGET -> y, SK_ID_CURR -> dropped (identifier only), rest -> X."""
    y = df["TARGET"].astype(int)
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    return X, y


def build_pipeline(X):
    """Build the preprocessing + model Pipeline.

    Columns are split by dtype: `number` -> numeric branch, everything
    else -> categorical branch. This deliberately isn't an explicit
    bool/object include-list -- EMERGENCYSTATE_MODE loads as pandas'
    nullable `boolean` dtype (not numpy bool), which `number` correctly
    excludes, routing it to OneHotEncoder (verified to handle its
    True/False/NA values as three clean categories) rather than into
    SimpleImputer+StandardScaler, which nullable boolean was never tested
    against here.
    """
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    return pipeline, numeric_features, categorical_features


def evaluate(pipeline, X, y):
    """Stratified 5-fold CV, reporting ROC-AUC and PR-AUC mean/std."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "pr_auc_mean": float(np.mean(scores["test_average_precision"])),
        "pr_auc_std": float(np.std(scores["test_average_precision"])),
    }


def main():
    logger.info("Loading development data from %s", DEV_PATH)
    df = load_dev_data()
    X, y = split_features_target(df)

    pipeline, numeric_features, categorical_features = build_pipeline(X)
    logger.info(
        "Built pipeline: %d numeric + %d categorical = %d input features",
        len(numeric_features), len(categorical_features), X.shape[1],
    )

    logger.info("Running stratified %d-fold CV", N_SPLITS)
    metrics = evaluate(pipeline, X, y)

    results = {
        "model": "LogisticRegression",
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "metrics": metrics,
        "config": {
            "model_params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "cv_splits": N_SPLITS,
            "cv_random_state": RANDOM_STATE,
            "n_rows": int(len(X)),
            "data_source": "data/processed/dev.parquet (holdout.parquet not used)",
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", RESULTS_PATH)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

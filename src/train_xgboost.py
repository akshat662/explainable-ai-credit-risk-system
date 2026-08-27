"""
Phase 6: XGBoost nonlinear candidate model.

Trains an untuned XGBClassifier on development data only, under the same
stratified 5-fold CV protocol as the Phase 5 logistic regression baseline,
so the two are directly comparable. Categorical columns are cast to
pandas 'category' dtype and passed via enable_categorical=True rather than
one-hot encoded or dropped -- this keeps the identical 146-feature set the
baseline used (not a subset), relying on XGBoost's native categorical-
split and missing-value handling instead of an imputer/encoder Pipeline.
This is a data-handling necessity, not a hyperparameter tune: the eight
model hyperparameters below are exactly as specified and untouched.
"""

import json
import logging
import os

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

from data_quality import DEV_PATH
from profile_data import PROJECT_ROOT, get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "xgboost_results.json")
COMPARISON_PATH = os.path.join(PROJECT_ROOT, "reports", "model_comparison.json")
BASELINE_RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "baseline_results.json")

MODEL_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    eval_metric="aucpr",
    random_state=42,
)
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


def prepare_categoricals(X):
    """Cast non-numeric columns to pandas 'category' dtype for XGBoost.

    Nullable-boolean columns (e.g. EMERGENCYSTATE_MODE) are mapped to the
    strings "True"/"False" first -- XGBoost's categorical encoder requires
    category values to be a single type (string or int) and rejects
    pandas' boolean dtype outright (verified directly). .map() leaves
    missing entries as NaN rather than turning them into a literal
    "<NA>" category, so real missingness is preserved.
    """
    X = X.copy()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()
    for col in categorical_features:
        if str(X[col].dtype) == "boolean":
            X[col] = X[col].map({True: "True", False: "False"})
        X[col] = X[col].astype("category")
    numeric_features = [c for c in X.columns if c not in categorical_features]
    return X, numeric_features, categorical_features


def evaluate(X, y):
    """Stratified 5-fold CV, reporting ROC-AUC and PR-AUC mean/std."""
    model = XGBClassifier(**MODEL_PARAMS, enable_categorical=True)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(model, X, y, cv=cv, scoring=["roc_auc", "average_precision"])
    return {
        "roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "roc_auc_std": float(np.std(scores["test_roc_auc"])),
        "pr_auc_mean": float(np.mean(scores["test_average_precision"])),
        "pr_auc_std": float(np.std(scores["test_average_precision"])),
    }


def build_comparison(xgboost_results, baseline_path=BASELINE_RESULTS_PATH, output_path=COMPARISON_PATH):
    """Write a side-by-side comparison of the logistic baseline and XGBoost."""
    if not os.path.isfile(baseline_path):
        raise FileNotFoundError(
            f"Expected baseline results at '{baseline_path}' but it was not found. "
            f"Run src/train_baseline.py first."
        )
    with open(baseline_path) as f:
        baseline_results = json.load(f)

    comparison = {
        "logistic_regression": {
            "feature_count": baseline_results["feature_count"],
            "metrics": baseline_results["metrics"],
        },
        "xgboost": {
            "feature_count": xgboost_results["feature_count"],
            "metrics": xgboost_results["metrics"],
        },
        "delta_xgboost_minus_logistic": {
            "roc_auc_mean": (
                xgboost_results["metrics"]["roc_auc_mean"] - baseline_results["metrics"]["roc_auc_mean"]
            ),
            "pr_auc_mean": (
                xgboost_results["metrics"]["pr_auc_mean"] - baseline_results["metrics"]["pr_auc_mean"]
            ),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved model comparison to %s", output_path)
    return comparison


def main():
    logger.info("Loading development data from %s", DEV_PATH)
    df = load_dev_data()
    X, y = split_features_target(df)
    X, numeric_features, categorical_features = prepare_categoricals(X)

    logger.info(
        "Prepared %d features (%d numeric, %d categorical) for XGBoost",
        X.shape[1], len(numeric_features), len(categorical_features),
    )

    logger.info("Running stratified %d-fold CV", N_SPLITS)
    metrics = evaluate(X, y)

    results = {
        "model": "XGBClassifier",
        "feature_count": int(X.shape[1]),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "metrics": metrics,
        "config": {
            "model_params": {**MODEL_PARAMS, "enable_categorical": True},
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

    build_comparison(results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

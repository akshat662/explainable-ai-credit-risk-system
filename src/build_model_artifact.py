"""
Phase 10: One-time frozen model artifact build.

Trains the SAME frozen XGBoost exactly once (reusing
evaluate_holdout_calibration.train_final_model and train_xgboost.MODEL_PARAMS
unchanged -- no new training logic, no tuning, no CV here) and persists it so
the Streamlit app (src/inference.py, app/app.py) never retrains on startup.

This script is deliberately separate from the application. It is a build
step, run once by a developer, not by end users and not by the app itself:

    python src/build_model_artifact.py

Produces two files:
  - models/xgboost_frozen.json   -- the trained booster, XGBoost's native
                                     format (model.save_model()).
  - models/feature_schema.json   -- the frozen 147-column model-input
                                     contract: exact column order, which
                                     columns are categorical and their exact
                                     training-time category sets (required
                                     for XGBoost's native categorical split
                                     encoding to recode inference-time values
                                     correctly -- verified empirically that a
                                     single-row categorical column must carry
                                     an explicit, non-empty `categories=`
                                     list, since a lone NaN value gives pandas
                                     zero inferred categories and XGBoost
                                     hard-errors on that), and a per-column
                                     "template" default (dev median for
                                     numeric columns, dev mode for categorical
                                     columns) used by inference.py to fill any
                                     of the 147 model features the interactive
                                     UI does not collect.

Only dev.parquet is read. The untouched final-evaluation set is never
referenced -- verified below by a static self-check of this module's own
source, the same pattern evaluate_final_holdout.py uses for its
threshold-sweep guard.
"""

import inspect
import json
import logging
import os

from evaluate_holdout_calibration import train_final_model
from profile_data import PROJECT_ROOT
from train_xgboost import MODEL_PARAMS, load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_frozen.json")
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")

EXPECTED_N_FEATURES = 147


FORBIDDEN_HOLDOUT_IDENTIFIERS = ("HOLDOUT_PATH", "holdout.parquet", "load_holdout_data", "HOLDOUT_FRACTION")


def verify_no_holdout_reference():
    """Static self-check: this file must never import or use anything that
    reads holdout data -- checked by identifier, not the prose word
    "holdout" (which legitimately appears in comments/docstrings explaining
    why holdout is out of scope here). Mirrors evaluate_final_holdout.py's
    verify_no_holdout_threshold_search -- a repo-hygiene assertion about the
    source itself, not just its output. This function's own source is
    excluded from the scan, since it necessarily names the forbidden
    identifiers in order to check for them.
    """
    full_source = inspect.getsource(inspect.getmodule(verify_no_holdout_reference))
    own_source = inspect.getsource(verify_no_holdout_reference)
    remainder_lines = [
        line for line in full_source.replace(own_source, "").splitlines()
        if not line.startswith("FORBIDDEN_HOLDOUT_IDENTIFIERS")
    ]
    remainder = "\n".join(remainder_lines)
    for identifier in FORBIDDEN_HOLDOUT_IDENTIFIERS:
        assert identifier not in remainder, (
            f"build_model_artifact.py must never reference '{identifier}' -- "
            f"this artifact is built from dev.parquet only."
        )
    logger.info("Static check passed: no holdout-reading identifier in build_model_artifact.py.")


def _to_native(value):
    """Convert one dev-column statistic (numpy/pandas scalar) to a JSON-safe
    native Python value, preserving None for missing/undefined statistics."""
    import numpy as np
    import pandas as pd

    if value is None or (isinstance(value, float) and value != value):  # NaN
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build_template_defaults(X_dev, numeric_features, categorical_features):
    """Per-column fill value for features the interactive UI does not collect.

    Numeric -> dev median (skipping NaN). Categorical -> dev mode (most
    frequent non-null value). Computed once from dev.parquet only -- these
    are aggregate dataset statistics, not individual applicant records, and
    are the same kind of quantity already reported in reports/missing_values.md.

    This is a documented demo simplification, not a claim that an unexposed
    field's true value equals the dev median/mode for any given applicant --
    see the "Demo limitations" section of the Phase 10 documentation.
    """
    defaults = {}
    for col in numeric_features:
        defaults[col] = _to_native(X_dev[col].median(skipna=True))
    for col in categorical_features:
        mode = X_dev[col].mode(dropna=True)
        defaults[col] = _to_native(mode.iloc[0]) if len(mode) > 0 else None
    return defaults


def build_categories(X_dev, categorical_features):
    """Exact training-time category set per categorical column, in order.

    Required so inference.py can construct a single-row DataFrame whose
    pandas 'category' dtype matches training exactly -- XGBoost's native
    categorical encoder recodes by category VALUE (verified empirically,
    not order/position-dependent), but a local single-row category dtype
    with zero inferred categories (e.g. a lone NaN with no explicit
    `categories=` list) hard-errors at predict time. Freezing the real
    training-time category list up front avoids that failure mode entirely
    and guarantees every UI dropdown only ever offers values the model
    actually saw during training.
    """
    return {col: [str(c) for c in X_dev[col].cat.categories.tolist()] for col in categorical_features}


def main():
    verify_no_holdout_reference()

    logger.info("Loading dev.parquet and training the frozen XGBoost model (MODEL_PARAMS unchanged)")
    dev_df = load_dev_data()
    X_dev, y_dev = split_features_target(dev_df)
    X_dev, numeric_features, categorical_features = prepare_categoricals(X_dev)

    assert X_dev.shape[1] == EXPECTED_N_FEATURES, (
        f"expected {EXPECTED_N_FEATURES} model features, got {X_dev.shape[1]}"
    )

    model = train_final_model(X_dev, y_dev)

    fitted_params = model.get_params()
    for key, expected in MODEL_PARAMS.items():
        assert fitted_params.get(key) == expected, (
            f"fitted model param '{key}'={fitted_params.get(key)!r} does not match "
            f"frozen MODEL_PARAMS['{key}']={expected!r} -- refusing to save a "
            f"model artifact that does not match the frozen configuration."
        )
    logger.info("Confirmed fitted model params match train_xgboost.MODEL_PARAMS exactly")

    logger.info("Computing per-column template defaults (dev median/mode) for unexposed features")
    template_defaults = build_template_defaults(X_dev, numeric_features, categorical_features)
    categories = build_categories(X_dev, categorical_features)

    os.makedirs(MODELS_DIR, exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info("Saved frozen model to %s", MODEL_PATH)

    import shap
    import xgboost

    schema = {
        "feature_names": list(X_dev.columns),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "categories": categories,
        "template_defaults": template_defaults,
        "model_params": {k: v for k, v in MODEL_PARAMS.items()},
        "n_dev_rows_trained_on": int(len(X_dev)),
        "built_with": {"xgboost": xgboost.__version__, "shap": shap.__version__},
    }
    with open(SCHEMA_PATH, "w") as f:
        json.dump(schema, f, indent=2)
    logger.info("Saved feature schema to %s", SCHEMA_PATH)

    print(
        f"\nPhase 10 model artifact built: {EXPECTED_N_FEATURES} features, "
        f"{len(categorical_features)} categorical, {len(numeric_features)} numeric, "
        f"trained on {len(X_dev):,} dev rows. The untouched final-evaluation set was never read."
    )


if __name__ == "__main__":
    main()

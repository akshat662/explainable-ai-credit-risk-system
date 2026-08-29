"""
Phase 10: Production-style inference for the frozen XGBoost credit-risk model.

This module is the ONLY thing the Streamlit app (app/app.py) imports for
prediction and explanation. It does not train, cross-validate, sweep
thresholds, fit a calibrator, or read holdout data -- it loads the artifact
already built once by src/build_model_artifact.py and turns a curated set of
applicant inputs into the exact 147-column vector the frozen model expects.

FEATURE RECONSTRUCTION, HONESTLY STATED: the frozen pipeline's 147 features
are produced by a relational SQL aggregation over bureau.csv and
previous_application.csv (src/features.py) -- there is no realistic way for
an interactive form to re-run that aggregation for a hypothetical applicant
who has no rows in those tables. This module does not try to fake that.
Instead:
  - Bureau/previous-application fields the UI exposes (e.g.
    bureau_total_loans, prev_avg_term) are collected as the SAME
    already-aggregated summary numbers the model consumes -- the user
    supplies the aggregate directly, matching the model's actual input
    contract, rather than synthetic per-loan records being fabricated and
    aggregated behind the scenes.
  - The six ratio features (credit_income_ratio, annuity_income_ratio,
    credit_annuity_ratio, goods_credit_ratio, income_per_person,
    debt_income_ratio) are recomputed here with the exact same formulas
    src/features.py's SQL uses (see RATIO_FEATURES below) -- not
    approximated.
  - DAYS_EMPLOYED's sentinel handling reuses
    data_quality.DAYS_EMPLOYED_SENTINEL and reproduces
    data_quality.clean_sentinel_values()'s CASE WHEN logic exactly.
  - Every one of the 147 model features NOT covered by the curated UI
    (mostly building/apartment descriptors, FLAG_DOCUMENT_2..21, and other
    low-signal administrative fields -- see
    reports/phase10_deployment_report.md) is filled from a frozen
    per-column dev-set median/mode template computed once in
    build_model_artifact.py. This is a disclosed demo simplification, not a
    claim that it reflects any individual applicant -- surfaced explicitly
    in the app's "Model information" section.

CATEGORICAL ENCODING, VERIFIED (not assumed): XGBoost's native
`enable_categorical` support (used throughout this project, see
train_xgboost.prepare_categoricals) recodes categorical inputs by VALUE
against the training-time category set, not by pandas' local per-DataFrame
category codes -- confirmed empirically. However, a single-row DataFrame
whose categorical value is missing has ZERO locally-inferred categories
(pandas can't infer categories from one NaN), and XGBoost hard-errors on a
zero-category column ("Categorical feature must have at least one
category"). build_feature_row() below always casts categorical columns with
an explicit `categories=` list taken from the frozen schema, which avoids
that failure mode and guarantees every category value used at inference was
actually seen during training.
"""

import json
import os

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from data_quality import DAYS_EMPLOYED_SENTINEL
from decision_threshold import LGD, MARGIN
from evaluate_final_holdout import FROZEN_THRESHOLD
from profile_data import PROJECT_ROOT
from shap_explainability import display_name, explain_applicant
from train_xgboost import MODEL_PARAMS

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_frozen.json")
SCHEMA_PATH = os.path.join(MODELS_DIR, "feature_schema.json")

EXPECTED_N_FEATURES = 147

# Same formulas as src/features.py's build_applicant_features() SQL --
# division-by-zero guarded the same way NULLIF(denominator, 0) is (returns
# None/NaN, never inf or an error).
RATIO_FEATURES = (
    "credit_income_ratio", "annuity_income_ratio", "credit_annuity_ratio",
    "goods_credit_ratio", "income_per_person", "debt_income_ratio",
)


class ModelArtifactMissing(RuntimeError):
    """Raised when models/xgboost_frozen.json or feature_schema.json is absent."""


def _require_artifact(path):
    if not os.path.isfile(path):
        raise ModelArtifactMissing(
            f"'{path}' not found. Run `python src/build_model_artifact.py` once "
            f"(from the project root, with dev.parquet present) before starting the app."
        )


def load_schema():
    """Load the frozen 147-feature schema built by build_model_artifact.py."""
    _require_artifact(SCHEMA_PATH)
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)

    assert len(schema["feature_names"]) == EXPECTED_N_FEATURES, (
        f"feature_schema.json has {len(schema['feature_names'])} features, "
        f"expected {EXPECTED_N_FEATURES} -- artifact is stale or corrupt, rebuild it."
    )
    assert schema["model_params"] == MODEL_PARAMS, (
        "feature_schema.json's recorded model_params no longer match "
        "train_xgboost.MODEL_PARAMS -- the model artifact was built against a "
        "different configuration than the one currently frozen in code. "
        "Re-run src/build_model_artifact.py."
    )
    return schema


def load_model():
    """Load the frozen XGBoost booster. Never fits/trains anything."""
    _require_artifact(MODEL_PATH)
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def load_explainer(model):
    """shap.TreeExplainer over the frozen, already-loaded model (see
    src/shap_explainability.py for the empirical log-odds/raw-margin
    verification this project relies on -- unchanged here)."""
    return shap.TreeExplainer(model)


def _safe_div(numerator, denominator):
    """NULLIF(denominator, 0) semantics: undefined ratio -> None, never inf/error."""
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def resolve_days_employed(is_currently_employed, years_employed):
    """Reproduces data_quality.clean_sentinel_values()'s CASE WHEN exactly:
    construct the same raw DAYS_EMPLOYED encoding Home Credit uses (365243
    for Pensioner/Unemployed applicants, via the imported
    DAYS_EMPLOYED_SENTINEL constant -- not a new literal), then apply the
    identical sentinel -> NaN + anomaly-flag transform used at training
    time. Returns (days_employed, days_employed_anomaly)."""
    if not is_currently_employed:
        raw_days_employed = DAYS_EMPLOYED_SENTINEL
    else:
        raw_days_employed = -round(years_employed * 365)

    if raw_days_employed == DAYS_EMPLOYED_SENTINEL:
        return None, 1
    return raw_days_employed, 0


def build_feature_row(user_inputs, schema):
    """Construct the single-row, 147-column model-input DataFrame.

    Starts from the frozen dev median/mode template (schema['template_defaults']),
    overrides with the fields the UI actually collected (user_inputs), then
    recomputes the six ratio features from the (possibly just-overridden)
    raw values -- exactly mirroring src/features.py's SQL, not a new
    approximation.
    """
    row = dict(schema["template_defaults"])
    row.update(user_inputs)

    debt = row.get("bureau_total_debt")
    debt_clipped = None if debt is None else max(debt, 0.0)

    row["credit_income_ratio"] = _safe_div(row.get("AMT_CREDIT"), row.get("AMT_INCOME_TOTAL"))
    row["annuity_income_ratio"] = _safe_div(row.get("AMT_ANNUITY"), row.get("AMT_INCOME_TOTAL"))
    row["credit_annuity_ratio"] = _safe_div(row.get("AMT_CREDIT"), row.get("AMT_ANNUITY"))
    row["goods_credit_ratio"] = _safe_div(row.get("AMT_GOODS_PRICE"), row.get("AMT_CREDIT"))
    row["income_per_person"] = _safe_div(row.get("AMT_INCOME_TOTAL"), row.get("CNT_FAM_MEMBERS"))
    row["debt_income_ratio"] = _safe_div(debt_clipped, row.get("AMT_INCOME_TOTAL"))

    ordered = {col: row.get(col) for col in schema["feature_names"]}
    df = pd.DataFrame([ordered], columns=schema["feature_names"])

    for col in schema["categorical_features"]:
        df[col] = pd.Categorical(df[col], categories=schema["categories"][col])
    for col in schema["numeric_features"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    assert list(df.columns) == schema["feature_names"], (
        "build_feature_row: column order drifted from the frozen schema"
    )
    assert df.shape == (1, EXPECTED_N_FEATURES), (
        f"build_feature_row: expected shape (1, {EXPECTED_N_FEATURES}), got {df.shape}"
    )
    return df


def predict_probability(model, row_df):
    """Raw XGBoost default-probability for one applicant row. Frozen
    probability source (Phase 7.8): never calibrated (Platt/isotonic)."""
    probability = float(model.predict_proba(row_df)[:, 1][0])
    assert 0.0 <= probability <= 1.0, f"predicted probability out of range: {probability}"
    return probability


def decide(probability, threshold=FROZEN_THRESHOLD):
    """Frozen 0.110 decision rule (Phase 8): reject if p_default >= threshold."""
    return "REJECT" if probability >= threshold else "APPROVE"


def distance_to_threshold_pp(probability, threshold=FROZEN_THRESHOLD):
    """Signed distance from the policy threshold, in percentage points."""
    return (probability - threshold) * 100


def explain(model, explainer, row_df, schema, applicant_label="applicant"):
    """Decision-aware local SHAP explanation, via the unchanged Phase 9
    explain_applicant() -- probability/decision computed independently of
    SHAP, SHAP values reported in log-odds space (see shap_explainability.py)."""
    return explain_applicant(
        applicant_label, row_df, model, explainer, schema["feature_names"], threshold=FROZEN_THRESHOLD,
    )


def business_context():
    """Static economic constants used to derive (not re-derive) the frozen
    threshold -- for display only, never used in prediction/decisioning here."""
    return {"lgd": LGD, "margin": MARGIN, "threshold": FROZEN_THRESHOLD}

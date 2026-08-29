"""
Phase 10 tests: src/inference.py against the frozen model artifact.

These are lightweight regression/contract tests, not a model-quality
evaluation -- they check that the deployment layer faithfully reproduces
the frozen Phase 6-9 configuration and never touches holdout data. They
require models/xgboost_frozen.json and models/feature_schema.json to
already exist (built once via `python src/build_model_artifact.py`).
"""

import inspect
import os

import pytest

import inference as inf
from decision_threshold import LGD, MARGIN
from evaluate_final_holdout import FROZEN_THRESHOLD
from train_xgboost import MODEL_PARAMS

pytestmark = pytest.mark.skipif(
    not os.path.isfile(inf.MODEL_PATH) or not os.path.isfile(inf.SCHEMA_PATH),
    reason="model artifact not built -- run `python src/build_model_artifact.py` first",
)


@pytest.fixture(scope="module")
def schema():
    return inf.load_schema()


@pytest.fixture(scope="module")
def model():
    return inf.load_model()


@pytest.fixture(scope="module")
def explainer(model):
    return inf.load_explainer(model)


# --- frozen configuration is unchanged and correctly propagated ------------

def test_frozen_threshold_is_exactly_0_110():
    assert FROZEN_THRESHOLD == 0.110
    assert inf.FROZEN_THRESHOLD == 0.110


def test_model_params_not_redefined():
    assert MODEL_PARAMS == {
        "n_estimators": 300,
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "eval_metric": "aucpr",
        "random_state": 42,
    }


def test_schema_model_params_match_frozen_source(schema):
    assert schema["model_params"] == MODEL_PARAMS


def test_business_context_matches_frozen_constants():
    ctx = inf.business_context()
    assert ctx == {"lgd": LGD, "margin": MARGIN, "threshold": FROZEN_THRESHOLD}


# --- schema contract ---------------------------------------------------

def test_schema_has_exactly_147_features(schema):
    assert len(schema["feature_names"]) == 147
    assert len(schema["numeric_features"]) + len(schema["categorical_features"]) == 147


def test_schema_categories_are_nonempty_for_every_categorical_column(schema):
    for col in schema["categorical_features"]:
        assert len(schema["categories"][col]) > 0, f"{col} has an empty category list"


def test_schema_has_a_template_default_for_every_feature(schema):
    for col in schema["feature_names"]:
        assert col in schema["template_defaults"], f"missing template default for {col}"


# --- prediction / decision contract -------------------------------------

def test_build_feature_row_shape_and_order(schema):
    row = inf.build_feature_row({}, schema)
    assert row.shape == (1, 147)
    assert list(row.columns) == schema["feature_names"]


def test_prediction_in_unit_interval(model, schema):
    row = inf.build_feature_row({}, schema)
    p = inf.predict_probability(model, row)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("probability,expected", [(0.109, "APPROVE"), (0.110, "REJECT"), (0.111, "REJECT"), (0.0, "APPROVE"), (1.0, "REJECT")])
def test_decision_rule_boundary(probability, expected):
    assert inf.decide(probability) == expected


def test_distance_to_threshold_sign_and_magnitude():
    assert inf.distance_to_threshold_pp(0.16) == pytest.approx(5.0)
    assert inf.distance_to_threshold_pp(0.06) == pytest.approx(-5.0)


def test_strong_and_weak_synthetic_profiles_land_on_expected_sides(model, schema):
    strong = {
        "CODE_GENDER": "F", "FLAG_OWN_CAR": "Y", "FLAG_OWN_REALTY": "Y", "CNT_CHILDREN": 0,
        "AMT_INCOME_TOTAL": 250000.0, "AMT_CREDIT": 400000.0, "AMT_ANNUITY": 18000.0,
        "AMT_GOODS_PRICE": 380000.0, "NAME_INCOME_TYPE": "Working", "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married", "NAME_HOUSING_TYPE": "House / apartment", "DAYS_BIRTH": -14000,
        "OWN_CAR_AGE": 5.0, "OCCUPATION_TYPE": "Core staff", "CNT_FAM_MEMBERS": 2.0,
        "ORGANIZATION_TYPE": "Business Entity Type 3", "EXT_SOURCE_1": 0.75, "EXT_SOURCE_2": 0.78,
        "EXT_SOURCE_3": 0.72, "DAYS_EMPLOYED": -2920, "days_employed_anomaly": 0,
        "bureau_total_loans": 4, "bureau_active_loans": 1, "bureau_closed_loans": 3,
        "bureau_total_credit": 500000.0, "bureau_total_debt": 50000.0, "bureau_total_overdue": 0.0,
        "bureau_credit_type_variety": 2, "prev_app_count": 3, "prev_approval_ratio": 1.0,
        "prev_refusal_ratio": 0.0, "prev_avg_requested": 200000.0, "prev_avg_granted": 200000.0,
        "prev_avg_term": 12.0,
    }
    weak = dict(strong)
    weak.update({
        "EXT_SOURCE_1": 0.10, "EXT_SOURCE_2": 0.12, "EXT_SOURCE_3": 0.08,
        "bureau_total_debt": 650000.0, "bureau_active_loans": 6, "bureau_total_overdue": 25000.0,
        "prev_approval_ratio": 0.2, "prev_refusal_ratio": 0.6, "DAYS_EMPLOYED": None,
        "days_employed_anomaly": 1, "AMT_CREDIT": 900000.0, "AMT_INCOME_TOTAL": 90000.0,
    })

    p_strong = inf.predict_probability(model, inf.build_feature_row(strong, schema))
    p_weak = inf.predict_probability(model, inf.build_feature_row(weak, schema))

    assert inf.decide(p_strong) == "APPROVE"
    assert inf.decide(p_weak) == "REJECT"
    assert p_weak > p_strong


def test_no_bureau_or_prev_history_produces_missing_not_zero(schema):
    row_no_history = inf.build_feature_row({
        "bureau_total_loans": None, "bureau_active_loans": None, "bureau_closed_loans": None,
        "bureau_total_credit": None, "bureau_total_debt": None, "bureau_total_overdue": None,
        "bureau_credit_type_variety": None,
    }, schema)
    assert row_no_history["bureau_total_loans"].isna().all()
    assert row_no_history["debt_income_ratio"].isna().all(), (
        "debt_income_ratio must be NaN, not 0, when bureau_total_debt is unknown"
    )


def test_ratio_features_match_features_py_formulas(schema):
    row = inf.build_feature_row({
        "AMT_CREDIT": 400000.0, "AMT_INCOME_TOTAL": 200000.0, "AMT_ANNUITY": 20000.0,
        "AMT_GOODS_PRICE": 380000.0, "CNT_FAM_MEMBERS": 2.0, "bureau_total_debt": 50000.0,
    }, schema)
    assert row["credit_income_ratio"].iloc[0] == pytest.approx(400000.0 / 200000.0)
    assert row["annuity_income_ratio"].iloc[0] == pytest.approx(20000.0 / 200000.0)
    assert row["credit_annuity_ratio"].iloc[0] == pytest.approx(400000.0 / 20000.0)
    assert row["goods_credit_ratio"].iloc[0] == pytest.approx(380000.0 / 400000.0)
    assert row["income_per_person"].iloc[0] == pytest.approx(200000.0 / 2.0)
    assert row["debt_income_ratio"].iloc[0] == pytest.approx(50000.0 / 200000.0)


def test_ratio_features_handle_zero_denominator_as_missing_not_error(schema):
    row = inf.build_feature_row({"AMT_INCOME_TOTAL": 0.0, "AMT_CREDIT": 100000.0}, schema)
    assert row["credit_income_ratio"].isna().all()


def test_days_employed_sentinel_matches_data_quality_module():
    from data_quality import DAYS_EMPLOYED_SENTINEL

    days_employed, anomaly = inf.resolve_days_employed(False, 0.0)
    assert anomaly == 1
    assert days_employed is None

    days_employed, anomaly = inf.resolve_days_employed(True, 5.0)
    assert anomaly == 0
    assert days_employed == -round(5.0 * 365)
    assert days_employed != DAYS_EMPLOYED_SENTINEL


# --- SHAP explanation -----------------------------------------------------

def test_shap_explanation_runs_and_is_decision_consistent(model, explainer, schema):
    row = inf.build_feature_row({}, schema)
    probability = inf.predict_probability(model, row)
    explanation = inf.explain(model, explainer, row, schema, applicant_label="pytest-applicant")

    assert explanation["probability"] == pytest.approx(probability)
    assert explanation["decision"] == inf.decide(probability)
    assert explanation["shap_output_space"].startswith("log-odds")
    assert len(explanation["positive_risk_contributors"]) > 0
    assert len(explanation["negative_protective_contributors"]) > 0


# --- static repo-hygiene checks: no holdout, no retraining ----------------

FORBIDDEN_RUNTIME_IDENTIFIERS = (
    "HOLDOUT_PATH", "holdout.parquet", "load_holdout_data",
    ".fit(", "train_final_model(", "cross_validate", "GridSearch", "sweep_thresholds",
)


@pytest.mark.parametrize("relative_path", ["src/inference.py", "app/app.py"])
def test_no_holdout_access_or_retraining_in_deployment_code(relative_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(project_root, relative_path)) as f:
        source = f.read()
    for identifier in FORBIDDEN_RUNTIME_IDENTIFIERS:
        assert identifier not in source, f"{relative_path} must not contain '{identifier}'"


def test_build_model_artifact_never_reads_holdout():
    import build_model_artifact

    build_model_artifact.verify_no_holdout_reference()  # raises AssertionError if violated

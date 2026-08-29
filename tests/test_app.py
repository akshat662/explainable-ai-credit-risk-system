"""
Phase 10 tests: headless smoke tests for app/app.py via Streamlit's
AppTest -- drives the actual UI code (no browser) to catch integration
bugs that unit-testing src/inference.py alone would miss.
"""

import os

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(PROJECT_ROOT, "app", "app.py")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgboost_frozen.json")
SCHEMA_PATH = os.path.join(PROJECT_ROOT, "models", "feature_schema.json")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(MODEL_PATH) or not os.path.isfile(SCHEMA_PATH),
    reason="model artifact not built -- run `python src/build_model_artifact.py` first",
)

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

DEMO_LABELS_AND_EXPECTED_DECISIONS = [
    ("Illustrative strong applicant (synthetic)", "APPROVE"),
    ("Illustrative near-threshold applicant (synthetic)", "REJECT"),
    ("Illustrative weak applicant (synthetic)", "REJECT"),
]


def test_app_loads_without_exception():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception


def test_app_never_retrains_or_reads_holdout_at_runtime():
    """The app must load a pre-built artifact, not fit a new model, on every run."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    assert not at.exception
    # A retrain would take several seconds (246k rows, 300 trees); loading
    # a cached artifact should be near-instant. Not a strict proof, but a
    # useful canary alongside the static source-scan in test_inference.py.
    import time

    start = time.time()
    at.run(timeout=60)
    assert time.time() - start < 5.0, "app rerun was suspiciously slow -- check it isn't retraining"


@pytest.mark.parametrize("label,expected_decision", DEMO_LABELS_AND_EXPECTED_DECISIONS)
def test_demo_profile_end_to_end(label, expected_decision):
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)

    demo_button = next(b for b in at.button if b.label == label)
    demo_button.click().run(timeout=60)
    assert not at.exception

    assess_button = next(b for b in at.button if b.label == "Assess applicant")
    assess_button.click().run(timeout=60)
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Decision"] == expected_decision
    assert metrics["Policy threshold"] == "11.0%"

    probability_str = metrics["Default probability"].rstrip("%")
    assert 0.0 <= float(probability_str) <= 100.0


def test_unchecking_history_toggles_does_not_crash():
    """No bureau / no previous-application / not-employed paths must not error."""
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=60)
    at.checkbox(key="in_has_bureau_history").uncheck().run(timeout=60)
    at.checkbox(key="in_has_prev_apps").uncheck().run(timeout=60)
    at.checkbox(key="in_is_employed").uncheck().run(timeout=60)
    assert not at.exception

    assess_button = next(b for b in at.button if b.label == "Assess applicant")
    assess_button.click().run(timeout=60)
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Decision"] in {"APPROVE", "REJECT"}

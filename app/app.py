"""
Phase 10: Streamlit deployment layer for the frozen credit-risk system.

    Applicant input -> feature construction -> frozen XGBoost model
    -> raw default probability -> 0.110 business threshold
    -> APPROVE / REJECT -> decision-aware SHAP explanation

This file is UI ONLY. All model/feature logic lives in src/inference.py,
which is the sole import from this file for anything model-related --
app.py never trains, calibrates, sweeps a threshold, or reads the
untouched final-evaluation set. See src/inference.py's module docstring
for exactly how the 147-feature model input is reconstructed from a
curated subset of applicant fields.

Run with:  streamlit run app/app.py   (from the project root)
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import inference as inf  # noqa: E402
from shap_explainability import display_name  # noqa: E402

st.set_page_config(page_title="Explainable Credit Risk Decision System", layout="wide")


# ---------------------------------------------------------------------------
# Cached, load-once resources (never retrained by the app)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_schema():
    return inf.load_schema()


@st.cache_resource
def get_model():
    return inf.load_model()


@st.cache_resource
def get_explainer(_model):
    return inf.load_explainer(_model)


# ---------------------------------------------------------------------------
# Deterministic synthetic demo applicants -- NOT derived from any dataset
# applicant (dev or holdout). Hand-constructed and verified against the
# frozen model/threshold to illustrate both decision outcomes plus a
# near-boundary case. Clearly labeled as synthetic everywhere they appear.
# ---------------------------------------------------------------------------

DEMO_PROFILES = {
    "Illustrative strong applicant (synthetic)": dict(
        loan_type="Cash loans", gender="F", owns_car=True, car_age=5.0, owns_realty=True,
        children=0, family_members=2, family_status="Married", education="Higher education",
        housing="House / apartment", age_years=38,
        income=250000.0, credit=400000.0, annuity=18000.0, goods_price=380000.0,
        income_type="Working", is_employed=True, years_employed=8.0,
        occupation="Core staff", organization="Business Entity Type 3",
        ext_source_1=0.75, ext_source_2=0.78, ext_source_3=0.72,
        has_bureau_history=True, bureau_total_loans=4, bureau_active_loans=1,
        bureau_closed_loans=3, bureau_total_credit=500000.0, bureau_total_debt=50000.0,
        bureau_total_overdue=0.0, bureau_credit_type_variety=2,
        has_prev_apps=True, prev_app_count=3, prev_approval_ratio=1.0, prev_refusal_ratio=0.0,
        prev_avg_requested=200000.0, prev_avg_granted=200000.0, prev_avg_term=12.0,
    ),
    "Illustrative near-threshold applicant (synthetic)": dict(
        loan_type="Cash loans", gender="M", owns_car=False, car_age=None, owns_realty=True,
        children=1, family_members=3, family_status="Married", education="Secondary / secondary special",
        housing="House / apartment", age_years=31,
        income=135000.0, credit=600000.0, annuity=29000.0, goods_price=580000.0,
        income_type="Commercial associate", is_employed=True, years_employed=4.0,
        occupation="Sales staff", organization="Self-employed",
        ext_source_1=0.42, ext_source_2=0.40, ext_source_3=0.38,
        has_bureau_history=True, bureau_total_loans=5, bureau_active_loans=3,
        bureau_closed_loans=2, bureau_total_credit=550000.0, bureau_total_debt=300000.0,
        bureau_total_overdue=0.0, bureau_credit_type_variety=2,
        has_prev_apps=True, prev_app_count=2, prev_approval_ratio=0.5, prev_refusal_ratio=0.5,
        prev_avg_requested=250000.0, prev_avg_granted=150000.0, prev_avg_term=18.0,
    ),
    "Illustrative weak applicant (synthetic)": dict(
        loan_type="Cash loans", gender="M", owns_car=False, car_age=None, owns_realty=False,
        children=3, family_members=4, family_status="Single / not married", education="Secondary / secondary special",
        housing="Rented apartment", age_years=23,
        income=90000.0, credit=900000.0, annuity=45000.0, goods_price=850000.0,
        income_type="Working", is_employed=False, years_employed=0.0,
        occupation="Low-skill Laborers", organization="Construction",
        ext_source_1=0.10, ext_source_2=0.12, ext_source_3=0.08,
        has_bureau_history=True, bureau_total_loans=8, bureau_active_loans=6,
        bureau_closed_loans=2, bureau_total_credit=700000.0, bureau_total_debt=650000.0,
        bureau_total_overdue=25000.0, bureau_credit_type_variety=4,
        has_prev_apps=True, prev_app_count=5, prev_approval_ratio=0.2, prev_refusal_ratio=0.6,
        prev_avg_requested=300000.0, prev_avg_granted=100000.0, prev_avg_term=36.0,
    ),
}

DEFAULT_PROFILE = DEMO_PROFILES["Illustrative near-threshold applicant (synthetic)"]

FORM_KEYS = list(DEFAULT_PROFILE.keys())

# The 147-feature model input is mostly a frozen dev-set median/mode template
# (see src/inference.py); these are the raw/derived columns this form actually
# collects and overrides. Used only to annotate SHAP contributors honestly --
# never affects prediction or explanation logic.
USER_CONTROLLED_FEATURES = frozenset({
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "DAYS_BIRTH", "OWN_CAR_AGE",
    "OCCUPATION_TYPE", "CNT_FAM_MEMBERS", "ORGANIZATION_TYPE", "EXT_SOURCE_1", "EXT_SOURCE_2",
    "EXT_SOURCE_3", "DAYS_EMPLOYED", "days_employed_anomaly",
    "bureau_total_loans", "bureau_active_loans", "bureau_closed_loans", "bureau_total_credit",
    "bureau_total_debt", "bureau_total_overdue", "bureau_credit_type_variety",
    "prev_app_count", "prev_approval_ratio", "prev_refusal_ratio", "prev_avg_requested",
    "prev_avg_granted", "prev_avg_term",
    "credit_income_ratio", "annuity_income_ratio", "credit_annuity_ratio",
    "goods_credit_ratio", "income_per_person", "debt_income_ratio",
})


def _set_or_clear(key, value):
    # None (e.g. car_age for a non-car-owning profile) is skipped rather than
    # written: a conditionally rendered widget (e.g. the car-age slider) can't
    # accept None as its session_state-seeded value once its condition later
    # becomes true -- leaving the key unset lets the widget fall back to its
    # own explicit default the first time it actually renders (see
    # render_input_form's "in_car_age"/"in_years_employed" handling).
    sk = f"in_{key}"
    if value is not None:
        st.session_state[sk] = value
    else:
        st.session_state.pop(sk, None)


def apply_profile(profile):
    """Overwrite every form field's session_state with a demo profile."""
    for key in FORM_KEYS:
        _set_or_clear(key, profile[key])


def seed_defaults():
    """Seed session_state with DEFAULT_PROFILE for any key not already set,
    once, before any widget is instantiated -- so every widget below can be
    created with only `key=` (no `value=`/`index=`), which is what avoids
    Streamlit's "widget created with a default value but also had its value
    set via the Session State API" warning on every rerun after a demo
    profile has been loaded."""
    for key in FORM_KEYS:
        sk = f"in_{key}"
        if sk not in st.session_state:
            _set_or_clear(key, DEFAULT_PROFILE[key])


# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------

def render_demo_picker():
    st.caption(
        "Load a synthetic, hand-constructed demo applicant (not a real person, "
        "not derived from this project's development or holdout data) to try the "
        "system quickly, or fill in the form yourself below."
    )
    cols = st.columns(len(DEMO_PROFILES))
    for col, (label, profile) in zip(cols, DEMO_PROFILES.items()):
        if col.button(label, use_container_width=True):
            apply_profile(profile)
            st.rerun()


def render_input_form(schema):
    seed_defaults()

    org_options = sorted(schema["categories"]["ORGANIZATION_TYPE"])
    occupation_options = ["Not specified"] + sorted(schema["categories"]["OCCUPATION_TYPE"])

    st.subheader("Applicant profile")
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", ["F", "M", "XNA"], key="in_gender")
    age_years = c2.slider("Age (years)", 20, 70, key="in_age_years")
    family_status = c3.selectbox("Family status", schema["categories"]["NAME_FAMILY_STATUS"], key="in_family_status")
    c4, c5, c6 = st.columns(3)
    children = c4.number_input("Number of children", 0, 15, key="in_children")
    family_members = c5.number_input("Family members (incl. applicant)", 1, 15, key="in_family_members")
    education = c6.selectbox("Education", schema["categories"]["NAME_EDUCATION_TYPE"], key="in_education")
    c7, c8, c9 = st.columns(3)
    housing = c7.selectbox("Housing situation", schema["categories"]["NAME_HOUSING_TYPE"], key="in_housing")
    owns_realty = c8.checkbox("Owns real estate", key="in_owns_realty")
    owns_car = c9.checkbox("Owns a car", key="in_owns_car")
    car_age = None
    if owns_car:
        if "in_car_age" not in st.session_state:
            car_age = st.slider("Car age (years)", 0, 40, 8, key="in_car_age")
        else:
            car_age = st.slider("Car age (years)", 0, 40, key="in_car_age")

    st.subheader("Financial information")
    c1, c2 = st.columns(2)
    loan_type = c1.selectbox("Loan type", schema["categories"]["NAME_CONTRACT_TYPE"], key="in_loan_type")
    income_type = c2.selectbox("Income type", schema["categories"]["NAME_INCOME_TYPE"], key="in_income_type")
    c3, c4 = st.columns(2)
    income = c3.number_input("Total annual income", 25000.0, 10_000_000.0, step=5000.0, key="in_income")
    credit = c4.number_input("Requested credit amount", 45000.0, 5_000_000.0, step=5000.0, key="in_credit")
    c5, c6 = st.columns(2)
    annuity = c5.number_input("Loan annuity (installment amount)", 1500.0, 500_000.0, step=500.0, key="in_annuity")
    goods_price = c6.number_input("Price of goods being financed", 40000.0, 5_000_000.0, step=5000.0, key="in_goods_price")

    st.subheader("Employment")
    c1, c2, c3 = st.columns(3)
    is_employed = c1.checkbox("Currently employed", key="in_is_employed")
    years_employed = 0.0
    if is_employed:
        if "in_years_employed" not in st.session_state:
            years_employed = c1.slider("Years employed", 0.0, 45.0, 3.0, key="in_years_employed")
        else:
            years_employed = c1.slider("Years employed", 0.0, 45.0, key="in_years_employed")
    occupation_choice = c2.selectbox("Occupation", occupation_options, key="in_occupation")
    organization = c3.selectbox("Employer organization type", org_options, key="in_organization")

    st.subheader("External credit scores")
    st.caption(
        "Normalized external credit-bureau scores from the source dataset "
        "(0 = weakest, 1 = strongest). These are the single strongest global "
        "SHAP drivers of this model's predictions."
    )
    c1, c2, c3 = st.columns(3)
    ext1 = c1.slider("External score 1", 0.0, 1.0, step=0.01, key="in_ext_source_1")
    ext2 = c2.slider("External score 2", 0.0, 1.0, step=0.01, key="in_ext_source_2")
    ext3 = c3.slider("External score 3", 0.0, 1.0, step=0.01, key="in_ext_source_3")

    st.subheader("Credit history (credit bureau)")
    has_bureau_history = st.checkbox(
        "Applicant has bureau-reported credit history", key="in_has_bureau_history",
        help="Unchecked means no bureau record at all -- kept as missing (NaN), "
             "the same way the frozen pipeline distinguishes 'no history' from 'zero history'.",
    )
    bureau_fields = (
        "bureau_total_loans", "bureau_active_loans", "bureau_closed_loans",
        "bureau_total_credit", "bureau_total_debt", "bureau_total_overdue",
        "bureau_credit_type_variety",
    )
    if has_bureau_history:
        c1, c2, c3 = st.columns(3)
        bureau = {}
        bureau["bureau_total_loans"] = c1.number_input("Total bureau loans", 1, 100, key="in_bureau_total_loans")
        bureau["bureau_active_loans"] = c2.number_input("Active bureau loans", 0, 100, key="in_bureau_active_loans")
        bureau["bureau_closed_loans"] = c3.number_input("Closed bureau loans", 0, 100, key="in_bureau_closed_loans")
        c4, c5, c6 = st.columns(3)
        bureau["bureau_total_credit"] = c4.number_input("Total bureau credit amount", 0.0, 20_000_000.0, step=10000.0, key="in_bureau_total_credit")
        bureau["bureau_total_debt"] = c5.number_input("Total bureau debt (outstanding)", 0.0, 20_000_000.0, step=10000.0, key="in_bureau_total_debt")
        bureau["bureau_total_overdue"] = c6.number_input("Total bureau amount overdue", 0.0, 5_000_000.0, step=1000.0, key="in_bureau_total_overdue")
        bureau["bureau_credit_type_variety"] = st.number_input("Distinct bureau credit types", 1, 15, key="in_bureau_credit_type_variety")
    else:
        # No bureau record at all: explicitly None (missing), never the dev-median
        # template fallback -- "no history" and "zero history" are different facts
        # throughout this project's feature pipeline (see features.py), and the UI
        # must preserve that distinction the same way.
        bureau = {field: None for field in bureau_fields}

    st.subheader("Previous Home Credit applications")
    has_prev_apps = st.checkbox(
        "Applicant has previous Home Credit applications", key="in_has_prev_apps",
        help="Unchecked means no prior applications at all -- kept as missing (NaN), not zero.",
    )
    prev_fields = (
        "prev_app_count", "prev_approval_ratio", "prev_refusal_ratio",
        "prev_avg_requested", "prev_avg_granted", "prev_avg_term",
    )
    if has_prev_apps:
        c1, c2, c3 = st.columns(3)
        prev = {}
        prev["prev_app_count"] = c1.number_input("Number of previous applications", 1, 100, key="in_prev_app_count")
        prev["prev_approval_ratio"] = c2.slider("Prior approval ratio", 0.0, 1.0, step=0.05, key="in_prev_approval_ratio")
        prev["prev_refusal_ratio"] = c3.slider("Prior refusal ratio", 0.0, 1.0, step=0.05, key="in_prev_refusal_ratio")
        c4, c5, c6 = st.columns(3)
        prev["prev_avg_requested"] = c4.number_input("Avg. previously requested amount", 0.0, 5_000_000.0, step=5000.0, key="in_prev_avg_requested")
        prev["prev_avg_granted"] = c5.number_input("Avg. previously granted amount", 0.0, 5_000_000.0, step=5000.0, key="in_prev_avg_granted")
        prev["prev_avg_term"] = c6.number_input("Avg. previous loan term (months)", 0.0, 96.0, step=1.0, key="in_prev_avg_term")
    else:
        prev = {field: None for field in prev_fields}

    days_employed, days_employed_anomaly = inf.resolve_days_employed(is_employed, years_employed)

    user_inputs = {
        "NAME_CONTRACT_TYPE": loan_type,
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": "Y" if owns_car else "N",
        "FLAG_OWN_REALTY": "Y" if owns_realty else "N",
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": -round(age_years * 365),
        "OWN_CAR_AGE": car_age,
        "OCCUPATION_TYPE": None if occupation_choice == "Not specified" else occupation_choice,
        "CNT_FAM_MEMBERS": family_members,
        "ORGANIZATION_TYPE": organization,
        "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3,
        "DAYS_EMPLOYED": days_employed,
        "days_employed_anomaly": days_employed_anomaly,
        **bureau,
        **prev,
    }
    return user_inputs


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

def render_risk_result(probability, threshold, decision):
    distance_pp = inf.distance_to_threshold_pp(probability, threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Default probability", f"{probability * 100:.1f}%")
    c2.metric("Policy threshold", f"{threshold * 100:.1f}%")
    c3.metric("Distance from threshold", f"{distance_pp:+.1f} pp")
    c4.metric("Decision", decision)

    fig, ax = plt.subplots(figsize=(8, 1.4))
    ax.barh([0], [1.0], color="#e6e6e6", height=0.5)
    ax.barh([0], [min(probability, 1.0)], color="#c0392b" if decision == "REJECT" else "#2e7d32", height=0.5)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5)
    ax.text(threshold, 0.45, f" threshold {threshold * 100:.1f}%", va="bottom", ha="left" if threshold < 0.85 else "right", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Predicted default probability")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if decision == "REJECT":
        st.error(
            f"The predicted default risk ({probability*100:.1f}%) is above the frozen policy "
            f"threshold ({threshold*100:.1f}%), so the applicant is classified as **REJECT** "
            f"({distance_pp:+.1f} percentage points past the boundary)."
        )
    else:
        st.success(
            f"The predicted default risk ({probability*100:.1f}%) is below the frozen policy "
            f"threshold ({threshold*100:.1f}%), so the applicant is classified as **APPROVE** "
            f"({distance_pp:+.1f} percentage points from the boundary)."
        )
    st.caption(
        "This is a probability estimate from a statistical model, not a certainty -- it "
        "reflects patterns in historical applicants with similar characteristics, not a "
        "guarantee about this specific applicant's future behavior."
    )


def render_shap(explanation):
    st.subheader("Why did this applicant land here?")
    st.caption(
        f"Risk is {explanation['distance_to_threshold_percentage_points']:+.1f} percentage "
        f"points {'above' if explanation['distance_to_threshold_percentage_points'] >= 0 else 'below'} "
        "the policy threshold. The factors below explain the model's underlying risk score "
        "(SHAP values, in log-odds/raw-margin space -- **not** a literal percentage-point "
        "probability contribution, since the probability transform is nonlinear) -- they do "
        "not imply that any feature *causes* default."
    )

    pos = explanation["positive_risk_contributors"]
    neg = explanation["negative_protective_contributors"]

    labels, values, colors = [], [], []
    for c in reversed(neg):
        labels.append(c["display_name"])
        values.append(c["shap_value_log_odds"])
        colors.append("#2e7d32")
    for c in pos:
        labels.append(c["display_name"])
        values.append(c["shap_value_log_odds"])
        colors.append("#c0392b")

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(labels) + 1))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP contribution (log-odds / raw margin space)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    def _bullet(c):
        note = "" if c["feature"] in USER_CONTROLLED_FEATURES else " *(dev-typical default, not entered above)*"
        return f"- {c['display_name']} (`{c['feature']}` = {c['feature_value']}){note}"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Key factors pushing risk higher:**")
        for c in pos:
            st.markdown(_bullet(c))
    with c2:
        st.markdown("**Key factors pushing risk lower:**")
        for c in neg:
            st.markdown(_bullet(c))

    if any(c["feature"] not in USER_CONTROLLED_FEATURES for c in pos + neg):
        st.caption(
            "Fields marked *dev-typical default* were not collected by this form -- "
            "they were filled from the dev-set median/mode template (see 'Model "
            "information' below) and still influenced this particular prediction."
        )


def render_model_info(schema):
    with st.expander("Model information"):
        st.markdown(
            f"""
- **Model**: XGBoost (`n_estimators={schema['model_params']['n_estimators']}`,
  `max_depth={schema['model_params']['max_depth']}`,
  `learning_rate={schema['model_params']['learning_rate']}`), frozen and unchanged since
  benchmarking -- trained once on {schema['n_dev_rows_trained_on']:,} development applicants.
- **Probability**: raw, uncalibrated XGBoost output. Platt scaling and isotonic
  regression were evaluated and rejected -- a bootstrap test found isotonic's tiny
  holdout Brier-score edge statistically indistinguishable from noise.
- **Decision threshold**: {inf.FROZEN_THRESHOLD:.3f}, selected on development data only
  under an expected-cost model (loss given default = {inf.LGD:.2f}, margin if repaid =
  {inf.MARGIN:.2f}), then evaluated exactly once on an untouched holdout set. This app
  applies that frozen threshold; it does not re-derive or adjust it.
- **Explainability**: SHAP (`TreeExplainer`), values reported in log-odds/raw-margin
  space -- verified empirically to satisfy `base_value + sum(SHAP) == raw model output`.
- **Evaluation protocol**: 5-fold cross-validation and threshold selection on
  development data; final performance (ROC-AUC 0.7713, PR-AUC 0.2650 at
  {inf.FROZEN_THRESHOLD:.3f}) measured once on a held-out set never used during
  development.
- **Demo data**: the "load a demo applicant" profiles above are synthetic and
  hand-constructed for illustration -- they are not real applicants and are not
  derived from this project's development or holdout data. Fields this form does not
  collect (mostly building/apartment descriptors and administrative flags) are filled
  with dev-set typical values (median/mode), a disclosed simplification for this demo.
"""
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("Explainable Credit Risk Decision System")
    st.caption(
        "A portfolio project demonstrating an explainable, business-aware credit "
        "risk decision pipeline -- not a production banking system."
    )

    try:
        schema = get_schema()
        model = get_model()
        explainer = get_explainer(model)
    except inf.ModelArtifactMissing as e:
        st.error(str(e))
        st.stop()

    render_demo_picker()
    st.divider()

    user_inputs = render_input_form(schema)

    st.divider()
    if st.button("Assess applicant", type="primary"):
        row = inf.build_feature_row(user_inputs, schema)
        probability = inf.predict_probability(model, row)
        decision = inf.decide(probability)

        st.header("Risk result")
        render_risk_result(probability, inf.FROZEN_THRESHOLD, decision)

        st.divider()
        explanation = inf.explain(model, explainer, row, schema, applicant_label="submitted-applicant")
        render_shap(explanation)

    st.divider()
    render_model_info(schema)


if __name__ == "__main__":
    main()

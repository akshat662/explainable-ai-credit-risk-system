"""
Phase 9: SHAP explainability for the frozen XGBoost model.

Explains WHY the frozen model assigns a given risk score, and connects
that score to the frozen 0.110 business decision threshold. Does not
retrain, tune, or otherwise modify the model, the threshold, or any
economic assumption -- all imported unchanged from prior phases.

SHAP OUTPUT SPACE (verified empirically against the actual installed
shap==0.52.0 / xgboost==3.4.1, not assumed from memory -- see
DECISIONS.md): shap.TreeExplainer defaults to model_output="raw" for
XGBClassifier, meaning SHAP values are additive in RAW MARGIN (log-odds)
space:

    base_value + sum(shap_values) == model.predict(X, output_margin=True)

verified to hold within ~2e-6 (float32 precision). Consequently:
sigmoid(base_value + sum(shap_values)) == predict_proba exactly. This
module therefore:
  - reports individual SHAP values as log-odds contributions (never as
    "this feature changed the probability by X%"), since sigmoid is
    nonlinear and log-odds contributions do not decompose additively
    into per-feature probability changes;
  - computes the predicted probability, its distance to the 0.110
    threshold, and the APPROVE/REJECT decision directly from
    model.predict_proba -- never by summing or transforming SHAP values.

Global importance uses a deterministic sample of DEVELOPMENT applicants
only (never holdout labels or holdout-derived ranking). Local explanation
is a reusable function intended for direct reuse in Phase 10 (Streamlit).
"""

import json
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import shap

from evaluate_final_holdout import FROZEN_THRESHOLD
from evaluate_holdout_calibration import train_final_model
from profile_data import PROJECT_ROOT
from train_xgboost import load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GLOBAL_IMPORTANCE_CSV_PATH = os.path.join(PROJECT_ROOT, "reports", "shap_global_importance.csv")
GLOBAL_IMPORTANCE_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "shap_global_importance.png")
SUMMARY_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "shap_summary.png")
LOCAL_EXAMPLE_PATH = os.path.join(PROJECT_ROOT, "reports", "shap_local_example.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "shap_explainability_report.md")

# Deterministic dev-only sample for global SHAP -- computing/plotting SHAP
# for all 246,008 dev rows is unnecessary for a stable importance ranking
# and would make the beeswarm plot unreadable; 2,000 is a standard SHAP
# sample size that balances ranking stability, runtime, and plot clarity.
GLOBAL_SAMPLE_SIZE = 2000
GLOBAL_SAMPLE_RANDOM_STATE = 42
TOP_N_GLOBAL_FEATURES = 20
TOP_K_LOCAL_CONTRIBUTORS = 5

# Additivity check tolerance (base_value + sum(shap) vs raw margin output).
ADDITIVITY_TOLERANCE = 1e-4

# Small, explicit display-name mapping for features likely to appear in the
# top rankings or be used in local-explanation demos. Not exhaustive by
# design -- unmapped features simply show their raw column name, which is
# still technically correct, just less immediately readable.
FEATURE_DISPLAY_NAMES = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "AMT_CREDIT": "Requested credit amount",
    "AMT_INCOME_TOTAL": "Total applicant income",
    "AMT_ANNUITY": "Loan annuity (installment amount)",
    "AMT_GOODS_PRICE": "Price of goods being financed",
    "DAYS_BIRTH": "Applicant age (days, negative = past)",
    "DAYS_EMPLOYED": "Days employed (negative = past)",
    "days_employed_anomaly": "Employment sentinel flag (Pensioner/Unemployed)",
    "DAYS_ID_PUBLISH": "Days since ID document issued",
    "DAYS_REGISTRATION": "Days since registration change",
    "DAYS_LAST_PHONE_CHANGE": "Days since last phone number change",
    "OWN_CAR_AGE": "Car age (years)",
    "CNT_FAM_MEMBERS": "Number of family members",
    "CNT_CHILDREN": "Number of children",
    "REGION_POPULATION_RELATIVE": "Region population density (relative)",
    "REGION_RATING_CLIENT": "Region risk rating",
    "credit_income_ratio": "Credit-to-income ratio",
    "annuity_income_ratio": "Annuity-to-income ratio",
    "credit_annuity_ratio": "Credit-to-annuity ratio",
    "goods_credit_ratio": "Goods price-to-credit ratio",
    "income_per_person": "Income per family member",
    "debt_income_ratio": "Bureau debt-to-income ratio",
    "bureau_total_debt": "Total bureau-reported debt",
    "bureau_total_credit": "Total bureau-reported credit",
    "bureau_avg_credit": "Average bureau-reported credit",
    "bureau_max_credit": "Maximum bureau-reported credit",
    "bureau_total_loans": "Number of bureau-reported loans",
    "bureau_active_loans": "Number of active bureau loans",
    "bureau_closed_loans": "Number of closed bureau loans",
    "bureau_total_overdue": "Total bureau-reported overdue amount",
    "bureau_avg_days_overdue": "Average days overdue (bureau)",
    "bureau_max_days_overdue": "Maximum days overdue (bureau)",
    "bureau_credit_type_variety": "Variety of bureau credit types",
    "bureau_total_prolongs": "Number of bureau credit prolongations",
    "bureau_had_negative_debt": "Had a negative bureau debt value (data anomaly flag)",
    "prev_app_count": "Number of previous Home Credit applications",
    "prev_approval_ratio": "Previous application approval ratio",
    "prev_refusal_ratio": "Previous application refusal ratio",
    "prev_avg_requested": "Average previously requested amount",
    "prev_avg_granted": "Average previously granted amount",
    "prev_grant_ratio": "Previous grant-to-request ratio",
    "prev_avg_term": "Average previous loan term",
    "NAME_EDUCATION_TYPE": "Education level",
    "NAME_INCOME_TYPE": "Income type",
    "NAME_FAMILY_STATUS": "Family status",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer organization type",
    "CODE_GENDER": "Gender",
    "FLAG_OWN_CAR": "Owns a car",
    "FLAG_OWN_REALTY": "Owns real estate",
}


def display_name(feature):
    """Human-readable label for a feature, falling back to the raw name."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def load_frozen_model_and_dev():
    """Train the frozen XGBoost model on all of dev.parquet.

    Reuses evaluate_holdout_calibration.train_final_model (imported, not
    redefined) -- the same deterministic training call already used and
    validated in Phase 8.5 to score holdout. No new model, no tuning.
    """
    dev_df = load_dev_data()
    X_dev, y_dev = split_features_target(dev_df)
    X_dev, _, _ = prepare_categoricals(X_dev)
    model = train_final_model(X_dev, y_dev)
    return model, X_dev, y_dev, dev_df


def sample_dev_for_global_shap(X_dev, sample_size=GLOBAL_SAMPLE_SIZE, random_state=GLOBAL_SAMPLE_RANDOM_STATE):
    """Deterministic dev-only sample for global SHAP computation."""
    return X_dev.sample(n=sample_size, random_state=random_state)


def compute_shap_values(model, X):
    """TreeExplainer SHAP values (raw/log-odds space) for X."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, np.asarray(shap_values)


def validate_additivity(model, explainer, shap_values, X, tolerance=ADDITIVITY_TOLERANCE):
    """base_value + sum(shap_values) must equal the raw margin model output."""
    additive_output = explainer.expected_value + shap_values.sum(axis=1)
    raw_margin_output = model.predict(X, output_margin=True)
    max_abs_diff = float(np.max(np.abs(additive_output - raw_margin_output)))
    passed = max_abs_diff < tolerance
    logger.info(
        "Additivity check (base_value + sum(SHAP) vs raw margin output): max_abs_diff=%.2e, tolerance=%.2e, passed=%s",
        max_abs_diff, tolerance, passed,
    )
    assert passed, f"SHAP additivity check failed: max_abs_diff={max_abs_diff:.2e} >= tolerance={tolerance:.2e}"
    return max_abs_diff


def global_importance_table(shap_values, feature_names, top_n=TOP_N_GLOBAL_FEATURES):
    """Mean |SHAP value| per feature (log-odds space), ranked descending."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    table = pd.DataFrame({
        "feature": feature_names,
        "display_name": [display_name(f) for f in feature_names],
        "mean_abs_shap_value": mean_abs_shap,
    }).sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table, table.head(top_n)


def plot_global_importance(top_table, output_path=GLOBAL_IMPORTANCE_PLOT_PATH):
    fig, ax = plt.subplots(figsize=(9, 7))
    ordered = top_table.iloc[::-1]  # largest at top
    ax.barh(ordered["display_name"], ordered["mean_abs_shap_value"], color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (log-odds / raw margin space)")
    ax.set_title(f"Global Feature Importance — Top {len(top_table)} (SHAP, dev sample n={GLOBAL_SAMPLE_SIZE})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_summary_beeswarm(shap_values, X_sample, output_path=SUMMARY_PLOT_PATH):
    """SHAP beeswarm/summary plot (log-odds space) over the dev sample."""
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=TOP_N_GLOBAL_FEATURES)
    fig = plt.gcf()
    fig.suptitle("SHAP Summary (log-odds / raw margin space)", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def explain_applicant(applicant_id, X_row, model, explainer, feature_names,
                       threshold=FROZEN_THRESHOLD, top_k=TOP_K_LOCAL_CONTRIBUTORS):
    """Explain one applicant. Reusable as-is by Phase 10 (Streamlit).

    `X_row` must be a single-row DataFrame with the same 147 columns
    (dtypes, categorical encoding) the frozen model was trained on --
    callers should pass a row already produced by the frozen
    prepare_categoricals() pipeline, not raw, unprocessed data.

    Probability, decision, and distance-to-threshold are computed
    directly via model.predict_proba -- never derived from summing SHAP
    values, since SHAP is additive in log-odds space, not probability
    space (see module docstring). SHAP values are reported as log-odds
    contributions only.
    """
    assert list(X_row.columns) == list(feature_names), "explain_applicant: feature columns must match the frozen model input exactly"

    probability = float(model.predict_proba(X_row)[:, 1][0])
    decision = "REJECT" if probability >= threshold else "APPROVE"
    distance_to_threshold_pp = (probability - threshold) * 100  # percentage points, probability space -- valid

    row_shap = explainer.shap_values(X_row)
    row_shap = np.asarray(row_shap).reshape(-1)
    base_value = float(explainer.expected_value)
    raw_margin_output = float(model.predict(X_row, output_margin=True)[0])

    contributions = pd.DataFrame({
        "feature": feature_names,
        "display_name": [display_name(f) for f in feature_names],
        "feature_value": [X_row.iloc[0][f] for f in feature_names],
        "shap_value_log_odds": row_shap,
    })
    contributions["feature_value"] = contributions["feature_value"].apply(
        lambda v: None if pd.isna(v) else (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
    )

    positive = contributions[contributions["shap_value_log_odds"] > 0].sort_values(
        "shap_value_log_odds", ascending=False
    ).head(top_k)
    negative = contributions[contributions["shap_value_log_odds"] < 0].sort_values(
        "shap_value_log_odds", ascending=True
    ).head(top_k)

    def _rows(df):
        return [
            {
                "feature": r["feature"],
                "display_name": r["display_name"],
                "feature_value": r["feature_value"],
                "shap_value_log_odds": round(float(r["shap_value_log_odds"]), 5),
            }
            for _, r in df.iterrows()
        ]

    return {
        "applicant_id": applicant_id,
        "probability": probability,
        "threshold": threshold,
        "decision": decision,
        "distance_to_threshold_percentage_points": distance_to_threshold_pp,
        "shap_output_space": "log-odds (raw margin); NOT probability space",
        "base_value_log_odds": base_value,
        "raw_margin_output_log_odds": raw_margin_output,
        "positive_risk_contributors": _rows(positive),
        "negative_protective_contributors": _rows(negative),
        "note": (
            "SHAP values are additive in log-odds space, not probability space. "
            "They indicate direction and relative magnitude of each feature's "
            "contribution to the model's risk score, not a literal percentage-point "
            "change in predicted probability."
        ),
    }


DEMO_TARGET_DISTANCE_PP = 5.0  # percentage points above threshold


def select_demo_applicant(model, X_dev, threshold=FROZEN_THRESHOLD, target_distance_pp=DEMO_TARGET_DISTANCE_PP):
    """Deterministically pick an illustrative near-boundary REJECT case.

    Picks the REJECT-side (probability >= threshold) dev applicant whose
    probability is closest to threshold + target_distance_pp. A clear,
    legible margin (default ~5 percentage points, matching this project's
    own conceptual example) is more illustrative of the "distance from
    threshold" reasoning than either an arbitrary applicant or one sitting
    exactly at the boundary (which produces an uninformative ~0.0pp
    distance). Deterministic (argmin over the full dev set, no randomness).
    """
    probabilities = model.predict_proba(X_dev)[:, 1]
    reject_mask = probabilities >= threshold
    assert reject_mask.any(), "no dev applicant is on the REJECT side of the threshold -- cannot pick a boundary demo"
    candidate_positions = np.flatnonzero(reject_mask)
    target_probability = threshold + target_distance_pp / 100
    distances = np.abs(probabilities[candidate_positions] - target_probability)
    best = candidate_positions[np.argmin(distances)]
    return int(best)


def format_local_explanation_text(explanation):
    """Human-readable rendering of an explain_applicant() result."""
    lines = [
        f"Applicant: {explanation['applicant_id']}",
        "",
        f"Predicted default probability: {explanation['probability']*100:.1f}%",
        f"Decision threshold: {explanation['threshold']*100:.1f}%",
        f"Distance from threshold: {explanation['distance_to_threshold_percentage_points']:+.1f} percentage points",
        f"Decision: {explanation['decision']}",
        "",
        "Main risk contributors (higher model risk score, log-odds space):",
    ]
    for i, c in enumerate(explanation["positive_risk_contributors"], start=1):
        lines.append(f"  {i}. {c['display_name']} ({c['feature']} = {c['feature_value']}), SHAP={c['shap_value_log_odds']:+.4f}")
    lines.append("")
    lines.append("Main protective contributors (lower model risk score, log-odds space):")
    for i, c in enumerate(explanation["negative_protective_contributors"], start=1):
        lines.append(f"  {i}. {c['display_name']} ({c['feature']} = {c['feature_value']}), SHAP={c['shap_value_log_odds']:+.4f}")
    return "\n".join(lines)


def validate_setup(model, X_dev, shap_values, X_sample):
    """Sanity checks required before trusting the SHAP artifacts."""
    assert X_dev.shape[1] == 147, f"expected 147 model features, got {X_dev.shape[1]}"
    assert shap_values.shape == (len(X_sample), X_dev.shape[1]), (
        f"SHAP values shape {shap_values.shape} does not match sample/feature dimensions "
        f"({len(X_sample)}, {X_dev.shape[1]})"
    )
    assert list(X_sample.columns) == list(X_dev.columns), "SHAP feature names must align with model input feature order"
    logger.info("Setup validation passed: 147 features confirmed, SHAP array shape and feature order aligned.")


def main():
    logger.info("Training frozen XGBoost model on all of dev.parquet (no tuning, no CV)")
    model, X_dev, y_dev, dev_df = load_frozen_model_and_dev()

    logger.info(
        "Sampling %d dev applicants (random_state=%d) for global SHAP",
        GLOBAL_SAMPLE_SIZE, GLOBAL_SAMPLE_RANDOM_STATE,
    )
    X_sample = sample_dev_for_global_shap(X_dev)

    logger.info("Computing SHAP values (TreeExplainer, raw/log-odds output space)")
    explainer, shap_values = compute_shap_values(model, X_sample)

    validate_setup(model, X_dev, shap_values, X_sample)
    additivity_max_diff = validate_additivity(model, explainer, shap_values, X_sample)

    full_table, top_table = global_importance_table(shap_values, list(X_dev.columns))
    full_table.to_csv(GLOBAL_IMPORTANCE_CSV_PATH, index=False)
    logger.info("Saved %s", GLOBAL_IMPORTANCE_CSV_PATH)

    plot_global_importance(top_table)
    plot_summary_beeswarm(shap_values, X_sample)

    logger.info("Selecting a local explanation example (development applicant, presentation only)")
    demo_idx = select_demo_applicant(model, X_dev)
    demo_row = X_dev.iloc[[demo_idx]]
    demo_id = int(dev_df.iloc[demo_idx]["SK_ID_CURR"])
    local_explanation = explain_applicant(demo_id, demo_row, model, explainer, list(X_dev.columns))

    with open(LOCAL_EXAMPLE_PATH, "w") as f:
        json.dump(local_explanation, f, indent=2, default=float)
    logger.info("Saved %s", LOCAL_EXAMPLE_PATH)

    print(format_local_explanation_text(local_explanation))

    write_report(top_table, local_explanation, additivity_max_diff, len(X_dev.columns))

    print("\nPhase 9 SHAP explainability complete: global + local explanations generated "
          "against the frozen model and 0.110 threshold; no model, threshold, or holdout data changed.")


def write_report(top_table, local_explanation, additivity_max_diff, n_features, output_path=REPORT_PATH):
    lines = [
        "# SHAP Explainability Report (Phase 9)",
        "",
        "## Model used",
        "",
        "Frozen XGBoost (`train_xgboost.MODEL_PARAMS`, imported unchanged), trained "
        "once on all of `dev.parquet` -- the same deterministic training call used "
        "and validated in Phase 8.5. No tuning, no new model family. Raw, "
        "uncalibrated probability output (Phase 7.8 decision, unchanged).",
        "",
        "## Data used",
        "",
        f"**Global**: a deterministic sample of {GLOBAL_SAMPLE_SIZE:,} development "
        f"applicants (`random_state={GLOBAL_SAMPLE_RANDOM_STATE}`), sampled from the "
        "246,008-row dev set used throughout this project. Sampling is used because "
        "(a) SHAP importance rankings stabilize well before the full dev set is used, "
        "and (b) a beeswarm plot over 246,008 points is unreadable and slow to render "
        "-- 2,000 is a standard SHAP sample size. `holdout.parquet` is not read "
        "anywhere in this phase, and no holdout label is used for feature ranking.",
        "",
        f"**Local**: a single development applicant "
        f"(`SK_ID_CURR={local_explanation['applicant_id']}`), for demonstration "
        "purposes only. The `explain_applicant()` function itself is data-source-"
        "agnostic (it accepts any correctly-prepared single-row DataFrame) and is "
        "intended for direct reuse in Phase 10.",
        "",
        "## SHAP method",
        "",
        "`shap.TreeExplainer` (exact, tree-structure-based SHAP for gradient-boosted "
        f"trees) — `shap=={shap.__version__}`, `xgboost` matching "
        "`train_xgboost.MODEL_PARAMS`.",
        "",
        "## SHAP output space",
        "",
        "**Verified empirically against the installed shap/xgboost versions, not "
        "assumed**: `TreeExplainer`'s default `model_output` is `\"raw\"` for "
        "`XGBClassifier`, meaning SHAP values are additive in **log-odds (raw "
        "margin) space**:",
        "",
        "```",
        "base_value + sum(shap_values) == model.predict(X, output_margin=True)",
        "sigmoid(base_value + sum(shap_values)) == model.predict_proba(X)[:, 1]",
        "```",
        "",
        f"Additivity check: `max(|base_value + sum(SHAP) - raw_margin_output|)` = "
        f"**{additivity_max_diff:.2e}** over the {GLOBAL_SAMPLE_SIZE:,}-applicant "
        f"sample (tolerance {ADDITIVITY_TOLERANCE:.0e}) — **passed**.",
        "",
        "Individual SHAP values are reported as log-odds contributions throughout "
        "this project's outputs — never rescaled or described as a literal "
        "percentage-point probability change, since `sigmoid` is nonlinear and a "
        "log-odds decomposition does not translate into an additive probability-space "
        "decomposition. The predicted probability, its distance to the 0.110 "
        "threshold, and the APPROVE/REJECT decision are computed directly via "
        "`model.predict_proba` / `model.predict(..., output_margin=True)` — never by "
        "summing or transforming SHAP values.",
        "",
        "## Global importance results",
        "",
        f"Top {min(15, len(top_table))} features by mean |SHAP value| (log-odds space, "
        f"dev sample, n={GLOBAL_SAMPLE_SIZE:,}):",
        "",
        "| Rank | Feature | Display name | Mean \\|SHAP\\| (log-odds) |",
        "|---|---|---|---|",
    ]
    for _, r in top_table.head(15).iterrows():
        lines.append(f"| {r['rank']} | `{r['feature']}` | {r['display_name']} | {r['mean_abs_shap_value']:.4f} |")

    lines += [
        "",
        "Full top-20 table: `reports/shap_global_importance.csv`. Bar chart: "
        "`reports/shap_global_importance.png`. Beeswarm/summary plot (shows "
        "direction and per-applicant spread, not just magnitude): "
        "`reports/shap_summary.png`.",
        "",
        "These features are associated with higher or lower model risk scores. "
        "This ranking describes the model's learned behavior, not a causal claim "
        "about real-world default risk.",
        "",
        "## Local explanation example",
        "",
        "```",
        format_local_explanation_text(local_explanation),
        "```",
        "",
        "Full structured output: `reports/shap_local_example.json`. The "
        "`explain_applicant()` function producing this is reusable directly (same "
        "signature, same return structure) for Phase 10's Streamlit interface.",
        "",
        "## Relationship to the 0.110 decision threshold",
        "",
        "SHAP explains *why the model produced a given prediction*. The 0.110 "
        "threshold — frozen independently in Phase 8 from dev-only expected-cost "
        "analysis, unrelated to and unaffected by SHAP — is what converts that "
        "prediction into a business decision. These are two separate steps and are "
        "kept conceptually separate throughout this module: `explain_applicant()` "
        "computes the probability and decision first (via the frozen model and "
        "threshold, with no SHAP involvement), and only then computes SHAP "
        "contributions to explain *why* the model arrived at that probability. "
        "SHAP does not determine, adjust, or influence the threshold in any way.",
        "",
        "## Limitations",
        "",
        "- SHAP explains the model's learned behavior, not a real-world causal "
        "mechanism — a feature being a \"risk contributor\" means the model's "
        "score moved in the risk direction when that feature took its observed "
        "value, not that the feature *causes* default.",
        "- SHAP does not establish or guarantee fairness, absence of bias, or "
        "regulatory compliance; no such claim is made here.",
        "- Global importance is computed on a 2,000-applicant deterministic sample, "
        "not the full 246,008-row dev set — chosen for plot readability and "
        "runtime, and unlikely to materially change a top-15/20 ranking, but not "
        "verified against the full-dev ranking in this phase.",
        "- Missing values are passed through to the model exactly as the frozen "
        "pipeline produces them (e.g. `NaN` for bureau features when an applicant "
        "has no bureau history) — `TreeExplainer` attributes these via the tree's "
        "learned default-direction split behavior, not any artificial zero-fill "
        "introduced for explainability.",
        "- Reject inference (documented in `reports/final_model_card.md`) applies "
        "here too: explanations, like the model itself, are only demonstrated "
        "against the historically-approved applicant population.",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()

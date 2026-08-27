"""
Phase 7.8 Part 1 & 3: Bootstrap-validate the isotonic calibration decision.

reports/holdout_calibration_results.json (Phase 7.5) reports a single-point
Brier score difference between raw XGBoost and isotonic calibration on
holdout (0.066840 vs. 0.066826, a gap of 0.000014). A single point estimate
cannot say whether that gap reflects a real effect or sampling noise. This
script reproduces the frozen pipeline's per-row holdout predictions (no
retraining in the sense of a new configuration -- same MODEL_PARAMS, same
dev.parquet, same fitted isotonic calibrator from dev OOF, same
holdout.parquet -- just persisting values that were previously only
aggregated), verifies they reproduce the existing aggregate metrics, then
bootstraps the paired Brier(raw) - Brier(isotonic) difference.

It also computes a constant-probability (dev prevalence) baseline Brier
score on holdout, for context on how much either model improves over the
simplest possible predictor.

holdout.parquet is read once, only for scoring -- never for .fit().
"""

import json
import logging
import os

import numpy as np
from sklearn.metrics import brier_score_loss

from calibrate_model import fit_calibrators, generate_oof_predictions
from data_quality import DEV_PATH
from evaluate_holdout_calibration import load_holdout_data, train_final_model
from profile_data import PROJECT_ROOT, get_connection
from train_xgboost import load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "calibration_bootstrap_results.json")
EXISTING_HOLDOUT_RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "holdout_calibration_results.json")

N_ITERATIONS = 1000
RANDOM_STATE = 42
CONSISTENCY_TOLERANCE = 1e-6  # allows for tiny float non-determinism (e.g. multi-threaded hist reductions)


def reproduce_holdout_predictions():
    """Reproduce raw_xgb/isotonic holdout predictions from the frozen pipeline.

    Identical procedure to src/evaluate_holdout_calibration.py: dev-only
    XGBoost training and OOF generation, dev-OOF-only isotonic fitting,
    holdout used exclusively for .predict()/.predict_proba().
    """
    dev_df = load_dev_data()

    oof_df = generate_oof_predictions(dev_df)
    _oof_predictions, calibrators = fit_calibrators(oof_df)

    X_dev, y_dev = split_features_target(dev_df)
    X_dev, _, _ = prepare_categoricals(X_dev)
    final_model = train_final_model(X_dev, y_dev)

    holdout_df = load_holdout_data()
    X_holdout, y_holdout = split_features_target(holdout_df)
    X_holdout, _, _ = prepare_categoricals(X_holdout)

    raw_holdout = final_model.predict_proba(X_holdout)[:, 1]
    isotonic_holdout = calibrators["isotonic"].predict(raw_holdout)

    return y_holdout.values, raw_holdout, isotonic_holdout, dev_df["TARGET"].mean()


def verify_against_existing_results(y_holdout, raw_holdout, isotonic_holdout):
    """Confirm reproduced predictions match Phase 7.5's saved aggregate metrics.

    A mismatch here would mean the "frozen" pipeline isn't actually
    deterministic -- this must be checked, not assumed, before trusting
    the reproduced per-row values for the bootstrap below.
    """
    with open(EXISTING_HOLDOUT_RESULTS_PATH) as f:
        existing = json.load(f)

    reproduced_raw_brier = brier_score_loss(y_holdout, raw_holdout)
    reproduced_iso_brier = brier_score_loss(y_holdout, isotonic_holdout)

    raw_diff = abs(reproduced_raw_brier - existing["raw_xgb"]["brier_score"])
    iso_diff = abs(reproduced_iso_brier - existing["isotonic"]["brier_score"])

    logger.info(
        "Consistency check -- raw Brier: reproduced=%.8f, existing=%.8f, diff=%.2e",
        reproduced_raw_brier, existing["raw_xgb"]["brier_score"], raw_diff,
    )
    logger.info(
        "Consistency check -- isotonic Brier: reproduced=%.8f, existing=%.8f, diff=%.2e",
        reproduced_iso_brier, existing["isotonic"]["brier_score"], iso_diff,
    )

    assert raw_diff < CONSISTENCY_TOLERANCE, (
        f"Reproduced raw XGBoost holdout predictions do not match the saved "
        f"Phase 7.5 results (diff={raw_diff:.2e}) -- the pipeline is not "
        f"reproducing the frozen model faithfully."
    )
    assert iso_diff < CONSISTENCY_TOLERANCE, (
        f"Reproduced isotonic holdout predictions do not match the saved "
        f"Phase 7.5 results (diff={iso_diff:.2e}) -- the pipeline is not "
        f"reproducing the frozen model faithfully."
    )
    logger.info("Reproduced predictions match saved Phase 7.5 results within tolerance.")


def bootstrap_brier_difference(y, raw, isotonic, n_iterations=N_ITERATIONS, random_state=RANDOM_STATE):
    """Paired bootstrap of Brier(raw) - Brier(isotonic).

    Paired (same resampled row indices for both methods each iteration) so
    the difference reflects the two methods' behavior on identical
    resampled applicants, not independent sampling noise from each side.
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    diffs = np.empty(n_iterations)

    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        y_s, raw_s, iso_s = y[idx], raw[idx], isotonic[idx]
        brier_raw = np.mean((raw_s - y_s) ** 2)
        brier_iso = np.mean((iso_s - y_s) ** 2)
        diffs[i] = brier_raw - brier_iso

    return diffs


def interpret(mean_diff, ci_lower, ci_upper):
    """CI entirely > 0 => isotonic meaningfully better. CI spans 0 => noise. CI entirely < 0 => isotonic worse."""
    if ci_lower <= 0 <= ci_upper:
        return (
            "NOT statistically meaningful: the 95% bootstrap CI for "
            "Brier(raw) - Brier(isotonic) includes 0, so the observed "
            "point-estimate improvement is indistinguishable from sampling "
            "noise at this sample size."
        )
    if ci_lower > 0:
        return (
            "Statistically meaningful improvement: the 95% bootstrap CI for "
            "Brier(raw) - Brier(isotonic) is entirely positive, i.e. "
            "isotonic's lower Brier score is unlikely to be sampling noise."
        )
    return (
        "Statistically meaningful REGRESSION: the 95% bootstrap CI for "
        "Brier(raw) - Brier(isotonic) is entirely negative, i.e. isotonic "
        "is reliably worse than raw XGBoost."
    )


def main():
    logger.info("Reproducing frozen-pipeline holdout predictions (dev-only fit, holdout scored once)")
    y_holdout, raw_holdout, isotonic_holdout, dev_prevalence = reproduce_holdout_predictions()

    verify_against_existing_results(y_holdout, raw_holdout, isotonic_holdout)

    logger.info("Running %d-iteration paired bootstrap (random_state=%d)", N_ITERATIONS, RANDOM_STATE)
    diffs = bootstrap_brier_difference(y_holdout, raw_holdout, isotonic_holdout)

    mean_diff = float(np.mean(diffs))
    ci_lower = float(np.percentile(diffs, 2.5))
    ci_upper = float(np.percentile(diffs, 97.5))
    interpretation = interpret(mean_diff, ci_lower, ci_upper)

    # Part 3: constant-probability baseline, p = dev prevalence (never derived from holdout)
    constant_proba = np.full(len(y_holdout), dev_prevalence)
    constant_brier = float(brier_score_loss(y_holdout, constant_proba))
    raw_brier = float(brier_score_loss(y_holdout, raw_holdout))
    isotonic_brier = float(brier_score_loss(y_holdout, isotonic_holdout))

    results = {
        "bootstrap": {
            "n_iterations": N_ITERATIONS,
            "random_state": RANDOM_STATE,
            "mean_difference_raw_minus_isotonic": mean_diff,
            "ci_2_5_pct": ci_lower,
            "ci_97_5_pct": ci_upper,
            "interpretation": interpretation,
        },
        "brier_comparison": {
            "constant_predictor": {
                "p": float(dev_prevalence),
                "brier_score": constant_brier,
                "note": "p = dev.parquet TARGET mean; never derived from holdout",
            },
            "raw_xgb": {
                "brier_score": raw_brier,
                "relative_improvement_vs_constant_pct": (constant_brier - raw_brier) / constant_brier * 100,
            },
            "isotonic": {
                "brier_score": isotonic_brier,
                "relative_improvement_vs_constant_pct": (constant_brier - isotonic_brier) / constant_brier * 100,
            },
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved bootstrap results to %s", RESULTS_PATH)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

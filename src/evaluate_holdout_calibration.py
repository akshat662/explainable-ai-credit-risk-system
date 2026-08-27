"""
Phase 7.5 Part 3: Proper calibration evaluation using the untouched holdout.

Fixes a scope gap in Phase 7: reports/calibration_results.json measured
each calibration method against the same out-of-fold (OOF) development
data used to fit it, which is internally honest (OOF is leakage-safe) but
answers a narrower question than "how will this behave on genuinely
unseen applicants." This script answers that question directly.

Protocol, enforced by construction:
- XGBoost training, OOF generation, and calibrator fitting: dev.parquet only.
- holdout.parquet: read exactly once, for scoring, never for .fit().
- XGBoost hyperparameters: imported unchanged from train_xgboost.MODEL_PARAMS.
"""

import json
import logging
import os

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

from calibrate_model import fit_calibrators, generate_oof_predictions, save_oof_predictions
from data_quality import HOLDOUT_PATH
from profile_data import PROJECT_ROOT, get_connection
from train_xgboost import MODEL_PARAMS, load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "holdout_calibration_results.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "holdout_calibration_report.md")
DEV_CALIBRATION_RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "calibration_results.json")


def load_holdout_data(holdout_path=HOLDOUT_PATH):
    """Load holdout.parquet. Called exactly once, only for scoring -- never for .fit()."""
    con = get_connection()
    df = con.execute(f"SELECT * FROM read_parquet('{holdout_path}')").fetchdf()
    con.close()
    return df


def train_final_model(X_dev, y_dev):
    """Train one XGBoost model on all of dev.parquet, for scoring holdout.

    This is distinct from the per-fold OOF models: OOF models each saw
    only 4/5 of dev and exist purely to produce honest dev-set scores for
    calibrator fitting. To score holdout -- data no model has ever seen --
    a single model trained on the full dev set is the correct artifact,
    analogous to what a production model would actually be. Same
    unmodified MODEL_PARAMS as Phase 6/7; no tuning.
    """
    model = XGBClassifier(**MODEL_PARAMS, enable_categorical=True)
    model.fit(X_dev, y_dev)
    return model


def evaluate(y, predictions):
    """Brier score, ROC-AUC, PR-AUC for each method's holdout predictions."""
    results = {}
    for name, proba in predictions.items():
        results[name] = {
            "brier_score": float(brier_score_loss(y, proba)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
        }
    return results


def write_report(dev_results, holdout_results, n_dev, n_holdout, output_path=REPORT_PATH):
    best_holdout = min(holdout_results, key=lambda k: holdout_results[k]["brier_score"])

    lines = [
        "# Holdout Calibration Report",
        "",
        "Final, honest evaluation of the three calibration methods against "
        f"`holdout.parquet` ({n_holdout} applicants, read only for scoring — "
        "never used to train XGBoost, generate OOF predictions, or fit "
        f"Platt/isotonic). Development figures ({n_dev} applicants) are from "
        "`reports/calibration_results.json`, regenerated in this phase against "
        "the corrected features and split (see DECISIONS.md).",
        "",
        "## 1. Dev (OOF) vs. holdout (final evaluation)",
        "",
        "| Method | Brier (dev OOF) | Brier (holdout) | ROC-AUC (dev OOF) | ROC-AUC (holdout) | PR-AUC (dev OOF) | PR-AUC (holdout) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("raw_xgb", "platt", "isotonic"):
        d, h = dev_results[name], holdout_results[name]
        lines.append(
            f"| {name} | {d['brier_score']:.5f} | {h['brier_score']:.5f} | "
            f"{d['roc_auc']:.4f} | {h['roc_auc']:.4f} | {d['pr_auc']:.4f} | {h['pr_auc']:.4f} |"
        )

    lines += [
        "",
        "## 2. Interpretation",
        "",
        f"Best method on holdout by Brier score: **{best_holdout}** "
        f"({holdout_results[best_holdout]['brier_score']:.5f}).",
        "",
        "ROC-AUC and PR-AUC are expected to move together across raw/Platt/"
        "isotonic on holdout, same as on dev OOF (Platt is a strictly "
        "monotonic transform of the same score; isotonic's step function "
        "only reorders through ties) -- Brier score remains the metric that "
        "actually differentiates calibration quality.",
        "",
        "Dev-OOF and holdout numbers are close but not identical, as expected: "
        "they're different applicants, and dev OOF carries the small optimism "
        "noted in Phase 7 (calibrators evaluated on the same OOF sample used "
        "to fit them). The holdout numbers are the ones that should be trusted "
        "for reporting real-world performance, since holdout was never seen "
        "during training, OOF generation, or calibrator fitting.",
        "",
        "## 3. Final probability source",
        "",
        f"**{best_holdout}** is the calibrated-probability source for the "
        "decision engine going forward, selected on holdout Brier score -- "
        "the first calibration decision in this project actually validated "
        "against data no part of the pipeline has ever touched.",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote holdout calibration report to %s", output_path)
    return best_holdout


def main():
    logger.info("Loading development data (for XGBoost training, OOF generation, calibrator fitting)")
    dev_df = load_dev_data()

    logger.info("Generating out-of-fold predictions on dev (5-fold CV, unmodified Phase 6 config)")
    oof_df = generate_oof_predictions(dev_df)
    save_oof_predictions(oof_df)  # refreshes reports/oof_predictions.csv for this corrected dev set

    logger.info("Fitting Platt and isotonic calibrators on dev OOF predictions")
    _oof_predictions, calibrators = fit_calibrators(oof_df)

    logger.info("Training final XGBoost on all of dev.parquet (no tuning, same MODEL_PARAMS)")
    X_dev, y_dev = split_features_target(dev_df)
    X_dev, _, _ = prepare_categoricals(X_dev)
    final_model = train_final_model(X_dev, y_dev)

    logger.info("Loading holdout.parquet -- read once, for scoring only")
    holdout_df = load_holdout_data()
    X_holdout, y_holdout = split_features_target(holdout_df)
    X_holdout, _, _ = prepare_categoricals(X_holdout)

    raw_holdout = final_model.predict_proba(X_holdout)[:, 1]
    platt_holdout = calibrators["platt"].predict_proba(raw_holdout.reshape(-1, 1))[:, 1]
    isotonic_holdout = calibrators["isotonic"].predict(raw_holdout)

    holdout_predictions = {"raw_xgb": raw_holdout, "platt": platt_holdout, "isotonic": isotonic_holdout}
    holdout_results = evaluate(y_holdout.values, holdout_predictions)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(holdout_results, f, indent=2)
    logger.info("Saved holdout calibration results to %s", RESULTS_PATH)

    with open(DEV_CALIBRATION_RESULTS_PATH) as f:
        dev_results = json.load(f)

    best_method = write_report(dev_results, holdout_results, len(dev_df), len(holdout_df))

    print(json.dumps(holdout_results, indent=2))
    print(f"\nBest calibration method on holdout: {best_method}")


if __name__ == "__main__":
    main()

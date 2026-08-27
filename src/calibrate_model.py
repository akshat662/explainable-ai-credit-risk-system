"""
Phase 7: Probability calibration.

Converts raw XGBoost scores into genuinely calibrated default
probabilities. Calibration is fit and evaluated entirely on out-of-fold
XGBoost predictions from dev.parquet -- holdout.parquet is never touched,
and the underlying XGBoost model is retrained per fold with the exact
Phase 6 hyperparameters (imported, not restated), unchanged.
"""

import json
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from profile_data import PROJECT_ROOT
from train_xgboost import MODEL_PARAMS, load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OOF_PATH = os.path.join(PROJECT_ROOT, "reports", "oof_predictions.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "calibration_results.json")
CURVE_PATH = os.path.join(PROJECT_ROOT, "reports", "calibration_curve.png")

N_SPLITS = 5
RANDOM_STATE = 42


def generate_oof_predictions(df):
    """Train Phase 6's exact XGBoost config per fold, predict on each held-out fold.

    Reuses train_xgboost.MODEL_PARAMS/prepare_categoricals unmodified --
    Phase 7 calibrates these scores, it does not re-tune the model that
    produces them. Every row gets exactly one prediction, made by a model
    that never saw that row during training.
    """
    X, y = split_features_target(df)
    X, _, _ = prepare_categoricals(X)

    oof_proba = np.zeros(len(df))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model = XGBClassifier(**MODEL_PARAMS, enable_categorical=True)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
        logger.info("Fold %d/%d complete", fold, N_SPLITS)

    return pd.DataFrame({
        "SK_ID_CURR": df["SK_ID_CURR"].values,
        "TARGET": y.values,
        "xgb_probability": oof_proba,
    })


def save_oof_predictions(oof_df, output_path=OOF_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    oof_df.to_csv(output_path, index=False)
    logger.info("Saved OOF predictions to %s", output_path)


def fit_calibrators(oof_df):
    """Fit Platt scaling (LogisticRegression) and isotonic regression on OOF predictions.

    Both are fit directly on the full OOF set. This does not reintroduce
    the leakage OOF exists to avoid -- each raw score was already produced
    by a model that never saw that row -- but Step 3's metrics are then
    computed on this same OOF set for every method, including the two
    calibrators fit on it. That carries a small optimism relative to a
    further, fully disjoint calibration-eval split; documented here as a
    known limitation of this phase's scope rather than assumed away.
    """
    y = oof_df["TARGET"].values
    raw = oof_df["xgb_probability"].values

    platt = LogisticRegression()
    platt.fit(raw.reshape(-1, 1), y)
    platt_proba = platt.predict_proba(raw.reshape(-1, 1))[:, 1]

    isotonic = IsotonicRegression(out_of_bounds="clip")
    isotonic.fit(raw, y)
    isotonic_proba = isotonic.predict(raw)

    return {"raw_xgb": raw, "platt": platt_proba, "isotonic": isotonic_proba}


def evaluate_calibration(y, predictions):
    """Brier score, ROC-AUC, PR-AUC for each calibration method."""
    results = {}
    for name, proba in predictions.items():
        results[name] = {
            "brier_score": float(brier_score_loss(y, proba)),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
        }
    return results


def plot_calibration_curve(y, predictions, output_path=CURVE_PATH):
    """Reliability diagram comparing raw XGB, Platt, and isotonic calibration."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")

    for name, proba in predictions.items():
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed default rate)")
    ax.set_title("Calibration Curve: Raw XGBoost vs. Platt vs. Isotonic")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved calibration curve to %s", output_path)


def main():
    logger.info("Loading development data")
    df = load_dev_data()

    logger.info("Generating out-of-fold predictions (%d-fold CV)", N_SPLITS)
    oof_df = generate_oof_predictions(df)
    save_oof_predictions(oof_df)

    predictions = fit_calibrators(oof_df)
    results = evaluate_calibration(oof_df["TARGET"].values, predictions)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved calibration results to %s", RESULTS_PATH)

    plot_calibration_curve(oof_df["TARGET"].values, predictions)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

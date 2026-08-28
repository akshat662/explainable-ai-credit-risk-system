"""
Phase 8.5: Final holdout evaluation.

This is the FIRST phase in the project allowed to read holdout.parquet.
That is intentional and load-bearing: model configuration (Phase 6),
probability source (Phase 7.8), and business threshold (Phase 8) were all
frozen using dev-only data before this file ever opens holdout.parquet.

Scope is deliberately narrow: train exactly one XGBoost model on all of
dev.parquet (frozen MODEL_PARAMS, no tuning, no CV), score holdout once,
and apply the frozen 0.110 threshold once. No threshold sweep, no
calibration, no new decisions. Everything reused from prior phases is
imported, not redefined, so this evaluation exactly matches the frozen
pipeline it is measuring.
"""

import inspect
import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from data_quality import HOLDOUT_PATH
from decision_threshold import LGD, MARGIN, evaluate_threshold
from evaluate_holdout_calibration import load_holdout_data, train_final_model
from profile_data import PROJECT_ROOT
from train_xgboost import DEV_PATH, load_dev_data, prepare_categoricals, split_features_target

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "final_holdout_results.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "final_holdout_report.md")
CONFUSION_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "final_holdout_confusion_matrix.png")
SUMMARY_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "final_holdout_decision_summary.png")

# Frozen from Phase 8. Not derived here, not swept here, not changed here.
FROZEN_THRESHOLD = 0.110
EXPECTED_DEV_ROWS = 246008
EXPECTED_HOLDOUT_ROWS = 61503


def verify_no_holdout_threshold_search():
    """Static self-check: this file must not contain a threshold sweep.

    Reads this module's own source and asserts it never imports or calls
    decision_threshold.sweep_thresholds / select_threshold (Phase 8's
    sweep machinery), and that evaluate_threshold is called on holdout
    data exactly once. This is a repo-hygiene assertion about the code
    itself, not just its output -- it fails loudly if a future edit
    accidentally reintroduces a sweep here.

    This function's own source is excluded from the scan: it necessarily
    mentions the forbidden names in order to check for them, which would
    otherwise make the check trivially fail against itself.
    """
    full_source = inspect.getsource(inspect.getmodule(verify_no_holdout_threshold_search))
    own_source = inspect.getsource(verify_no_holdout_threshold_search)
    source = full_source.replace(own_source, "")

    assert "sweep_thresholds" not in source, (
        "evaluate_final_holdout.py must never import or call sweep_thresholds -- "
        "threshold selection is frozen from Phase 8, not re-derived here."
    )
    assert "select_threshold" not in source, (
        "evaluate_final_holdout.py must never call select_threshold -- "
        "the threshold (0.110) is already selected and frozen."
    )

    n_holdout_evaluate_calls = source.count("evaluate_threshold(y_holdout")
    assert n_holdout_evaluate_calls == 1, (
        f"Expected exactly one evaluate_threshold() call against holdout data, "
        f"found {n_holdout_evaluate_calls} -- this must be a single fixed-threshold "
        f"application, not a sweep."
    )

    logger.info("Static check passed: no threshold sweep/selection logic present; "
                "evaluate_threshold(holdout) called exactly once.")


def main():
    assert FROZEN_THRESHOLD == 0.110, "threshold must exactly equal the Phase 8 frozen value 0.110"
    assert LGD == 0.60, "LGD must exactly equal the frozen value 0.60"
    assert MARGIN == 0.08, "margin must exactly equal the frozen value 0.08"

    verify_no_holdout_threshold_search()

    # --- DEV: model fitting only. No CV, no tuning, no calibration. ---
    logger.info("Loading dev.parquet and training exactly one XGBoost model (frozen MODEL_PARAMS)")
    dev_df = load_dev_data(DEV_PATH)
    assert len(dev_df) == EXPECTED_DEV_ROWS, f"dev row count {len(dev_df)} != expected {EXPECTED_DEV_ROWS}"
    assert dev_df["SK_ID_CURR"].is_unique, "duplicate SK_ID_CURR found in dev.parquet"

    X_dev, y_dev = split_features_target(dev_df)
    X_dev, _, _ = prepare_categoricals(X_dev)
    final_model = train_final_model(X_dev, y_dev)

    # --- HOLDOUT: read for the first time in this project, scoring only. ---
    logger.info("Loading holdout.parquet -- first read of holdout in this project")
    holdout_df = load_holdout_data(HOLDOUT_PATH)
    assert len(holdout_df) == EXPECTED_HOLDOUT_ROWS, (
        f"holdout row count {len(holdout_df)} != expected {EXPECTED_HOLDOUT_ROWS}"
    )
    assert holdout_df["SK_ID_CURR"].is_unique, "duplicate SK_ID_CURR found in holdout.parquet"

    overlap = set(dev_df["SK_ID_CURR"]) & set(holdout_df["SK_ID_CURR"])
    assert len(overlap) == 0, f"{len(overlap)} SK_ID_CURR values appear in both dev and holdout"

    X_holdout, y_holdout = split_features_target(holdout_df)
    X_holdout, _, _ = prepare_categoricals(X_holdout)

    p_holdout = final_model.predict_proba(X_holdout)[:, 1]
    y_holdout_arr = y_holdout.astype(int).values

    assert len(p_holdout) == len(holdout_df), "prediction count must equal holdout row count"
    assert np.all((p_holdout >= 0) & (p_holdout <= 1)), "all probabilities must be in [0, 1]"

    # --- Predictive performance ---
    n_holdout = len(holdout_df)
    holdout_prevalence = float(np.mean(y_holdout_arr))
    roc_auc = float(roc_auc_score(y_holdout_arr, p_holdout))
    pr_auc = float(average_precision_score(y_holdout_arr, p_holdout))
    brier = float(brier_score_loss(y_holdout_arr, p_holdout))

    dev_prevalence = float(dev_df["TARGET"].mean())
    constant_baseline_proba = np.full(n_holdout, dev_prevalence)
    constant_brier = float(brier_score_loss(y_holdout_arr, constant_baseline_proba))

    logger.info(
        "Holdout predictive performance: n=%d, prevalence=%.4f%%, ROC-AUC=%.4f, PR-AUC=%.4f, Brier=%.6f",
        n_holdout, holdout_prevalence * 100, roc_auc, pr_auc, brier,
    )

    # --- Business decision: apply the frozen threshold exactly once. ---
    decision_result = evaluate_threshold(y_holdout_arr, p_holdout, FROZEN_THRESHOLD, lgd=LGD, margin=MARGIN)

    approve_mask = p_holdout < FROZEN_THRESHOLD
    reject_mask = ~approve_mask
    default_rate_rejected = (
        float(np.mean(y_holdout_arr[reject_mask])) if reject_mask.sum() > 0 else float("nan")
    )

    assert (
        decision_result["true_positives"] + decision_result["false_positives"]
        + decision_result["true_negatives"] + decision_result["false_negatives"]
    ) == n_holdout, "confusion matrix must sum to holdout row count"
    assert abs((decision_result["approval_rate"] + decision_result["rejection_rate"]) - 1.0) < 1e-9, (
        "approval_rate + rejection_rate must equal 1"
    )

    # --- Cost baselines, evaluated with holdout's actual labels. ---
    approve_all_cost = float(np.mean(np.where(y_holdout_arr == 1, LGD, -MARGIN)))
    reject_all_cost = 0.0

    logger.info(
        "Holdout business decision at frozen threshold %.3f: approval_rate=%.2f%%, "
        "mean_expected_cost=%.6f (approve-all=%.6f, reject-all=%.6f)",
        FROZEN_THRESHOLD, decision_result["approval_rate"] * 100,
        decision_result["mean_expected_cost"], approve_all_cost, reject_all_cost,
    )

    results = {
        "methodology_note": (
            "The 0.110 threshold was selected using dev-only data in Phase 8, "
            "before holdout.parquet was ever read. This evaluation applies that "
            "pre-specified policy to holdout exactly once; it does not select, "
            "sweep, or optimize a threshold on holdout."
        ),
        "counts": {
            "n_dev": len(dev_df),
            "n_holdout": n_holdout,
            "dev_holdout_overlap": len(overlap),
        },
        "predictive_performance": {
            "holdout_default_prevalence": holdout_prevalence,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "brier_score": brier,
            "constant_baseline_brier_score": constant_brier,
            "constant_baseline_probability_source": "dev.parquet TARGET mean (frozen, not re-derived from holdout)",
        },
        "business_decision": {
            "frozen_threshold": FROZEN_THRESHOLD,
            "lgd": LGD,
            "margin": MARGIN,
            **decision_result,
            "default_rate_rejected": default_rate_rejected,
        },
        "cost_comparison": {
            "approve_all_mean_cost": approve_all_cost,
            "reject_all_mean_cost": reject_all_cost,
            "frozen_threshold_mean_cost": decision_result["mean_expected_cost"],
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Saved %s", RESULTS_PATH)

    plot_confusion_matrix(decision_result, n_holdout)
    plot_decision_summary(decision_result, approve_all_cost, reject_all_cost)
    write_markdown_report(results, dev_prevalence)

    print(json.dumps(results, indent=2, default=float))
    print(
        "\nPhase 8.5 complete: the frozen 0.110 decision policy has been evaluated "
        "once on the untouched holdout; no further threshold optimization was performed."
    )


def plot_confusion_matrix(decision_result, n_holdout, output_path=CONFUSION_PLOT_PATH):
    matrix = np.array([
        [decision_result["true_negatives"], decision_result["false_positives"]],
        [decision_result["false_negatives"], decision_result["true_positives"]],
    ])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Approved", "Rejected"])
    ax.set_yticklabels(["Non-default (0)", "Default (1)"])
    ax.set_xlabel("Decision (positive = default/risky)")
    ax.set_ylabel("Actual outcome")
    ax.set_title(f"Holdout Confusion Matrix @ threshold={FROZEN_THRESHOLD:.3f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                     color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_decision_summary(decision_result, approve_all_cost, reject_all_cost, output_path=SUMMARY_PLOT_PATH):
    policies = ["Approve\nall", "Reject\nall", f"Frozen\nthreshold ({FROZEN_THRESHOLD:.3f})"]
    costs = [approve_all_cost, reject_all_cost, decision_result["mean_expected_cost"]]
    colors = ["tab:orange", "tab:gray", "tab:green"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(policies, costs, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean expected cost per applicant (fraction of exposure)")
    ax.set_title("Holdout Policy Comparison (lower is better)")
    y_min, y_max = min(costs), max(costs)
    headroom = max(0.1 * (y_max - y_min), 0.005)
    ax.set_ylim(y_min - headroom, y_max + headroom)
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, cost, f"{cost:.5f}",
                 ha="center", va="bottom" if cost >= 0 else "top")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def write_markdown_report(results, dev_prevalence, output_path=REPORT_PATH):
    pred = results["predictive_performance"]
    biz = results["business_decision"]
    cost = results["cost_comparison"]
    counts = results["counts"]

    dev_roc_auc, dev_roc_auc_std = 0.7634, 0.0044
    dev_pr_auc, dev_pr_auc_std = 0.2499, 0.0087

    lines = [
        "# Final Holdout Evaluation Report (Phase 8.5)",
        "",
        "## 1. Executive summary",
        "",
        f"The frozen model + frozen 0.110 decision threshold were evaluated exactly "
        f"once on the untouched holdout set ({counts['n_holdout']:,} applicants). "
        f"Holdout ROC-AUC {pred['roc_auc']:.4f}, PR-AUC {pred['pr_auc']:.4f}, Brier "
        f"score {pred['brier_score']:.6f}. At the frozen threshold, "
        f"{biz['approval_rate']*100:.2f}% of applicants are approved with a mean "
        f"expected cost of {biz['mean_expected_cost']:.6f} per applicant -- better "
        f"than both the approve-all ({cost['approve_all_mean_cost']:.6f}) and "
        f"reject-all ({cost['reject_all_mean_cost']:.6f}) baselines. "
        "**This is a report of how a pre-specified policy generalized, not a "
        "search for a better one.**",
        "",
        "## 2. Frozen model configuration",
        "",
        "XGBoost, trained once on all of `dev.parquet` "
        "(`n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, "
        "colsample_bytree=0.8, tree_method=\"hist\", eval_metric=\"aucpr\", "
        "random_state=42`), imported unchanged from `train_xgboost.MODEL_PARAMS`. "
        "No cross-validation, no hyperparameter tuning, no calibration in this phase.",
        "",
        "## 3. Frozen feature state",
        "",
        "147 model features (120 raw `application_train` + 26 engineered bureau/"
        "previous-application/ratio features + 1 `days_employed_anomaly`), from "
        "the current, unmodified `src/features.py` / `src/data_quality.py`. "
        "Negative `AMT_CREDIT_SUM_DEBT` clipped to zero with `bureau_had_negative_debt` "
        "retained (Phase 7.5, unchanged). Dev/holdout split via the "
        "`ORDER BY SK_ID_CURR`-stabilized, deterministic 80/20 stratified split "
        "(Phase 7.5, unchanged).",
        "",
        "## 4. Frozen probability choice",
        "",
        "Raw, uncalibrated XGBoost probability (Phase 7.8 decision, statistically "
        "validated by bootstrap, unchanged). No isotonic calibration, no Platt "
        "scaling applied anywhere in this evaluation.",
        "",
        "## 5. Frozen business threshold",
        "",
        f"**Threshold = {biz['frozen_threshold']:.3f}** (reject if `p_default >= "
        f"{biz['frozen_threshold']:.3f}`), selected on dev-only data in Phase 8 under "
        f"LGD={biz['lgd']}, margin={biz['margin']}. Not re-derived, re-swept, or "
        "adjusted in this phase.",
        "",
        "## 6. Why holdout evaluation happens only now",
        "",
        "This is the first phase in the project permitted to read "
        "`holdout.parquet`. Model configuration (Phase 6), probability source "
        "(Phase 7.8, bootstrap-validated), and business threshold (Phase 8, "
        "dev-only sweep) were all frozen *before* this file's first "
        "`read_parquet('holdout.parquet')` call. Reading holdout any earlier "
        "would have let it silently influence a decision it exists to test "
        "honestly.",
        "",
        "**DEV** was used for: XGBoost model comparison, calibration analysis, "
        "bootstrap significance testing, threshold sweep and selection, "
        "sensitivity analysis. **HOLDOUT** is used for: final evaluation only "
        "-- nothing computed on holdout in this phase feeds back into any prior "
        "decision.",
        "",
        "## 7. Holdout predictive performance",
        "",
        f"- Holdout applicants: **{counts['n_holdout']:,}**",
        f"- Default prevalence: **{pred['holdout_default_prevalence']*100:.4f}%**",
        f"- ROC-AUC: **{pred['roc_auc']:.4f}**",
        f"- PR-AUC: **{pred['pr_auc']:.4f}**",
        f"- Brier score: **{pred['brier_score']:.6f}**",
        f"- Constant base-rate predictor Brier score (p = dev prevalence "
        f"{dev_prevalence*100:.4f}%, frozen, not re-derived from holdout): "
        f"**{pred['constant_baseline_brier_score']:.6f}**",
        "",
        "**Do not compare these directly to dev CV metrics as though they should "
        "match.** Dev CV metrics "
        f"(ROC-AUC {dev_roc_auc:.4f} ± {dev_roc_auc_std:.4f}, PR-AUC {dev_pr_auc:.4f} "
        f"± {dev_pr_auc_std:.4f}) are averaged over 5 folds, each trained on 80% of "
        "dev and validated on a held-out 20% *of dev*. Holdout metrics come from a "
        "single model trained on 100% of dev, scored once on an entirely separate "
        "61,503-applicant sample. Different training data, different evaluation "
        "data, and no folding/averaging on the holdout side -- some difference is "
        "expected by construction, not a sign of a problem either way.",
        "",
        "## 8. Holdout business-decision performance",
        "",
        f"At threshold {biz['frozen_threshold']:.3f} (**\"positive\" = default/risky "
        "applicant**, TARGET == 1; a REJECT decision is a positive prediction):",
        "",
        f"- Approved: **{biz['n_approved']:,}** ({biz['approval_rate']*100:.2f}%)",
        f"- Rejected: **{biz['n_rejected']:,}** ({biz['rejection_rate']*100:.2f}%)",
        f"- Default rate among approved: **{biz['default_rate_approved']*100:.4f}%**",
        f"- Default rate among rejected: **{biz['default_rate_rejected']*100:.4f}%**",
        f"- Defaults captured by rejection: **{biz['defaults_captured']:,}** "
        f"({biz['defaults_captured_rate']*100:.2f}% of all holdout defaults)",
        f"- Confusion matrix: TP={biz['true_positives']:,}, "
        f"FP={biz['false_positives']:,}, TN={biz['true_negatives']:,}, "
        f"FN={biz['false_negatives']:,}",
        f"- Sanity check: TP+FP+TN+FN = {biz['true_positives']+biz['false_positives']+biz['true_negatives']+biz['false_negatives']:,} "
        f"= holdout row count ({counts['n_holdout']:,})",
        "",
        "See `reports/final_holdout_confusion_matrix.png`.",
        "",
        "## 9. Cost comparison",
        "",
        "Same economic formulation locked in Phase 8 "
        "(`expected_cost_approve(p) = p*LGD - (1-p)*margin`), evaluated against "
        "*realized* holdout outcomes:",
        "",
        f"- Approve everyone: **{cost['approve_all_mean_cost']:.6f}** per applicant",
        f"- Reject everyone: **{cost['reject_all_mean_cost']:.6f}** per applicant",
        f"- Frozen {biz['frozen_threshold']:.3f} threshold: **{cost['frozen_threshold_mean_cost']:.6f}** per applicant",
        "",
        "See `reports/final_holdout_decision_summary.png`.",
        "",
        "## 10. Dev vs. holdout comparison",
        "",
        "| Metric | Dev (CV) | Holdout |",
        "|---|---|---|",
        f"| ROC-AUC | {dev_roc_auc:.4f} ± {dev_roc_auc_std:.4f} | {pred['roc_auc']:.4f} |",
        f"| PR-AUC | {dev_pr_auc:.4f} ± {dev_pr_auc_std:.4f} | {pred['pr_auc']:.4f} |",
        f"| Default prevalence | {dev_prevalence*100:.4f}% | {pred['holdout_default_prevalence']*100:.4f}% |",
        "",
        "(Dev figures above are the Phase 7.6 authoritative CV benchmarks, restated "
        "for reference -- not recomputed in this phase.)",
        "",
        "## 11. Generalization interpretation",
        "",
        "The threshold of 0.110 was frozen using dev data before holdout "
        "evaluation. These holdout results measure how that pre-specified "
        "policy generalizes -- they are not evidence that 0.110 is optimal on "
        "holdout, and no alternative holdout threshold was computed or "
        "considered. "
        + (
            "Holdout's frozen-threshold cost is *better* than dev's cost at the "
            "same threshold, which is a reportable observation, not a reason to "
            "revisit the threshold."
            if cost["frozen_threshold_mean_cost"] < -0.038287 else
            "Holdout's frozen-threshold cost is *worse* than dev's cost at the "
            "same threshold, which is reported honestly here, not a reason to "
            "revisit the threshold."
        ),
        "",
        "## 12. Limitations",
        "",
        "- This is a single holdout evaluation, not a repeated or bootstrapped "
        "one -- the reported numbers are one realization, same caveat as noted "
        "throughout this project for dev-based estimates.",
        "- LGD and margin remain fixed, illustrative constants, not measured "
        "quantities; see `reports/threshold_analysis.md` Section 11 for the full "
        "list of economic-model limitations, which apply identically here.",
        "- The constant-baseline Brier score uses dev's prevalence deliberately, "
        "not holdout's own realized prevalence, to avoid deriving any new "
        "quantity from holdout even for a contextual baseline.",
        "",
        "## 13. Final frozen-system status",
        "",
        "Model, probability source, feature pipeline, and business threshold all "
        "remain exactly as frozen entering this phase. No value in this report "
        "changes any of them.",
        "",
        "**Phase 8.5 complete: the frozen 0.110 decision policy has been "
        "evaluated once on the untouched holdout; no further threshold "
        "optimization was performed.**",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()

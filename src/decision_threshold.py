"""
Phase 8: Business decision thresholding on the frozen raw XGBoost probability.

Converts a default probability into an approve/reject decision under an
explicit expected-cost model, using dev-only out-of-fold (OOF) predictions
from the already-frozen XGBoost pipeline. holdout.parquet is never read
here -- threshold selection happens on dev only; final holdout evaluation
is a deliberately separate, later step.

ECONOMIC FORMULATION (see module docstring in reports/threshold_analysis.md
for the full derivation): a naive cost formula of `p*LGD - margin` implicitly
assumes the margin is earned unconditionally, regardless of repayment. That
is economically wrong and does not reproduce the agreed theoretical
break-even threshold. The corrected formulation makes margin conditional on
repayment (probability 1-p):

    expected_cost_approve(p) = p * LGD - (1 - p) * margin
    expected_cost_reject     = 0
    approve if p < margin / (margin + LGD)

This is the formula actually implemented below, and it reproduces the
agreed threshold margin/(margin+LGD) ~= 0.11765 exactly.
"""

import json
import logging
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from calibrate_model import OOF_PATH, generate_oof_predictions, save_oof_predictions
from profile_data import PROJECT_ROOT, get_connection
from train_xgboost import DEV_PATH, load_dev_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_JSON_PATH = os.path.join(PROJECT_ROOT, "reports", "threshold_analysis.json")
RESULTS_MD_PATH = os.path.join(PROJECT_ROOT, "reports", "threshold_analysis.md")
COST_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "cost_vs_threshold.png")
SENSITIVITY_CSV_PATH = os.path.join(PROJECT_ROOT, "reports", "sensitivity_analysis.csv")
SENSITIVITY_PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "sensitivity_heatmap.png")

# Agreed illustrative economics (per unit of loan exposure).
LGD = 0.60
MARGIN = 0.08

# Threshold sweep grid.
GRID_MIN = 0.01
GRID_MAX = 0.50
GRID_STEP = 0.001

# Tie-breaking tolerance: thresholds whose mean cost is within
# TIE_TOLERANCE_SE_MULTIPLES standard errors of the minimum-cost threshold's
# own per-applicant cost distribution are considered "essentially identical."
# This is a statistically grounded bar (same logic as the Phase 7.8 bootstrap
# CI test), not an arbitrary percentage -- a fixed relative-percent tolerance
# was tried first and rejected: on this cost distribution's scale it pulled
# in thresholds ~15-20 grid steps from the true optimum as "tied," which they
# were not (see DECISIONS.md).
TIE_TOLERANCE_SE_MULTIPLES = 1.0

# Sensitivity grids.
LGD_GRID = [0.40, 0.50, 0.60, 0.70, 0.80]
MARGIN_GRID = [0.04, 0.06, 0.08, 0.10, 0.12]


# ---------------------------------------------------------------------------
# Economics
# ---------------------------------------------------------------------------

def naive_expected_cost_approve(p, lgd=LGD, margin=MARGIN):
    """The incomplete formula: assumes margin is earned unconditionally.

    Documented for comparison only -- NOT the implemented decision rule.
    Kept so the reconciliation with the corrected formula is inspectable
    in code, not just prose.
    """
    return p * lgd - margin


def naive_threshold(lgd=LGD, margin=MARGIN):
    """Break-even threshold implied by the naive (incomplete) formula."""
    return margin / lgd


def expected_cost_approve(p, lgd=LGD, margin=MARGIN):
    """Corrected expected cost of approving an applicant with default probability p.

    margin is earned only if the loan is repaid (probability 1-p); LGD is
    lost only if the applicant defaults (probability p). cost > 0 means
    expected loss; cost < 0 means expected profit.
    """
    return p * lgd - (1 - p) * margin


def theoretical_threshold(lgd=LGD, margin=MARGIN):
    """Break-even threshold: approve if p < margin / (margin + LGD)."""
    return margin / (margin + lgd)


# ---------------------------------------------------------------------------
# Dev predictions (reuse the frozen pipeline; do not train a new model)
# ---------------------------------------------------------------------------

def load_dev_predictions(oof_path=OOF_PATH, dev_path=DEV_PATH):
    """Load dev-only OOF XGBoost predictions (SK_ID_CURR, TARGET, xgb_probability).

    Reuses the existing reports/oof_predictions.csv artifact if present and
    row-count-consistent with the current dev.parquet; otherwise regenerates
    it via calibrate_model.generate_oof_predictions(), which reuses the
    frozen train_xgboost.MODEL_PARAMS unchanged -- no new model is defined
    here. holdout.parquet is never referenced by this function or anything
    it calls.
    """
    con = get_connection()
    dev_row_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dev_path}')").fetchone()[0]
    con.close()

    if os.path.isfile(oof_path):
        oof_df = pd.read_csv(oof_path)
        if len(oof_df) == dev_row_count:
            logger.info("Reusing existing OOF predictions at %s (%d rows)", oof_path, len(oof_df))
            return oof_df
        logger.info(
            "Existing OOF file has %d rows but dev.parquet has %d -- regenerating",
            len(oof_df), dev_row_count,
        )

    logger.info("Generating fresh OOF predictions from the frozen XGBoost pipeline")
    dev_df = load_dev_data(dev_path)
    oof_df = generate_oof_predictions(dev_df)
    save_oof_predictions(oof_df, oof_path)
    return oof_df


# ---------------------------------------------------------------------------
# Threshold evaluation
# ---------------------------------------------------------------------------

def evaluate_threshold(y, p, threshold, lgd=LGD, margin=MARGIN):
    """Evaluate one threshold against dev-realized outcomes.

    Convention (stated explicitly per project requirement): "positive"
    means default / risky applicant (TARGET == 1). A prediction of
    "positive" (p >= threshold) results in a REJECT decision; a prediction
    of "negative" (p < threshold) results in APPROVE.

        TP = rejected & actually defaulted      (correctly avoided a loss)
        FP = rejected & actually would-have-repaid (a foregone good loan)
        TN = approved & actually repaid           (correctly profitable)
        FN = approved & actually defaulted        (a realized loss)

    Realized expected cost uses the *actual* TARGET label (this is a dev
    backtest, not a re-application of the probability formula): an
    approved applicant who defaulted costs +LGD; an approved applicant who
    repaid earns -margin (a gain); a rejected applicant costs 0 regardless
    of their true outcome, since no loan was issued.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)

    reject_mask = p >= threshold
    approve_mask = ~reject_mask

    tp = int(np.sum(reject_mask & (y == 1)))
    fp = int(np.sum(reject_mask & (y == 0)))
    tn = int(np.sum(approve_mask & (y == 0)))
    fn = int(np.sum(approve_mask & (y == 1)))
    assert tp + fp + tn + fn == n, "confusion matrix counts must sum to n"

    n_approved = int(np.sum(approve_mask))
    n_rejected = int(np.sum(reject_mask))
    approval_rate = n_approved / n
    rejection_rate = n_rejected / n
    assert abs((approval_rate + rejection_rate) - 1.0) < 1e-9, "approval_rate + rejection_rate must equal 1"

    total_defaults = int(np.sum(y == 1))
    default_rate_approved = float(np.mean(y[approve_mask])) if n_approved > 0 else float("nan")
    defaults_captured = tp  # defaults correctly rejected
    defaults_captured_rate = defaults_captured / total_defaults if total_defaults > 0 else float("nan")

    # Realized cost per applicant, using actual outcomes.
    realized_cost = np.where(
        approve_mask,
        np.where(y == 1, lgd, -margin),
        0.0,
    )
    total_cost = float(np.sum(realized_cost))
    mean_cost = float(np.mean(realized_cost))

    return {
        "threshold": float(threshold),
        "n_approved": n_approved,
        "n_rejected": n_rejected,
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "default_rate_approved": default_rate_approved,
        "defaults_captured": defaults_captured,
        "defaults_captured_rate": defaults_captured_rate,
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "total_expected_cost": total_cost,
        "mean_expected_cost": mean_cost,
    }


def sweep_thresholds(y, p, grid_min=GRID_MIN, grid_max=GRID_MAX, grid_step=GRID_STEP, lgd=LGD, margin=MARGIN):
    """Evaluate a grid of thresholds; return a DataFrame, one row per threshold."""
    grid = np.round(np.arange(grid_min, grid_max + grid_step / 2, grid_step), 6)
    assert len(grid) > 0, "threshold grid must be non-empty"
    assert np.all((grid > 0) & (grid < 1)), "threshold grid must lie strictly inside (0, 1)"

    rows = [evaluate_threshold(y, p, t, lgd=lgd, margin=margin) for t in grid]
    return pd.DataFrame(rows)


def select_threshold(y, p, sweep_df, se_multiples=TIE_TOLERANCE_SE_MULTIPLES, lgd=LGD, margin=MARGIN):
    """Pick the cost-minimizing threshold, breaking near-ties toward simplicity.

    Tie-break rule (documented per project requirement): compute the
    per-applicant realized cost distribution *at the minimum-cost
    threshold* and its standard error (std / sqrt(n)) -- the same
    "is this difference distinguishable from noise" logic the project
    already uses for the Phase 7.8 calibration decision. Among all grid
    thresholds whose mean cost is within `se_multiples` standard errors of
    the minimum ("near-optimal"), prefer a round hundredth (e.g. 0.10,
    0.11, 0.12) if one is near-optimal; among multiple near-optimal round
    hundredths, prefer the one closest to the true empirical minimum
    (not simply the smallest value) -- the simpler number wins only when
    it's actually competing with the true optimum, not an arbitrary
    tie-break toward the low end of the range.

    A naive fixed relative-percent tolerance was tried first and produced
    a threshold ~15 grid steps from the true minimum (see DECISIONS.md) --
    rejected because "1% of the mean cost" is not the same thing as
    "statistically indistinguishable from the minimum." The cost curve is
    genuinely flat near its minimum (expected, since it IS a minimum), so
    even a principled SE-based tolerance can still span several round
    hundredths -- handled here by tie-breaking toward proximity to the true
    minimum, not toward the smallest number in the band.
    """
    min_idx = sweep_df["mean_expected_cost"].idxmin()
    min_threshold = sweep_df.loc[min_idx, "threshold"]
    min_cost = sweep_df.loc[min_idx, "mean_expected_cost"]

    approve_mask = p < min_threshold
    per_applicant_cost = np.where(approve_mask, np.where(y == 1, lgd, -margin), 0.0)
    n = len(y)
    se = float(np.std(per_applicant_cost, ddof=1) / np.sqrt(n))
    tolerance = se_multiples * se

    near_optimal = sweep_df[sweep_df["mean_expected_cost"] <= min_cost + tolerance].copy()
    near_optimal["distance_to_round_hundredth"] = (
        near_optimal["threshold"] - np.round(near_optimal["threshold"], 2)
    ).abs()
    near_optimal["distance_to_true_minimum"] = (near_optimal["threshold"] - min_threshold).abs()

    is_round_hundredth = near_optimal["distance_to_round_hundredth"] < 1e-9
    candidates = near_optimal[is_round_hundredth] if is_round_hundredth.any() else near_optimal
    selected = candidates.sort_values(
        ["distance_to_round_hundredth", "distance_to_true_minimum"]
    ).iloc[0]

    return selected.to_dict(), len(near_optimal), tolerance


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(y, p, lgd_grid=LGD_GRID, margin_grid=MARGIN_GRID, grid_min=GRID_MIN, grid_max=GRID_MAX, grid_step=GRID_STEP):
    """For each (LGD, margin) pair, find the theoretical and empirical cost-minimizing threshold."""
    rows = []
    for lgd in lgd_grid:
        for margin in margin_grid:
            sweep_df = sweep_thresholds(y, p, grid_min, grid_max, grid_step, lgd=lgd, margin=margin)
            best_idx = sweep_df["mean_expected_cost"].idxmin()
            best_row = sweep_df.loc[best_idx]
            rows.append({
                "lgd": lgd,
                "margin": margin,
                "theoretical_threshold": theoretical_threshold(lgd, margin),
                "empirical_min_cost_threshold": best_row["threshold"],
                "empirical_min_cost": best_row["mean_expected_cost"],
                "approval_rate_at_empirical_threshold": best_row["approval_rate"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_threshold_logic(y, p, sweep_df, selected):
    """Hard checks required before trusting the threshold decision.

    Note on holdout: this module never imports HOLDOUT_PATH, never calls
    evaluate_holdout_calibration.load_holdout_data, and holdout.parquet
    does not appear anywhere in this file -- verified by inspection, not
    just asserted here, since a runtime check cannot prove a file was
    never opened without inspecting the code that would open it.
    """
    n = len(y)

    assert len(p) == n, f"prediction row count ({len(p)}) must equal dev row count ({n})"
    assert np.all((p >= 0) & (p <= 1)), "all probabilities must be in [0, 1]"

    assert len(sweep_df) > 0, "threshold grid must be non-empty"
    assert sweep_df["threshold"].is_monotonic_increasing, "threshold grid must be sorted"

    row_sums = sweep_df["true_positives"] + sweep_df["false_positives"] + sweep_df["true_negatives"] + sweep_df["false_negatives"]
    assert (row_sums == n).all(), "confusion-matrix counts must sum to dev row count at every threshold"

    rate_sums = sweep_df["approval_rate"] + sweep_df["rejection_rate"]
    assert np.allclose(rate_sums, 1.0), "approval_rate + rejection_rate must equal 1 at every threshold"

    assert selected["threshold"] in set(sweep_df["threshold"]), "selected threshold must exist in the evaluated grid"

    # Independent spot-check: recompute expected cost for the selected threshold from raw counts.
    row = sweep_df[sweep_df["threshold"] == selected["threshold"]].iloc[0]
    spot_check_cost = (row["false_negatives"] * LGD - row["true_negatives"] * MARGIN) / n
    assert abs(spot_check_cost - row["mean_expected_cost"]) < 1e-9, (
        f"spot-check cost {spot_check_cost} does not match vectorized cost {row['mean_expected_cost']}"
    )

    # Theoretical threshold must be mathematically consistent with the implemented cost formula:
    # expected_cost_approve(theoretical_threshold) must equal expected_cost_reject (0), analytically.
    t = theoretical_threshold(LGD, MARGIN)
    analytic_cost_at_threshold = expected_cost_approve(t, LGD, MARGIN)
    assert abs(analytic_cost_at_threshold) < 1e-9, (
        f"theoretical threshold does not zero the implemented cost formula: "
        f"expected_cost_approve({t}) = {analytic_cost_at_threshold}, expected 0"
    )

    logger.info("All validation checks passed.")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def plot_cost_vs_threshold(sweep_df, selected, theoretical_t, output_path=COST_PLOT_PATH):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(sweep_df["threshold"], sweep_df["mean_expected_cost"], label="Empirical mean expected cost (dev)")
    ax.axvline(theoretical_t, color="gray", linestyle="--", label=f"Theoretical break-even ({theoretical_t:.5f})")
    ax.axvline(selected["threshold"], color="red", linestyle=":", label=f"Selected threshold ({selected['threshold']:.3f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Threshold (reject if p_default >= threshold)")
    ax.set_ylabel("Mean expected cost per applicant (fraction of exposure)")
    ax.set_title("Expected Cost vs. Decision Threshold (dev, LGD=%.2f, margin=%.2f)" % (LGD, MARGIN))
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved cost-vs-threshold plot to %s", output_path)


def plot_sensitivity_heatmap(sensitivity_df, output_path=SENSITIVITY_PLOT_PATH):
    pivot = sensitivity_df.pivot(index="lgd", columns="margin", values="empirical_min_cost_threshold")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{m:.2f}" for m in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{l:.2f}" for l in pivot.index])
    ax.set_xlabel("Margin")
    ax.set_ylabel("LGD")
    ax.set_title("Empirical Cost-Minimizing Threshold by (LGD, Margin)")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Threshold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sensitivity heatmap to %s", output_path)


def save_reports(y, p, sweep_df, selected, sensitivity_df, n_tied, tie_tolerance, theoretical_t, dev_prevalence):
    n = len(y)
    reject_everyone_cost = 0.0
    approve_everyone_cost = float(dev_prevalence * LGD - (1 - dev_prevalence) * MARGIN)

    # Cost at the theoretical break-even threshold (evaluated directly, not snapped to grid).
    theoretical_eval = evaluate_threshold(y, p, theoretical_t)

    results = {
        "economics": {
            "lgd": LGD,
            "margin": MARGIN,
            "naive_formula": "p*LGD - margin (incomplete: assumes margin earned unconditionally)",
            "naive_threshold": naive_threshold(),
            "corrected_formula": "p*LGD - (1-p)*margin (margin conditional on repayment)",
            "theoretical_threshold": theoretical_t,
        },
        "baselines": {
            "dev_default_prevalence": dev_prevalence,
            "reject_everyone_mean_cost": reject_everyone_cost,
            "approve_everyone_mean_cost": approve_everyone_cost,
        },
        "theoretical_threshold_evaluation": theoretical_eval,
        "selected_threshold": {
            **selected,
            "tie_candidates_count": n_tied,
            "tie_tolerance_used": tie_tolerance,
        },
        "grid": {"min": GRID_MIN, "max": GRID_MAX, "step": GRID_STEP, "n_points": len(sweep_df)},
        "n_dev_rows": n,
    }

    os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Saved %s", RESULTS_JSON_PATH)

    sensitivity_df.to_csv(SENSITIVITY_CSV_PATH, index=False)
    logger.info("Saved %s", SENSITIVITY_CSV_PATH)

    return results, theoretical_eval


def write_markdown_report(y, sweep_df, selected, sensitivity_df, n_tied, tie_tolerance, theoretical_t, theoretical_eval, dev_prevalence, output_path=RESULTS_MD_PATH):
    n = len(y)
    approve_everyone_cost = dev_prevalence * LGD - (1 - dev_prevalence) * MARGIN
    naive_t = naive_threshold()

    lines = [
        "# Threshold Analysis Report (Phase 8)",
        "",
        "## 1. Objective",
        "",
        "Convert the frozen model's default probability into an approve/reject "
        "decision under an explicit, documented expected-cost model. This is a "
        "business-decisioning exercise, not a modeling exercise: no ROC-AUC/PR-AUC "
        "optimization happens in this phase.",
        "",
        "## 2. Frozen model / probability source",
        "",
        "**Raw, uncalibrated XGBoost** (Phase 7.8 decision, unchanged here). "
        "Predictions used for threshold selection are dev-only out-of-fold (OOF) "
        f"scores from the frozen `train_xgboost.MODEL_PARAMS` configuration "
        f"({n} dev rows). `holdout.parquet` is not read anywhere in this phase.",
        "",
        "## 3. Economic assumptions",
        "",
        f"- LGD (loss given default) = {LGD}",
        f"- Margin (net profit if repaid) = {MARGIN}",
        "- Both expressed as a fraction of loan exposure; the same unit throughout.",
        "",
        "## 4. Exact expected-cost formulation",
        "",
        "**A naive formula was considered and rejected.** The initial brief "
        f"suggested `expected_cost_approve = p*LGD - margin`, giving a break-even "
        f"threshold of `margin/LGD = {naive_t:.5f}`. This does **not** match the "
        f"agreed theoretical threshold ({theoretical_threshold():.5f}), and the "
        f"discrepancy is not a rounding issue -- it reflects a real economic "
        f"assumption error: the naive formula treats `margin` as earned "
        f"*unconditionally*, regardless of whether the loan is repaid.",
        "",
        "**Corrected formulation.** Margin is only earned if the loan is repaid "
        "(probability `1-p`); LGD is only lost if the applicant defaults "
        "(probability `p`):",
        "",
        "```",
        "expected_cost_approve(p) = p * LGD - (1 - p) * margin",
        "expected_cost_reject     = 0",
        "",
        "approve if expected_cost_approve(p) < 0",
        "        <=> p * LGD < (1 - p) * margin",
        "        <=> p * (LGD + margin) < margin",
        "        <=> p < margin / (margin + LGD)",
        "```",
        "",
        f"This reproduces the agreed theoretical threshold exactly: "
        f"`{MARGIN}/({MARGIN}+{LGD}) = {theoretical_threshold():.5f}`. "
        "**This corrected formula is what `evaluate_threshold()` implements.** "
        "The naive formula is retained in code "
        "(`naive_expected_cost_approve`, `naive_threshold`) for documentation "
        "and comparison only -- it is never used for the actual decision rule.",
        "",
        "## 5. Threshold derivation",
        "",
        f"Theoretical break-even threshold: `margin/(margin+LGD)` = "
        f"`{MARGIN}/{round(MARGIN+LGD, 10)}` = **{theoretical_threshold():.5f}**.",
        "",
        "## 6. Threshold sweep methodology",
        "",
        f"Grid: `[{GRID_MIN}, {GRID_MAX}]` step `{GRID_STEP}` "
        f"({len(sweep_df)} thresholds). For each threshold, applicants with "
        "`p_default >= threshold` are rejected; all others are approved. "
        "**\"Positive\" is defined as default/risky applicant (TARGET == 1)** "
        "throughout -- a REJECT decision corresponds to a positive prediction. "
        "Expected cost at each threshold is computed from *realized* dev "
        "outcomes (actual `TARGET` labels), not from re-applying the "
        "probability formula -- this is a historical backtest on dev, using "
        "dev's known labels, never holdout's.",
        "",
        "## 7. Dev results",
        "",
        f"- Dev default prevalence: **{dev_prevalence*100:.4f}%**",
        f"- \"Approve everyone\" mean expected cost: **{approve_everyone_cost:.6f}** "
        f"({'a net profit' if approve_everyone_cost < 0 else 'a net loss'} per "
        "applicant, since the population default rate is below the break-even "
        "threshold)",
        "- \"Reject everyone\" mean expected cost: **0.000000** (by definition)",
        f"- Cost at theoretical break-even threshold ({theoretical_t:.5f}): "
        f"**{theoretical_eval['mean_expected_cost']:.6f}**, approval rate "
        f"{theoretical_eval['approval_rate']*100:.2f}%",
        f"- **Selected (minimum-cost, tie-broken) threshold: {selected['threshold']:.3f}**, "
        f"mean expected cost **{selected['mean_expected_cost']:.6f}**, "
        f"approval rate **{selected['approval_rate']*100:.2f}%**, "
        f"rejection rate **{selected['rejection_rate']*100:.2f}%**",
        f"- At the selected threshold: {selected['defaults_captured']} of "
        f"{int(sum(y))} dev defaults correctly rejected "
        f"({selected['defaults_captured_rate']*100:.2f}% capture rate); "
        f"default rate among approved applicants: "
        f"{selected['default_rate_approved']*100:.4f}% (vs. {dev_prevalence*100:.4f}% "
        "unconditional)",
        f"- Confusion matrix at selected threshold: TP={selected['true_positives']}, "
        f"FP={selected['false_positives']}, TN={selected['true_negatives']}, "
        f"FN={selected['false_negatives']}",
        "",
        "**Tie-breaking rule applied:** among all grid thresholds within "
        f"{TIE_TOLERANCE_SE_MULTIPLES:.1f} standard error(s) of the minimum "
        f"mean cost ({n_tied} candidate thresholds, tolerance {tie_tolerance:.2e} "
        "-- a statistically grounded bar, not an arbitrary percentage; see "
        "DECISIONS.md), the one closest to a round hundredth "
        "(e.g. 0.10, 0.11, 0.12) was selected, for interpretability, since the "
        "cost difference among them is negligible.",
        "",
        "See `reports/cost_vs_threshold.png` for the full cost curve.",
        "",
        "## 8. Theoretical-vs-empirical threshold comparison",
        "",
        f"Theoretical: **{theoretical_t:.5f}**. Empirical (dev-minimum-cost, "
        f"pre-tie-break): **{sweep_df.loc[sweep_df['mean_expected_cost'].idxmin(), 'threshold']:.3f}**. "
        "These are expected to be close but not necessarily identical: the "
        "theoretical threshold assumes the model's probability is exactly "
        "calibrated at that point, while the empirical minimum reflects "
        "whatever calibration the raw (uncalibrated) XGBoost score actually "
        "has there. Rough agreement is a consistency check on the model and "
        "the formula, not a claim that raw XGBoost is perfectly calibrated "
        "(it is close, per Phase 7's reliability diagram, but not exact).",
        "",
        "## 9. Sensitivity analysis",
        "",
        f"LGD grid: {LGD_GRID}. Margin grid: {MARGIN_GRID}. For each "
        "combination, both the theoretical and empirical (dev) cost-minimizing "
        "threshold are computed. Full table: `reports/sensitivity_analysis.csv`; "
        "heatmap: `reports/sensitivity_heatmap.png`.",
        "",
        "| LGD | Margin | Theoretical threshold | Empirical threshold | Approval rate |",
        "|---|---|---|---|---|",
    ]
    for _, row in sensitivity_df.iterrows():
        lines.append(
            f"| {row['lgd']:.2f} | {row['margin']:.2f} | "
            f"{row['theoretical_threshold']:.5f} | "
            f"{row['empirical_min_cost_threshold']:.3f} | "
            f"{row['approval_rate_at_empirical_threshold']*100:.1f}% |"
        )

    lines += [
        "",
        "As LGD rises or margin falls, the break-even threshold drops (the "
        "lender must be more conservative — reject more applicants — since "
        "each default is costlier relative to the reward from a good loan). "
        "As margin rises or LGD falls, the threshold rises (more applicants "
        "are worth the risk).",
        "",
        "## 10. Business interpretation",
        "",
        f"At the agreed economics (LGD={LGD}, margin={MARGIN}), the "
        f"mathematically cost-minimizing policy approves "
        f"{selected['approval_rate']*100:.1f}% of dev applicants and rejects "
        f"{selected['rejection_rate']*100:.1f}%. This is a *substantial* "
        "rejection volume relative to the raw default rate "
        f"({dev_prevalence*100:.2f}%) -- the cost-minimizing lender rejects far "
        "more applicants than actually default, because the cost of a missed "
        "default (LGD) is large relative to the reward from a good loan "
        "(margin), so the model errs toward caution near the threshold. "
        "**This is a real trade-off, not a modeling artifact**: a purely "
        "cost-minimizing threshold can imply a much lower approval volume "
        "than a real lender focused on growth or market share would accept. "
        "In practice, lenders operate under approval-volume or growth "
        "constraints that a pure expected-cost minimization does not "
        "represent. The expected-cost threshold derived here is a **business "
        "policy choice given the stated economics**, not a universally "
        "\"correct\" number independent of business context — a different "
        "risk appetite, funding cost, or growth target would justify a "
        "different threshold along the sensitivity curve in Section 9.",
        "",
        "## 11. Limitations",
        "",
        "- LGD and margin are treated as constants across all applicants; in "
        "reality both likely vary by loan size, term, and applicant segment.",
        "- The expected-cost model ignores time value of money, funding cost, "
        "operational cost of processing an application, and any regulatory "
        "constraints on rejection rates or fair-lending requirements.",
        "- The empirical sweep uses dev's realized outcomes as a backtest; "
        "it is not a forward-looking guarantee, and dev itself is one "
        "realization of the applicant population (same caveat as noted for "
        "holdout in `reports/final_model_card.md`).",
        "- Raw XGBoost is close to, but not exactly, calibrated (Phase 7 "
        "finding) — the threshold sweep uses realized outcomes specifically "
        "so that residual miscalibration does not bias the *selected* "
        "threshold, but the theoretical break-even value in isolation does "
        "assume calibration at that point.",
        "",
        "## 12. Decision locked for final holdout evaluation",
        "",
        f"**Selected threshold: {selected['threshold']:.3f}** (reject if "
        f"`p_default >= {selected['threshold']:.3f}`), locked from dev-only "
        "analysis above. Per Phase 8 methodology rules, holdout.parquet was "
        "not read at any point in this phase.",
        "",
        "**Phase 8 threshold is now frozen; final holdout evaluation is the "
        "next step.**",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved %s", output_path)


def main():
    logger.info("Loading dev-only OOF predictions (frozen XGBoost pipeline, no retraining)")
    oof_df = load_dev_predictions()

    y = oof_df["TARGET"].astype(int).values
    p = oof_df["xgb_probability"].astype(float).values
    dev_prevalence = float(np.mean(y))

    logger.info("Sweeping thresholds on dev (grid %.3f-%.3f step %.3f)", GRID_MIN, GRID_MAX, GRID_STEP)
    sweep_df = sweep_thresholds(y, p)

    theoretical_t = theoretical_threshold()
    selected, n_tied, tie_tolerance = select_threshold(y, p, sweep_df)
    logger.info("Selected threshold: %.3f (mean cost %.6f)", selected["threshold"], selected["mean_expected_cost"])

    validate_threshold_logic(y, p, sweep_df, selected)

    logger.info("Running sensitivity analysis over LGD x margin grid")
    sensitivity_df = run_sensitivity_analysis(y, p)

    results, theoretical_eval = save_reports(
        y, p, sweep_df, selected, sensitivity_df, n_tied, tie_tolerance, theoretical_t, dev_prevalence,
    )
    write_markdown_report(
        y, sweep_df, selected, sensitivity_df, n_tied, tie_tolerance, theoretical_t, theoretical_eval, dev_prevalence,
    )
    plot_cost_vs_threshold(sweep_df, selected, theoretical_t)
    plot_sensitivity_heatmap(sensitivity_df)

    print(json.dumps(results, indent=2, default=float))
    print("\nPhase 8 threshold is now frozen; final holdout evaluation is the next step.")


if __name__ == "__main__":
    main()

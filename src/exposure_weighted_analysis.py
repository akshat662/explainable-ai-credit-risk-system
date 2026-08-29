"""
Pre-Phase-9 hardening: exposure-weighted secondary economic analysis.

Phase 8's threshold sweep (src/decision_threshold.py, untouched here)
assumes unit exposure -- every applicant contributes equally to expected
cost, regardless of loan size. This script asks a narrower, secondary
question: does weighting each applicant's outcome by their actual loan
size (AMT_CREDIT) change the picture materially?

This is explicitly a SECONDARY / SENSITIVITY analysis:
- It does not change, override, or regenerate the frozen 0.110 threshold
  or reports/threshold_analysis.json.
- It reuses LGD, MARGIN, and the threshold grid unchanged from
  decision_threshold.py -- no new economics, no re-tuning.
- It uses dev-only OOF predictions, exactly like Phase 8. holdout.parquet
  is never read.
- No new model is trained; AMT_CREDIT is read directly from dev.parquet
  and joined onto the existing OOF predictions by SK_ID_CURR.
"""

import json
import logging
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from decision_threshold import (
    GRID_MAX,
    GRID_MIN,
    GRID_STEP,
    LGD,
    MARGIN,
    load_dev_predictions,
)
from profile_data import PROJECT_ROOT, get_connection
from train_xgboost import DEV_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "exposure_weighted_analysis.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "reports", "exposure_weighted_analysis.md")
PLOT_PATH = os.path.join(PROJECT_ROOT, "reports", "exposure_weighted_cost_vs_threshold.png")

FROZEN_THRESHOLD = 0.110  # Phase 8's production threshold. Reported against, never changed.


def load_dev_exposure(dev_path=DEV_PATH):
    """Load SK_ID_CURR -> AMT_CREDIT from dev.parquet only (never holdout)."""
    con = get_connection()
    df = con.execute(f"SELECT SK_ID_CURR, AMT_CREDIT FROM read_parquet('{dev_path}')").fetchdf()
    con.close()
    assert df["AMT_CREDIT"].notna().all(), "AMT_CREDIT must not contain nulls for exposure weighting"
    assert (df["AMT_CREDIT"] > 0).all(), "AMT_CREDIT must be strictly positive to serve as an exposure weight"
    return df


def exposure_weighted_cost_at_threshold(y, p, exposure, threshold, lgd=LGD, margin=MARGIN):
    """Exposure-weighted realized cost at one threshold, using dev's actual labels.

    Same decision rule and cost formula as decision_threshold.evaluate_threshold
    (reject if p >= threshold; cost = LGD if approved-and-defaulted, -margin if
    approved-and-repaid, 0 if rejected) -- the only difference is each
    applicant's cost is scaled by their AMT_CREDIT rather than treated as a
    unit (1.0) exposure.
    """
    approve_mask = p < threshold
    unit_cost = np.where(approve_mask, np.where(y == 1, lgd, -margin), 0.0)
    dollar_cost = unit_cost * exposure

    total_exposure = float(np.sum(exposure))
    total_dollar_cost = float(np.sum(dollar_cost))

    return {
        "threshold": float(threshold),
        "total_dollar_cost": total_dollar_cost,
        "mean_dollar_cost_per_applicant": float(np.mean(dollar_cost)),
        "cost_per_dollar_of_total_exposure": total_dollar_cost / total_exposure,
        "approval_rate": float(np.mean(approve_mask)),
    }


def sweep_exposure_weighted(y, p, exposure, grid_min=GRID_MIN, grid_max=GRID_MAX, grid_step=GRID_STEP):
    grid = np.round(np.arange(grid_min, grid_max + grid_step / 2, grid_step), 6)
    rows = [exposure_weighted_cost_at_threshold(y, p, exposure, t) for t in grid]
    return rows


def main():
    logger.info("Loading dev-only OOF predictions (reused unchanged from Phase 8)")
    oof_df = load_dev_predictions()  # dev-only; holdout.parquet never referenced by this function

    logger.info("Loading AMT_CREDIT from dev.parquet for exposure weighting")
    exposure_df = load_dev_exposure()

    merged = oof_df.merge(exposure_df, on="SK_ID_CURR", how="inner", validate="one_to_one")
    assert len(merged) == len(oof_df), (
        f"exposure join dropped rows: {len(merged)} vs {len(oof_df)} OOF predictions -- "
        f"every dev applicant must have an AMT_CREDIT value"
    )

    y = merged["TARGET"].astype(int).values
    p = merged["xgb_probability"].astype(float).values
    exposure = merged["AMT_CREDIT"].astype(float).values
    n = len(merged)

    logger.info("Sweeping exposure-weighted cost over the same grid as Phase 8 (%d applicants)", n)
    sweep = sweep_exposure_weighted(y, p, exposure)

    costs = np.array([r["cost_per_dollar_of_total_exposure"] for r in sweep])
    best_idx = int(np.argmin(costs))
    exposure_weighted_optimal = sweep[best_idx]

    frozen_eval = exposure_weighted_cost_at_threshold(y, p, exposure, FROZEN_THRESHOLD)

    # Unit-exposure reference point for comparison: same threshold, weight = 1 for everyone.
    unit_eval = exposure_weighted_cost_at_threshold(y, p, np.ones(n), FROZEN_THRESHOLD)

    total_exposure = float(np.sum(exposure))
    logger.info(
        "Exposure-weighted optimum: threshold=%.3f, cost/$=%.6f (vs frozen 0.110: cost/$=%.6f)",
        exposure_weighted_optimal["threshold"], exposure_weighted_optimal["cost_per_dollar_of_total_exposure"],
        frozen_eval["cost_per_dollar_of_total_exposure"],
    )

    results = {
        "status": "SECONDARY / SENSITIVITY ANALYSIS -- does not change the frozen 0.110 threshold",
        "n_dev_applicants": n,
        "total_dev_exposure_amt_credit": total_exposure,
        "mean_exposure_per_applicant": total_exposure / n,
        "frozen_threshold": FROZEN_THRESHOLD,
        "frozen_threshold_exposure_weighted_evaluation": frozen_eval,
        "frozen_threshold_unit_weighted_evaluation_for_reference": unit_eval,
        "exposure_weighted_optimal_threshold_evaluation": exposure_weighted_optimal,
        "threshold_difference_optimal_minus_frozen": exposure_weighted_optimal["threshold"] - FROZEN_THRESHOLD,
        "grid": {"min": GRID_MIN, "max": GRID_MAX, "step": GRID_STEP, "n_points": len(sweep)},
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info("Saved %s", RESULTS_PATH)

    plot_exposure_weighted_curve(sweep, exposure_weighted_optimal, frozen_eval)
    write_markdown_report(results)

    print(json.dumps(results, indent=2, default=float))
    print(
        "\nExposure-weighted secondary analysis complete. The frozen production "
        f"threshold remains 0.110 (unchanged); this analysis is reported for "
        f"context only."
    )


def plot_exposure_weighted_curve(sweep, exposure_weighted_optimal, frozen_eval, output_path=PLOT_PATH):
    thresholds = [r["threshold"] for r in sweep]
    costs = [r["cost_per_dollar_of_total_exposure"] for r in sweep]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, costs, label="Exposure-weighted cost per $ of exposure (dev)")
    ax.axvline(FROZEN_THRESHOLD, color="red", linestyle=":",
               label=f"Frozen production threshold ({FROZEN_THRESHOLD:.3f})")
    ax.axvline(exposure_weighted_optimal["threshold"], color="purple", linestyle="--",
               label=f"Exposure-weighted optimum ({exposure_weighted_optimal['threshold']:.3f})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Threshold (reject if p_default >= threshold)")
    ax.set_ylabel("Cost per $ of total dev exposure")
    ax.set_title("Exposure-Weighted Secondary Analysis (dev only) -- does not change production policy")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", output_path)


def write_markdown_report(results, output_path=REPORT_PATH):
    frozen = results["frozen_threshold_exposure_weighted_evaluation"]
    unit_ref = results["frozen_threshold_unit_weighted_evaluation_for_reference"]
    optimal = results["exposure_weighted_optimal_threshold_evaluation"]
    diff = results["threshold_difference_optimal_minus_frozen"]

    lines = [
        "# Exposure-Weighted Secondary Analysis",
        "",
        "**Status: secondary / sensitivity analysis. Does not change the frozen "
        "production threshold (0.110) or `reports/threshold_analysis.json`.**",
        "",
        "## Why this exists",
        "",
        "Phase 8's primary threshold analysis assumes unit exposure -- every "
        "applicant's expected cost is weighted equally (LGD x 1, margin x 1), "
        "regardless of how large their loan actually is. This is a simplification: "
        "a defaulted $2M loan and a defaulted $50K loan both cost `1 x LGD` in the "
        "primary analysis. This script asks whether weighting each applicant's "
        "outcome by their actual loan size (`AMT_CREDIT`) changes the "
        "cost-minimizing operating point materially.",
        "",
        "## Method",
        "",
        f"- **Data**: dev-only OOF predictions (same `reports/oof_predictions.csv` "
        f"Phase 8 uses, reused unchanged), joined to `AMT_CREDIT` from "
        f"`dev.parquet` by `SK_ID_CURR` ({results['n_dev_applicants']:,} applicants, "
        "no join loss, no nulls, all positive). `holdout.parquet` is not read.",
        f"- **Economics**: identical LGD ({LGD}) and margin ({MARGIN}) constants "
        "imported unchanged from `decision_threshold.py`. Same threshold grid "
        f"(`[{GRID_MIN}, {GRID_MAX}]` step `{GRID_STEP}`).",
        "- **Weighting**: each applicant's realized cost (same formula as Phase "
        "8: `LGD` if approved-and-defaulted, `-margin` if approved-and-repaid, "
        "`0` if rejected) is multiplied by their `AMT_CREDIT`. The reported "
        "\"cost per $ of total exposure\" divides total dollar cost by total dev "
        "exposure -- the same *unit* (a fraction) as Phase 8's unit-exposure "
        "metric, so the two are directly comparable in scale.",
        f"- Total dev exposure: **${results['total_dev_exposure_amt_credit']:,.0f}**, "
        f"mean **${results['mean_exposure_per_applicant']:,.0f}** per applicant.",
        "",
        "## Results",
        "",
        "| | Threshold | Cost per $ of exposure | Approval rate |",
        "|---|---|---|---|",
        f"| Frozen threshold, unit-weighted (Phase 8 reference) | {FROZEN_THRESHOLD:.3f} | "
        f"{unit_ref['cost_per_dollar_of_total_exposure']:.6f} | {unit_ref['approval_rate']*100:.2f}% |",
        f"| Frozen threshold, exposure-weighted | {FROZEN_THRESHOLD:.3f} | "
        f"{frozen['cost_per_dollar_of_total_exposure']:.6f} | {frozen['approval_rate']*100:.2f}% |",
        f"| Exposure-weighted optimum | {optimal['threshold']:.3f} | "
        f"{optimal['cost_per_dollar_of_total_exposure']:.6f} | {optimal['approval_rate']*100:.2f}% |",
        "",
        f"The exposure-weighted cost-minimizing threshold ({optimal['threshold']:.3f}) "
        f"differs from the frozen production threshold (0.110) by "
        f"{diff:+.3f}. "
        + (
            "This is a small difference, consistent with the unit-exposure result "
            "remaining a reasonable operating point even under exposure weighting."
            if abs(diff) <= 0.01 else
            "This is a non-trivial difference -- exposure weighting does shift where "
            "the cost-minimizing point falls, worth noting explicitly rather than "
            "glossing over, though it does not by itself justify changing the "
            "already-frozen production policy without a separate, deliberate decision."
        ),
        "",
        "See `reports/exposure_weighted_cost_vs_threshold.png`.",
        "",
        "## Interpretation",
        "",
        "This analysis is informational context on top of Phase 8's decision, not "
        "a replacement for it. The unit-exposure result remains the primary Phase "
        "8 outcome and the frozen production threshold (0.110) is unchanged by "
        "this script. A production system that genuinely wants exposure-weighted "
        "decisioning would need this analysis to go through the same dev-only "
        "selection, sensitivity, and (eventually) holdout-validation discipline "
        "Phase 8 applied to the unit-exposure threshold -- not simply substitute "
        "this number in.",
        "",
        "## Limitations",
        "",
        "- `AMT_CREDIT` is the requested loan amount at application time, not "
        "necessarily the final disbursed or outstanding exposure over the loan's "
        "life -- treated here as a reasonable proxy, not an exact economic "
        "exposure figure.",
        "- LGD and margin are still applied as flat rates regardless of loan "
        "size; in reality both plausibly vary with loan size too (not modeled "
        "here, to avoid expanding scope beyond a single, clean weighting "
        "change).",
        "- This is a dev-only analysis; it has not been validated on holdout, "
        "consistent with Phase 8's methodology rules that holdout is not used "
        "for threshold selection or comparison of candidate thresholds.",
        "",
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()

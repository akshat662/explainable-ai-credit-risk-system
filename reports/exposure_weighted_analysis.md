# Exposure-Weighted Secondary Analysis

**Status: secondary / sensitivity analysis. Does not change the frozen production threshold (0.110) or `reports/threshold_analysis.json`.**

## Why this exists

Phase 8's primary threshold analysis assumes unit exposure -- every applicant's expected cost is weighted equally (LGD x 1, margin x 1), regardless of how large their loan actually is. This is a simplification: a defaulted $2M loan and a defaulted $50K loan both cost `1 x LGD` in the primary analysis. This script asks whether weighting each applicant's outcome by their actual loan size (`AMT_CREDIT`) changes the cost-minimizing operating point materially.

## Method

- **Data**: dev-only OOF predictions (same `reports/oof_predictions.csv` Phase 8 uses, reused unchanged), joined to `AMT_CREDIT` from `dev.parquet` by `SK_ID_CURR` (246,008 applicants, no join loss, no nulls, all positive). `holdout.parquet` is not read.
- **Economics**: identical LGD (0.6) and margin (0.08) constants imported unchanged from `decision_threshold.py`. Same threshold grid (`[0.01, 0.5]` step `0.001`).
- **Weighting**: each applicant's realized cost (same formula as Phase 8: `LGD` if approved-and-defaulted, `-margin` if approved-and-repaid, `0` if rejected) is multiplied by their `AMT_CREDIT`. The reported "cost per $ of total exposure" divides total dollar cost by total dev exposure -- the same *unit* (a fraction) as Phase 8's unit-exposure metric, so the two are directly comparable in scale.
- Total dev exposure: **$147,442,004,050**, mean **$599,338** per applicant.

## Results

| | Threshold | Cost per $ of exposure | Approval rate |
|---|---|---|---|
| Frozen threshold, unit-weighted (Phase 8 reference) | 0.110 | -0.038287 | 78.86% |
| Frozen threshold, exposure-weighted | 0.110 | -0.040028 | 78.86% |
| Exposure-weighted optimum | 0.114 | -0.040104 | 79.83% |

The exposure-weighted cost-minimizing threshold (0.114) differs from the frozen production threshold (0.110) by +0.004. This is a small difference, consistent with the unit-exposure result remaining a reasonable operating point even under exposure weighting.

See `reports/exposure_weighted_cost_vs_threshold.png`.

## Interpretation

This analysis is informational context on top of Phase 8's decision, not a replacement for it. The unit-exposure result remains the primary Phase 8 outcome and the frozen production threshold (0.110) is unchanged by this script. A production system that genuinely wants exposure-weighted decisioning would need this analysis to go through the same dev-only selection, sensitivity, and (eventually) holdout-validation discipline Phase 8 applied to the unit-exposure threshold -- not simply substitute this number in.

## Limitations

- `AMT_CREDIT` is the requested loan amount at application time, not necessarily the final disbursed or outstanding exposure over the loan's life -- treated here as a reasonable proxy, not an exact economic exposure figure.
- LGD and margin are still applied as flat rates regardless of loan size; in reality both plausibly vary with loan size too (not modeled here, to avoid expanding scope beyond a single, clean weighting change).
- This is a dev-only analysis; it has not been validated on holdout, consistent with Phase 8's methodology rules that holdout is not used for threshold selection or comparison of candidate thresholds.


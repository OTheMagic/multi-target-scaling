# Signed Gains and Comparator Rerun Audit

## What Is Being Compared?

The old shortcut's proof uses nonnegative scores. Feeding raw signed scores
into it is not a justified baseline. The valid comparisons here are the new
signed envelope versus the old shortcut applied to capped scores, and versus
the capped envelope, shifted shortcut, signed GWC, and CQHR separately.
There is no uniform signed-versus-capped containment theorem.

## Are Signed Gains Large?

Usually not in these experiments. For the main three-output Gaussian CQR study
(five features, 2,400 training observations, 600 test observations, 100 fresh
trials per setting), both nominal base and target miscoverage are 0.1:

| Calibration size | Reduction vs old capped shortcut | Sign-only reduction vs capped envelope | Reduction vs signed GWC | Reduction vs CQHR |
| --- | --- | --- | --- | --- |
| 30 | 4.23% | 3.34% | 4.63% | -3.82% |
| 50 | 1.63% | 1.25% | 2.77% | -1.89% |
| 100 | 0.85% | 0.69% | 1.31% | 0.57% |
| 200 | 0.11% | 0.04% | 0.62% | 0.68% |

A positive percentage means a smaller signed-envelope region. These are
100 times one minus the mean within-trial ratio of mean outcome volumes, not
ratios of pooled means. The CSV files retain paired Monte Carlo standard errors,
finite-pair counts, and both coverage estimates. Signed coverage in these four
settings ranges from 0.9003 to 0.9092.

An intuition check: in an idealized independent-coordinate model, accurately
fitted marginal 90% intervals have joint coverage 0.9^d, only 0.729 for d=3.
Reaching a joint 90% target generally calls for expansion. This does not forbid
shrinking an individual coordinate, but it limits how much broad shrinkage one
should expect. Conservative marginal 98% intervals in two independent
coordinates instead have joint coverage 0.9604, leaving room to shrink toward
the 90% target. This calculation is intuition, not an assumption used in the
method's proof or a claim about the fitted models' exact base coverage.

The seven freshly fitted Gaussian toys are mixed: versus the old capped shortcut,
the mean paired reductions range from -1.90% to
2.02%. Negative reductions are counterexamples
to uniform efficiency dominance across score representations.

Larger practical gains occur when the base intervals are very conservative.
In the conservative toy (500 trials, base miscoverage 0.02, target 0.1), the signed
method has coverage 0.9043. The old capped shortcut
has infinite volume in 258/500 trials
because capping removes all calibration variation in at least one coordinate.
Among its 242 finite-reference trials, the signed envelope
reduces volume by 9.28% on average. This is a
conditional finite-pair statistic, not an unconditional finite volume ratio.
Signed shrinkage versus the uncalibrated base interval is a separate comparison.

Against CQHR, the misspecified-width toy at common base miscoverage 0.1 shows
9.17% mean paired reduction, with coverage
0.9034 versus 0.9064.
Its ratio of mean volumes is 0.826, a different
estimator. Standard Gaussian toys do not show this CQHR advantage.

## Every Archived Comparator, Not Just Envelope

The audit classifies 609 archived CSV files and checks
3,402 method/scenario entries, including
mixed aggregates and rows present only in baseline-specific files. These entries
include duplicate historical views; they are not independent experiment counts.
Status: 3,401 covered by fresh reruns or fitted
replications, 0 unfilled affordable entries,
and 1 explicitly deferred signed full-LWC illustration.

The non-envelope families checked are TSCP_R, TSCP_GWC, TSCP_S, Unscaled,
Empirical Copula, Point CHR, Naive, Bonferroni, Population Oracle, and CQHR.
Existing completed small full-LWC results remain available; no new full-LWC
work was launched for this audit.

The audit found and repaired genuine omissions:

- Raw CQR Unscaled and Empirical Copula: evaluated for all 2,230 already-fresh
  fitted CQR trials, using the same fitted predictions as the other methods.
- Raw and capped comparators: completed for 930 already-fresh fitted toy trials.
- A baseline-only 50-dimensional Laplace setting at n_cal=30: 30 new trials,
  each with fresh training/calibration/test data and refitting. Ordinary
  baselines, including the population oracle, are complete; no full-cell search.
- An archived one-trial smoke check: replaced with three explicitly specified
  fresh trials. Its old table retained d=2, n_cal=12, alpha=0.1 and correlation
  0.4, but not training size or feature count. The replacement uses 80 training,
  20 test observations and two features. It is a new smoke replication, not an
  exact replay of an undocumented fit setup.

Adding a comparator to the fitted outputs of an existing fresh trial preserves
pairing and the required protocol: datasets and fits differ across trials,
while methods within a trial share them. It does not reuse one fitted model
across trials. Each new comparator sidecar links to the original fresh archive
by SHA-256. No quarantined residuals were used as current experimental inputs.

## Numerical Boundary Check

The newly added capped-toy comparison exposed inverse-link cancellation at an
exactly accepted zero. The implementation now retains that boundary when its
forward self-score is accepted, rather than turning it into an empty region.
900 boundary regression checks passed. All 2,230 existing main CQR trials were
rechecked; none required a material change to its stored capped-envelope bounds.
New toy comparator outputs were recomputed after the correction.

## Saved Evidence

- `results/signed_gain_by_configuration.csv` and `results/signed_gain_by_toy.csv`
- `results/comparator_rerun_audit.csv` and `results/obsolete_table_classification.csv`
- `results/cqr_baselines/` and `results/toy_baselines/`
- `repair_settings.json` and the new trial archives under `results/absolute/`
- `results/final_audit.json` and `results/zero_boundary_corrections.json`

Original notebook-compatible synthetic tables have fresh replacements, including
the omitted methods and baseline-only setting. Historical manuscript figures
remain historical documents; this audit does not silently relabel them as reruns.

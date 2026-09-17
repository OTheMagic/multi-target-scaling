# Reusable experiment computations

This package separates data generation, fitting, residual construction, method
application, and evaluation. It contains no experiment runners, sweeps, plotting,
downloads, caching, or result saving. Importing it does not draw data or fit models.
Functions return arrays, fitted estimators, or dictionaries for callers to use.

## Modules and public functions

Notation: `n` observations, `p` predictors, `d` targets, `m` test observations.
Target arrays are `(n,d)`; fitting functions also accept `(n,)` for one target.
Sizes, distributions, coefficients, quantile levels, model parameters, and seeds
are arguments. Public functions are exported from `utility.experiments`.

| Module / function | Inputs and returned values |
|---|---|
| `generation.make_multitarget_regression` | Sizes, independent noise law/scales, informative count, coefficients and randomness. Returns `X (n,p), y (n,d)`, plus coefficients when generated. |
| `generation.make_multitarget_regression_dependent_noise` | Same contract, Gaussian noise with equicorrelation or AR(1). |
| `generation.make_partial_heteroskedastic_regression` | Same contract, plus affected-coordinate fraction and variance strength. |
| `generation.make_contaminated_regression` | Same contract, plus row-contamination probability and multiplier. |
| `generation.draw_matched_regression` | Explicit coefficients `(p,d)`, size, Gaussian or normal/uniform noise, seed. Returns `X,y`. |
| `generation.draw_fitted_interval_stress_regression` | Explicit coefficients `(p,d)`, noise scales and design parameters. Returns `X,y,width_covariates`, with widths `(n,d)`. |
| `generation.draw_regression_splits` | Generator, fixed coefficients, separate stream seeds, train/cal/test sizes, optional validation size and calibration-pool size. Returns named `X_*`, `y_*`, `dgp_coef`, and width covariates when applicable. |
| `absolute.fit_point_model` | Training `X,y`, optional sklearn estimator and parameters. Returns a newly fitted clone; default OLS. |
| `absolute.predict_absolute_residuals` | Fitted model, `X,y`. Returns `(predictions, abs(y-predictions))`, both `(n,d)`. |
| `absolute.apply_absolute_method` | Calibration scores `(n,d)`, one method, alpha, test count, keys/seeds and method options. Returns per-test adjustments/closure `(m,d)`, emptiness `(m,)`, and native method objects. |
| `absolute.evaluate_absolute_regions` | That result and test scores `(m,d)`; optional point predictions. Returns outcome-space metrics and optionally outcome bounds. |
| `quantile.fit_quantile_models` | Training arrays, explicit lower/upper probabilities, boosting or linear-quantile learner, parameters/seeds, target-scaling and linear-initialization options. Returns lower/upper model lists and fitting metadata. |
| `quantile.predict_quantile_intervals` | Fitted model dictionary and `X`. Returns raw/ordered lower and upper arrays `(m,d)`, crossing flags and rate. |
| `quantile.quantile_scores` | `y`, ordered lower/upper bounds. Returns signed scores `max(lower-y,y-upper)` and interval lengths, both `(n,d)`. |
| `quantile.fit_residual_quantile_intervals` | Training `X,y`, positive width covariates `(n,d)`, base alpha and optional coordinate factors. Returns OLS center model and empirical training-residual width multipliers. |
| `quantile.predict_residual_quantile_intervals` | That fit, `X`, positive width covariates. Returns center, lower, upper and base lengths `(m,d)`. |
| `quantile.quantile_diagnostics` | Supplied validation observations/raw predictions, probabilities, optional oracle predictions/scales. Returns coordinate tail rates, pinball losses, crossings, base coverage/lengths and optional oracle errors. Does not select or refit models. |
| `quantile.apply_quantile_method` | Signed calibration scores/lengths `(n,d)`, actual test lengths `(m,d)`, one method, alpha and keys/options. Returns adjustments/closure `(m,d)`, empty flags `(m,)`, native results and test lengths. |
| `quantile.evaluate_quantile_predictions` | That result, physical signed test scores, optional ordered base bounds. Returns outcome metrics and optional inverted bounds. |
| `real.load_arff_dataset` | Explicit path or readable file object and feature count. Returns complete-case feature/target DataFrames. |
| `real.prepare_real_data` | Dataset name and supplied feature/target frames. Returns deterministically cleaned, aligned `X,y`. Air targets are columns of its supplied feature frame. |
| `real.split_real_data` | `X,y`, calibration/test fractions and explicit split seed. Returns train/cal/test arrays or frames and positional row indices. |
| `real.make_real_estimator` | Dataset name, optional categorical columns/model overrides/missingness threshold. Returns an unfitted estimator or preprocessing pipeline. |

`_evaluation.py` is the only shared private helper module. It consolidates
coverage and outcome-size summaries; it contains no method calibration.
Detailed defaults and parameter contracts are in each function's docstring.

## Retained workflow mapping and sources

* **Absolute synthetic:** [synthetic_absolute.py](../../envelope_method/synthetic_absolute.py)
  supplies fixed-coefficient, independent train/cal/test draws and calibration-pool
  prefixes. Use generation, then point fitting, absolute residuals, one-method
  application and evaluation. The four main generators are extracted from
  [data_generator.py](../data_generator.py) and only the needed pure generation
  helpers in [build_experiment_update.py](../../reviewer_update/build_experiment_update.py).
* **Boosted CQR:** [runner.py](../../envelope_method/cqr10/runner.py), with ordinary
  fit/predict/score helpers from [exps.py](../exps.py). Use the shared generator,
  `fit_quantile_models`, prediction, signed scores, application and evaluation.
  External validation is diagnostic only. Core model parameters are explicit:
  `n_estimators=800, max_depth=3, min_samples_leaf=20, learning_rate=.05,
  n_iter_no_change=60, validation_fraction=.2, tol=1e-5`, with
  `standardize_targets=True, linear_init=True`. Ordinary helper defaults remain
  100 trees, depth 3, leaf size 5 and learning rate .05.
* **Matched point/direct-quantile:** [matched_comparison.py](../../envelope_method/cqr10/matched_comparison.py).
  `draw_matched_regression` supplies shared observations to the absolute functions
  and `fit_quantile_models(model="linear_quantile")`. Its default linear
  quantile model is unpenalized, with `solver="highs-ipm"` and an intercept.
* **Fitted-interval designs:** [pilot_misspecification.py](../../envelope_method/cqr10/pilot_misspecification.py).
  Use the stress generator and residual-width fit/predict functions, then the
  same signed-score, method and evaluation operations. Width quantiles use
  training residuals and NumPy's linear interpolation.
* **Real data:** [real_exps.ipynb](../../real_exps.ipynb), with preparation/model/split
  operations from [exps.py](../exps.py). Supported names are stock, rf1/rf2,
  scm1d/scm20d, student, energy, air and crime; rf1 shares the ARFF loader but
  has no call in the retained notebook. Split defaults are .75/.05/.20, using
  the same supplied seed for both shuffled sklearn splits. Reuse the absolute
  functions after preparation; no separate real-data prediction pipeline exists.

These sources are provenance, never runtime imports. Source metric computations
come from [experiments.py](../../envelope_method/experiments.py):`metric`.

## Methods, signed domains, and metrics

Absolute method names: `unscaled`, `ec`, `point_chr`, `signed_gwc`/`gwc`,
`envelope`, `bonferroni`, `naive`, `oracle`, `split_standardized`. Oracle mean
and scale must be independently supplied. Quantile names: `base`, `unscaled`,
`ec`, `bonferroni`, `naive`, `oracle`, `split_standardized`, `cqhr`,
`signed_gwc`/`gwc`, `envelope`. Point CHR remains an absolute-residual method
and is explicitly excluded from the quantile adapter. All calibration uses
[utility.methods](../methods/README.md), including its exact conformal ranks.

Unscaled, EC, Bonferroni, naive, oracle and split standardization now return
one native `Rectangle` per test observation in `objects`. The quantile adapter
calls those methods directly and supplies each observation's attainable score
bound `lower=-base_lengths_test[i]/2`, as it does for signed GWC/Envelope.
Calibration residuals are not clipped or compared with a test-specific lower
bound. Negative upper bounds remain negative; an upper bound below the domain
lower bound produces an empty rectangle. Absolute callers explicitly retain
the zero lower bound and nonnegative-score validation.

`objects` is a list of `Rectangle` instances for those six methods, a list of
`EnvelopeRegion` instances for GWC/Envelope, a single `CQHRResult` for CQHR,
and `None` for Base. All adapters retain `adjustments (m,d)`,
`upper_closed (m,d)` and `empty (m,)`; quantile results also retain actual
test interval lengths and the calibration/test keys used. Evaluation calls
native membership rather than interpreting numeric upper bounds as closed sets.
Unscaled no longer uses a unit-length CQHR adapter. CQHR preserves its
reference-coordinate and minimum-length convention. Naive remains a heuristic;
adding a signed domain does not establish a coverage guarantee for it or EC.
CQR
scores remain signed; negative adjustments shrink intervals. Quantile crossings
are ordered coordinatewise, with raw predictions retained for diagnostics.
Outcome inversion is `base_lower-adjustment, base_upper+adjustment`; absolute
inversion is `point_prediction ± adjustment`. Both outcome faces inherit the
score upper-face closure. Empty masks must be consulted even when numeric
endpoints are returned. Closed zero-width singletons can be nonempty but have
zero volume; open singletons are empty. Empty rows have zero lengths/coverage.

**Every evaluator reports outcome-space volume and base-10 log volume.**
Per-row outputs are `outcome_lengths (m,d)`, `outcome_volumes (m,)`,
`log10_volumes (m,)`, `coordinate_covered (m,d)`, `covered (m,)`, and `empty (m,)`.
Summaries include `test_coverage`, `coordinate_coverage (d,)`, `outcome_volume`
(mean volume), `mean_log10_volume` (mean log, not log of mean),
`coordinate_lengths (d,)`, `mean_adjustments (d,)`, and `empty_rate`.
This approved metric change replaces real score-space volumes and synthetic
natural logarithms. Any zero side gives volume 0/log -inf, even with another
infinite side. Positive infinite volume gives log +inf; a mean containing both
infinite signs is NaN, with no observations discarded. Log-space recovery and
scaled nonnegative means guard against intermediate product/sum overflow.

## Reproducibility, preprocessing, and dependency boundary

The main generators preserve source draw order; the independent generator
interleaves coefficient and noise draws. Their coefficients are `(d,p)`; matched/stress generators
use `(p,d)`. Current synthetic usage has two informative predictors and no
coefficient normalization. To replay coefficient initialization, the absolute
source uses a generator draw of size `n_train+n_test`; boosted CQR uses size 1.
Matched/stress source coefficients are `arange(1,p*d+1).reshape(p,d)/(p*d)`.
Supply the resulting fixed coefficients to all split draws.

Explicit seeds/Generators replace hidden trial state. The matched generator
preserves integer-seed named substreams; its new Generator option draws an
integer root seed. Full legacy trial hashing is not reproduced automatically.
Calibration-pool size must stay fixed to preserve nested calibration prefixes.
Preserved distribution quirks: Cauchy/t ignore supplied noise scales; Gamma
uses zero-based target index as shape, including zero for its first target.
Shape and finite-scale checks now reject malformed generator inputs explicitly.

Method adapters share calibration keys across tests and supply independent test
keys, replacing numeric-only legacy membership and test-width deduplication.
EC/Bonferroni use `coordinate_keys (n,d)` and `test_keys (m,d)`; the other
quantile methods use `scalar_keys (n,)` and `test_keys (m,)`. Each test key
remains fixed across all candidate outcomes for that observation. Oracle
requires independent `mu` and positive `std`, scalar or `(d,)`; fitted
standardization methods still require positive finite sample scales. Quantile
split standardization uses `split_random_state=42` by default and reuses the
same split and scalar calibration keys for every test row.
Point CHR and split standardization require a fixed integer split seed. Missing
keys use a local Generator; supplied keys must meet the reviewed methods'
independence requirements. Exact unresolved key collisions raise; they are not
replaced with numeric jitter. Legacy boundary counts are not promised identical.

Air's mean imputer and crime's missingness selector now fit **training rows
only**, the approved correction to full-cohort preprocessing. Student encoding
already follows this rule. Crime retains its policy of no further imputation;
its forest must support remaining NaNs. Boosted target scaling fits the supplied
training set before the learner's internal stopping split, as in the source.

Dependencies are standard library, NumPy, pandas, SciPy, scikit-learn,
`utility.methods`, and this package. The approved indirect-only exception is
exactly `utility.rectangle`, `utility.signed_geometry`, `utility.tie_breaking`,
and `utility.conformal_utils`, including their existing mutual imports. The new
package imports none of those four directly and uses no path manipulation.

## Verification and limits

Run `python -B -m unittest utility.experiments.test_experiments` in an environment
with the existing scientific dependencies. **21 compact tests passed**, covering passive
imports/no data IO, transitive local dependency boundary, shapes at multiple
sample sizes, reproducibility, split agreement, training-only fitting,
crossings/signed residuals, current-method agreement, randomized boundaries,
shrinking/empty intervals, all six signed Rectangle methods, native return
objects, coordinate/scalar key reuse, Point CHR's signed-path exclusion,
infinite conformal cutoffs, and outcome metrics. Signed checks include both
negative and mixed-sign calibration scores, different test-specific domains,
and closed/open singletons. These verify execution, native-method agreement,
and interval inversion; they do not prove coverage or general mathematical
correctness.
Additional one-time checks compared 52 generator cases and the ordinary quantile
helpers exactly against isolated original functions without importing drivers.
Checks used the existing installed third-party distributions; no dependency
was added. No research data, plots, model artifacts or intermediate arrays were
saved, and no experiment suite ran.

Actual datasets, large production fits, statistical coverage guarantees, and
extreme numerical behavior inside the reviewed methods were not revalidated.
The methods README's existing limitations still apply. Real-data tests use
small supplied frames and in-memory ARFF input; they do not certify every raw
dataset schema or behavior across dependency versions.

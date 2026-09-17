# Audited method implementations

The six primary methods and four subsequently requested additional baselines
are included. Algorithms live here and do not import legacy methods or experiment
drivers. Existing utility files were inspected before extraction. Dependencies
are NumPy and scikit-learn (split methods); no dependencies were added.
The new utility/experiments package calls these methods; legacy experiment
drivers and notebooks still use their existing entry points. The file-by-file
usage audit below records the earlier extraction-stage review.

## Definitions and paper audit

- **EC:** Messoudi, Destercke and Rousseau (2021),
  [Copula-based conformal prediction for multi-target regression](https://arxiv.org/pdf/2101.12002),
  §§3.2–3.3.3, equations (25)–(27), Remark 4.1. Use empirical marginal ranks
  divided by n, one common marginal probability level, and empirical inversion.
  The same calibration sample fits the margins and copula. This construction
  does not establish a general finite-sample joint coverage guarantee; §4.2
  reports undercoverage in some examples.
- **Point CHR / CQHR:** Sampson and Chan (2024),
  [Conformal Multi-Target Hyperrectangles](https://arxiv.org/pdf/2406.04498),
  Algorithms 1–2, §§2–3, Theorem 1 proof. Point CHR is the absolute-residual
  specialization; factors of two cancel from full-width ratios. Each split
  uses its own conformal rank. CQHR matches where its width floor is inactive.
  The floor and Point CHR's 50/50 split are implementation choices.
- **Signed GWC / Envelope:** the project definition is
  [signed_envelope.tex](../../envelope_method/signed_envelope.tex),
  equations psiclosed, gwcscore–G, and surface–L.
- **Additional baselines:** [supplementary.tex](../../multi_target_scaling_latex/supplementary.tex),
  algorithms `alg:bonferroni`, `alg:std-plug-in`, `alg:std-ds`, and
  [body.tex](../../multi_target_scaling_latex/body.tex), `sec:pop-oracle`,
  `eq:residual-transformation` and `eq:set-valued-map`. These supply the
  definitions for the independent examples below; no new paper was assumed.

The paper audit used the supplied arXiv v1 PDFs. Random secondary keys are the
subsequently approved extension for ties, not an assertion that either paper
prescribes this exact software interface. Small checks are not coverage proofs.

## Inputs, outputs, and random ties

Calibration scores have shape (n, d); predictor fitting and residual generation
remain outside this package. Unscaled, EC, Bonferroni, naive standardization,
population oracle, and split standardization accept finite signed scores and
return Rectangle with (d,) lower, upper, and upper_closed arrays and an empty
property. Each accepts keyword-only lower=0.0. **Point CHR remains unchanged:**
it expects nonnegative residuals and returns a Rectangle with zero lower bounds.
Signed GWC and Envelope already accept finite signed scores and return EnvelopeRegion:
(d,) lower, upper, gwc_upper and upper_closed; scalar q_gwc, q_gwc_key,
test_key; empty, fallback, and evaluations diagnostics.

For signed methods, lower is a scalar or (d,) attainable **test-score** bound,
finite or -inf. Zero preserves the absolute-residual domain. For signed CQR,
pass lower=-test_interval_widths/2; use -inf when no finite lower bound is known.
Calibration scores remain unchanged even when below a test observation's lower
bound. Negative cutoffs are preserved: upper < lower, or equality at an open
upper face, makes the region empty. Empty regions reject all candidates and
have zero side lengths/volume. Respect empty during score inversion; score-space
volume generally differs from outcome-space volume. CQHR's output is described
below. Defaults remain alpha=.2 for Unscaled/EC/Point CHR and the four additional
baselines; alpha=.1 for CQHR/GWC/Envelope.
The conformal rank is ceil((n+1)*(1-alpha)), with an infinite cutoff above n.
The approved rank correction evaluates it exactly using Fraction(str(alpha)):
floating probabilities mean their displayed decimal values, and Fraction inputs
preserve explicitly supplied rational probabilities. For example, n=24 gives
rank 7 at alpha=.72 and rank 8 at alpha=.71999. No two-decimal rounding or
integer-boundary tolerance is applied. Pass Fraction(1, 3) if exactly one third
is intended; the decimal representation of the float 1/3 is slightly smaller.
EC instead uses its empirical common-level search.

**One comparison convention:** order (numeric value, independent uniform key)
lexicographically. A smaller numeric value always wins; keys matter only for
exact numeric equality. There is no added noise, tolerance, or index-based
tie-breaking. Keys lie in [0,1). Generated collisions within one draw are
redrawn; unresolved duplicate pairs during ranking or boundary comparison raise
ValueError rather than silently using row order. NaNs cannot be ranked.
Coverage arguments require independent uniform calibration/test keys, also
when supplied explicitly; fixed key fixtures only verify deterministic behavior.

| Method | Coordinate residual ranks | Scalar calibration ranks |
| --- | --- | --- |
| Unscaled | None | Keys for row maxima |
| EC | Keys for empirical margins and inverse boundaries | None |
| Point CHR | Keys for shape-split marginal quantiles | Independent keys for second-split excess maxima |
| CQHR | None | Keys for rescaled row maxima |
| Signed GWC | None | Keys for global supremum row maxima |
| Envelope | Distinct numeric knots only | Same row keys reused in GWC and every surface cell |
| Bonferroni | Keys for marginal quantiles | None |
| Naive / population oracle / split standardization | None | Keys for standardized row maxima |

**Raw ties in GWC and Envelope remain numerically equal.** Neither method
statistically ranks raw coordinates. GWC uses their means, standard deviations,
and score suprema. Envelope uses np.unique only to form geometric cell endpoints;
repeated equal endpoints are one knot, not a statistical-rank tie. Adding raw
noise or key-only cells would change that construction. Raw keys are therefore
used in EC and Point CHR, where coordinate order statistics actually occur.
Point CHR's selected coordinate keys do not alter its numeric base widths.

Default random_state=42 gives reproducible local generators; missing keys are
drawn sequentially for calibration and then the test instance. Supply explicit
coordinate_keys (n,d), scalar_keys (n,), and test_key (scalar) or test_keys
(EC: (d,); CQHR: (m,)) as documented by each function.

**A region fixes randomness for one test observation.** Its contains and
contain_points methods honor open/closed upper faces; a batch passed to
contain_points consists of candidate scores sharing that test key. Bare
point <= upper comparisons discard randomization. CQHR supplies separate
keys for its m test rows. Reuse calibration keys across tests, and draw independent
test keys. EnvelopeCalibration.predict draws a fresh test key on each call.
Repeated convenience calls with the same integer seed repeat their keys; pass
explicit keys, or a shared Generator with fixed supplied calibration keys.
Point CHR also passes random_state to scikit-learn for its split: retain an
integer split seed and supply explicit keys when controlling successive tests.

~~~python
import numpy as np
from utility.methods import gwc_prediction, cqhr_adjustments

scores = np.array([[0.], [0.], [2.]])
region = gwc_prediction(scores, alpha=.5, scalar_keys=[.1, .4, .7],
                        test_key=.2)
accepted = region.contains([0.])  # includes this test instance's boundary rule

result = cqhr_adjustments(np.ones((3, 2)), np.ones((3, 2)),
                          np.ones((2, 2)), alpha=.5,
                          scalar_keys=[.1, .4, .7], test_keys=[.2, .8])
accepted = result.contains(np.ones((2, 2)))  # [True, False]
adjustments = result.adjustments            # numeric widths alone omit the keys
~~~

Signed CQR example (one test interval [0,4], attainable score lower bound -2):

```python
from utility.methods import unscaled_prediction

region = unscaled_prediction(np.array([[-2.], [-1.], [0.]]), alpha=.5,
                             lower=-2., scalar_keys=[.1, .4, .7], test_key=.2)
# upper=[-1.], so inversion gives [0-(-1), 4+(-1)] = [1,3].
accepted = region.contains([-1.5])
```

utility/experiments/quantile.py passes each test row's actual -width/2 to
native region predictors, reuses calibration keys, and retains native objects.
Only CQHR returns CQHRResult; Unscaled now returns its own Rectangle. Outcome
lengths are base_width+2*upper, with empty rows zeroed and endpoint closure
preserved. Point CHR is available only through the absolute-residual dispatcher.

## Methods and provenance

### Unscaled — unscaled.py

Public: unscaled_prediction. Source: [utility/unscaled.py](../unscaled.py), same
function. Compute row maxima, their conformal pair cutoff, and equal upper
bounds in all dimensions. Bonferroni is extracted separately in additional.py.
The source's positive 1e-10 jitter was replaced with exact scalar secondary keys.
Checks: imports/execution, documented numeric difference from legacy jitter,
hand rank-three bound [4,4] from the
[Unscaled manuscript algorithm](../../multi_target_scaling_latex/supplementary.tex),
randomized tied boundaries, empty-calibration rejection, shapes and infinite-rank
fallback. The approved signed extension adds lower and empty-region support;
finite nonempty (n,d) inputs are validated, never automatically absolutized.
Signed rank/translation, lower-domain membership, invalid lower bounds and empty
regions are checked; scalar calibration and cutoff formulas are unchanged.

### Empirical Copula — ec.py

Public: EmpiricalCopula.fit/cdf/quantile_box, inverse_ecdf_transform,
empirical_copula_prediction. Source: [utility/copula.py](../copula.py), same names.
Rank residual/key pairs coordinatewise and divide by n; scan common levels for
the first empirical joint CDF at least 1-alpha; invert the same ordered margins.
Thus the paper's empirical CDF is applied to augmented, lexicographically ordered
observations. U is (n,d); CDF queries and thresholds are (d,); fit returns None.
inverse_ecdf_transform returns numeric (d,) bounds, or (bounds, keys) when
return_keys=True and the fit's coordinate_keys are supplied.

Approved changes from legacy: denominator n instead of n+1, common-level search,
and direct lookup on the probability grid instead of multiplication/ceil.
The first crossing need not attain the target exactly. Empty margins and
out-of-range inverse probabilities retain +inf, an API fallback.
Checks: hand ranks/CDF/inverse, tied coordinate acceptance/rejection, row/key
permutation, untied legacy bound [3,3], and inverse-rounding regression
(scores 1..24, alpha=.72: corrected 7 versus legacy 8).
The approved signed extension adds lower and empty-region support; fit, inverse
and prediction validate finite (n,d) margins with d>0, retaining the empty-fit
fallback. Signed marginal bounds, translation and lower-domain membership are
checked. No extra scalar conformal stage is introduced. Alpha validation remains
incomplete; finite-sample joint coverage is not established.

### Point CHR — point_chr.py

Public: data_spliting_CHR_prediction (original spelling retained). Source:
[utility/data_splitting.py](../data_splitting.py), same function.
Split rows 50/50; take first-half marginal quantiles; rescale second-half
excesses by base_upper[reference_dim]/base_upper; calibrate row maxima and
map the adjustment back. reference_dim defaults to 0. Keys follow split indices;
independent scalar keys govern final membership. Numeric formulas are unchanged.
Checks: numeric legacy agreement, randomized boundaries, and Algorithm 1's
odd-sample example (i,2i), i=1..7, seed 42, alpha=.3: ranks 3 and 4, upper [6,12].
Zero bases still make the original division undefined; the new rank helper
rejects resulting NaNs explicitly instead of returning a NaN rectangle.
No floor or fallback was invented. Reference/shape validation remains incomplete.
Historical first-half-rank reuse was already corrected in the selected source.

### CQHR — cqhr.py

Public: cqhr_adjustments and CQHRResult. Source:
[utility/cqhr.py](../cqhr.py), cqhr_adjustments/_validate_cqhr_inputs.
Signed CQR scores and calibration interval lengths are (n,d); test lengths
are (m,d). Floor lengths at min_length=1e-12, rescale by each row's
reference-length/coordinate-length ratio, calibrate row maxima, then invert
using test lengths. reference_dim defaults to 0; negative adjustments shrink.

**API change:** CQHRResult replaces the three-tuple. It exposes adjustments
(m,d), quantile_adjustment, calibration_scores (n,), quantile_key, test_keys
(m,), upper_closed (m,), and contains(raw_scores_test) returning (m,) decisions.
This keeps the test-time key comparison available. The width floor is preserved
but is not in the paper. Callers handle outcome inversion and empty intervals.
Checks: numeric legacy agreement for both reference dimensions; Algorithm 2
example gives row scores [1,-2,3], q=1, adjustments [[1,3],[1,.5]];
randomized boundaries, negatives, zero widths and invalid shapes/parameters.
Nonfinite validation remains incomplete.

### Signed GWC — gwc.py

Public: gwc_prediction. Source: the global stage of
[utility/envelope.py](../envelope.py), EnvelopeCalibration.__init__/predict.
Take coordinate score suprema over [lower,+inf), row maxima, a conformal
pair cutoff, and the signed inverse link. _gwc_from_statistics shares this
stage with Envelope. Numeric ddof=0 statistics and the inverse-cancellation
guard remain. No Envelope local search is run; evaluations are zero.
Checks: numeric original-stage agreement with a test key forcing closed faces,
independent signed formulas, tied scalar acceptance, invalid/empty domains,
-infinite lower bounds, infinite rank, and zero scale.
The original whole-domain fallback for any zero-scale coordinate is retained:
keys resolve ranks, not zero numerical variance. Extreme overflow is unverified.

### Envelope — envelope.py

Public: EnvelopeCalibration.predict and envelope_prediction; EnvelopeRegion
is exported. Source: [utility/envelope.py](../envelope.py), same names.
After signed GWC, form cells at distinct interior residual values; bound the
other coordinates over global intervals; calibrate each surface and retain the
largest surviving endpoint. search is backward, exhaustive, or rank;
return_cells=True returns the region plus evaluated cells. Global/cell cutoffs
use the same calibration row keys and one test key. Cells record q_key and
upper_closed; the result respects global open boundaries.
Extraction shared the global stage/geometry and retained local link_values.
Checks: numeric legacy agreement under closed-face keys across seven cases and
all searches; explicit tied key cases, search agreement and signed definitions.
For scores [-2,-1,0], alpha=.5, lower=-2, numeric GWC upper is
-1+1/sqrt(12), Envelope upper is -1; stationary/infinite-endpoint checks also pass.
General coverage and all-input search equivalence are not established.

### Additional baselines — additional.py

All four functions are exported from utility.methods, accept finite signed
scores (n,d), and return Rectangle bounds (d,) with open/closed upper faces.
Each now accepts lower=0.0 with the test-domain/empty-region convention above;
the shared _standardized_rectangle passes it through without changing calibration.
Defaults are alpha=.2 and random_state=42. They use the existing rectangle,
exact-rank, and random-key helpers; no legacy module is imported.

| Public function | Original source | Computational steps and assumptions |
| --- | --- | --- |
| bonferroni_prediction | [unscaled.py](../unscaled.py), same function | Allocate alpha/d to each marginal; take coordinate pair quantiles. coordinate_keys is (n,d), test_keys is (d,). Division uses Fraction(str(alpha))/d before ranking. |
| naive_prediction | [data_splitting.py](../data_splitting.py), same function | Estimate mean and sample std (ddof=1) from all rows; calibrate standardized row maxima; invert q*std+mean. Same-sample fitting remains a heuristic without the usual finite-sample guarantee. |
| data_splitting_oracle_prediction | [data_splitting.py](../data_splitting.py), same function | Standardize with supplied fixed mu/std, calibrate all rows, and invert. No data split occurs. mu/std are scalars or (d,); std must be positive. This is the population oracle, not res_rescaled.oracle_prediction's transductive calculation. |
| data_splitting_standardized_prediction | [data_splitting.py](../data_splitting.py), same function | Preserve the 50/50 split: first half estimates mean/sample std, second half calibrates standardized maxima using its own sample size. |

The three standardization functions accept scalar_keys (n,) and a scalar
test_key. Split-standardization keys follow the original row indices into the
second half. Keys are independent of raw coordinates and only rank final scalar
scores. Use region.contains/contain_points to retain randomized boundaries.

**Explicit changes:** additive jitter is replaced by the shared random-key
convention. Scalar oracle parameters now broadcast correctly to all d output
bounds (the source could produce a scalar and fail Rectangle construction).
Invalid shapes, nonfinite residuals, and nonpositive/nonfinite scales
raise ValueError. Naive needs at least two rows; split standardization needs at
least four, with positive fitted scales. These checks expose undefined divisions;
no replacement variance, floor, or fallback was invented. Numeric formulas,
ddof=1, default alpha, and split seed are preserved. Extreme overflow and
last-bit differences between affine score comparisons and inverted bounds remain
unverified; key randomization applies to exact floating equality.

**Checks:** all four import and execute. Separate manuscript examples give
Bonferroni [3,3], naive [2,4], population oracle [3,3], and split standardization
[6,12]. Actual legacy outputs agree within their former jitter perturbation;
with only that jitter disabled in memory, numeric formulas agree exactly.
This does not establish coverage, particularly for naive standardization.
Coordinate/scalar tie acceptance, boundary rejection, shapes, scalar oracle
broadcasting, invalid scales/inputs and infinite ranks were checked. Signed
rank/translation examples, lower-domain membership, unbounded lower faces and
empty regions are checked for all four; negative residuals are now valid.
For n=9, d=3, alpha=.3, exact Bonferroni allocation selects the marginal maximum;
forming .3/3 as a float would incorrectly select an infinite rank under the
decimal-rank convention. This boundary is covered by a regression check.

```python
from utility.methods import bonferroni_prediction

region = bonferroni_prediction(scores, alpha=.2, random_state=42)
accepted = region.contains(candidate_residuals)  # one test instance's fixed keys
```

## Shared helpers, source distinctions, and limits

- [tie_breaking.py](../tie_breaking.py): key draws/validation, lexicographic
  ordering, coordinate/scalar pair quantiles, candidate pair comparison.
- [rectangle.py](../rectangle.py): geometric bounds and open upper faces;
  scalar lower bounds broadcast to (d,), with finite or -inf values.
  allow_empty=True preserves inverted bounds for the six signed predictors;
  its default False preserves legacy constructor rejection. Empty membership,
  lengths/intersections and zero volume with a singleton/unbounded side are
  checked. Existing plotting/update helpers were not extended or exercised.
  [signed_geometry.py](../signed_geometry.py): score_at, interval_sup, link,
  and EnvelopeRegion. Numeric score formulas remain unchanged.
- [conformal_utils.py](../conformal_utils.py): corrected exact-arithmetic rank and legacy
  inclusive quantile/jitter helpers. New statistical calibration uses pair
  quantiles. Envelope's rank-search pruning alone uses the numeric quantile
  conservatively, retaining its outward 64*eps guard.
- Relevant notebooks were inspected as JSON, not executed. Archived utility
  copies matched the extracted sources before approved changes. Signed baseline
  adaptation in utility/exps.py is not imported. The Point CHR pilot and CQHR
  quarantine toy have assumption/validation differences and were not substituted.
- The user selected signed GWC over res_rescaled.standardized_prediction("GWC"),
  whose domain, jitter, inverse, alpha and degenerate behavior differ. The
  meeting-report full-cell Envelope reference is a different coupled construction,
  not this project-dispatched surface method. No identity question remains.

[test_methods.py](../test_methods.py) contains the reusable method checks:
python -B -m utility.test_methods. **28 tests passed**, using existing packages
in tmp/diagnostic_packages. Execution, numeric legacy agreement (including
test_key=0 to recover closed faces), intentional random membership changes,
and independent paper/project examples are distinguished. A separate import
audit found no file writes, research-data reads, or legacy/experiment imports
(third-party package metadata and styles were excluded from the data check).
The separate utility/experiments/test_experiments.py suite has **21 passing
small checks**, including native signed returns, per-test lower bounds, mixed
signs, shrinkage, randomized singleton boundaries, empty intervals, infinite
ranks and passive imports. No full experiments were run and no datasets,
plots, models, execution output or intermediate arrays were saved.

The former conformal_rank(24,.72) result of 8 is corrected to 7 under the decimal
convention above. Boundary and adjacent-float checks also confirm that genuinely
above-integer positions retain the higher rank and infinite-rank fallback.
The shared correction also reaches legacy callers importing this helper;
experiment scripts/settings and stored results were not changed or rerun.

One numerical issue remains separate from rank calculation and random ties:
exact floating score equality can differ from mathematical equality.
With residuals [0,0,10], a cutoff can be -0.5 while the corresponding self-score
evaluates to -0.5000000000000001. That comparison is numerically strict, so keys
do not decide it. Existing self-score guards remain; no isclose convention or
numerical repair was introduced.

## Usage audit: utility files outside methods/

Static review covered repository Python sources and notebook **code cells**,
including aliases, dispatchers, and internal helper calls. Historical snapshots,
quarantine files, report-local utility copies, and notebook-local redefinitions
were distinguished from calls to this root package. Third-party dependencies
were excluded. No experiment module or notebook was executed for this audit.
"No caller found" means a repository-local cleanup candidate, not proof that a
public API has no external/dynamic users. This is the extraction-stage audit,
not a refreshed reference index following addition of utility/experiments.
Nothing below was deleted or migrated.

| File | Current use | Unused functions or logic found |
| --- | --- | --- |
| [__init__.py](../__init__.py) | Empty package marker. | No executable logic to remove. |
| [conformal_utils.py](../conformal_utils.py) | All three functions used: new rank helpers, Envelope pruning, and legacy calibration. | None. add_jitter is unused by methods/ but still called by legacy methods and exps.py. |
| [copula.py](../copula.py) | All class/helper functions used by its predictor, exps.py (also direct CDF/inverse calls), notebooks, and comparison checks. | Predictor random_state argument unused; commented splitting code inactive. The legacy EC implementation still has callers. |
| [cqhr.py](../cqhr.py) | Validator and cqhr_adjustments used by exps.py, envelope_method/experiments.py, CQR10 pilots, and comparisons. | None found. Its tuple-returning API remains active. |
| [data_generator.py](../data_generator.py) | Both main generators and all four private helpers used by experiments/notebooks. | Alias make_multi_target_regression_dependent_noise has no caller found. See branch-specific behavior below. |
| [data_splitting.py](../data_splitting.py) | All four predictors used by exps.py dispatch, notebooks, diagnostics, or legacy comparisons. | No unused function found. Extraction does not make these live legacy entry points redundant. |
| [envelope.py](../envelope.py) | Geometry, calibration/prediction and all search branches used by exps.py, envelope_method/experiments.py, verification and comparisons. | Legacy EnvelopeRegion.info has no resolved caller found; same-named Rectangle.info calls are unrelated. |
| [exps.py](../exps.py) | Active legacy experiment driver; most entry points/helpers have callers. | _quantile_residual and heavy_t have no calls found. run_synthetic_experiment is imported but uncalled in exps.ipynb; its only actual call is in the uncalled single_dim_text_file helper. Local seed assignments in run_abs_res_synthetic_experiment and run_cqr_synthetic_experiment are unused. |
| [graphing_tools.py](../graphing_tools.py) | Reviewer plotting/table APIs and older comparison plots have notebook callers. | No caller for single_dim_text_file or module build_panel; format_pm/extract_stats only feed that uncalled build_panel. summarize_infinite_volume_rate is imported but uncalled. generate_latex_table_from_csvs ignores method_exlude. build_panel computes minimum-volume/highlight choices but always passes highlight=False. |
| [project_paths.py](../project_paths.py) | data_path, resolve_artifact and constants used by experiments, report/archive consumers and tests. | None found. |
| [rectangle.py](../rectangle.py) | Construction, empty, info, lengths/volume, intersection and membership used by current/legacy methods and checks. intersects currently has only a test caller. | No caller found for update_upper, update_lower, same_as or plot. dimensions and draw_2D are only reached by plot, so that plotting subtree has no entry call found. |
| [res_rescaled.py](../res_rescaled.py) | The standardized_* family, mean_index_solver, create_hyper_rectangle, check_coverage_rate and mean_clip are live through exps.py/diagnostics; shortcut and exhaustive paths remain reachable. | No caller for scaled_prediction; scaled_transformation, scaled_upperbound and scaled_threshold only serve that uncalled family. Import combinations is unused; product is used. |
| [signed_geometry.py](../signed_geometry.py) | All numerical helpers and EnvelopeRegion construction used by new GWC/Envelope; returned-region API exercised by checks and the new experiment package. | No unused implementation identified. Legacy experiment consumers still use the separate legacy region class. |
| [test_methods.py](../test_methods.py) | MethodChecks discovered by unittest; tests cover the primary and additional methods. | None found; test discovery is a use even without explicit function calls. |
| [test_project_paths.py](../test_project_paths.py) | Three pytest-discovered path tests. | None found. Inspected, not executed during this audit. |
| [tie_breaking.py](../tie_breaking.py) | All six helpers used directly by methods or internally by other helpers. | None found. |
| [unscaled.py](../unscaled.py) | Both predictors still called by exps.py dispatch/notebooks and legacy comparisons. | No unused function found; old numerical jitter remains active on these paths. |

The subsequently added [experiments/](../experiments/README.md) has its own
interface documentation and checks. Generated __pycache__/ bytecode was left
alone. Extraction-stage usage evidence includes
exps.py's method dispatcher, envelope_method/experiments.py, exps.ipynb,
smoke_test.ipynb, reviewer_abs_res_experiments.ipynb and real_exps.ipynb.
real_exps.ipynb defines its own build_panel/format_pm/extract_stats: those are
duplicate implementations, not evidence of calls to graphing_tools.py.

Additional observations, **not unused-code or deletion conclusions**:

- graphing_tools.compare_across_dims has a call in exps.ipynb without an import,
  so fresh-kernel execution needs review. graphing_tools also calls sns.set_theme
  on import; that is active plotting configuration, not dead logic.
- data_generator._draw_noise ignores scale for its Cauchy and Student-t branches;
  the gamma branch uses zero-based target_index as shape, hence zero in the
  first coordinate. These existing distribution choices need a separate
  scientific decision; none were changed here.
- exps.py still imports legacy methods. Its signed-baseline adaptation still
  uses additive jitter for Unscaled and the old EC construction. New package
  extraction does not update those experiment paths or their membership rules.

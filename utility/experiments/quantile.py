"""Fitting, signed residuals and evaluation for retained quantile designs.

Functions fit only explicitly supplied training observations.  Prediction and
calibration are separate operations; importing this module does neither.
"""

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler

from utility import methods

from ._evaluation import summarize_predictions


__all__ = [
    "fit_quantile_models", "predict_quantile_intervals", "quantile_scores",
    "fit_residual_quantile_intervals", "predict_residual_quantile_intervals",
    "quantile_diagnostics", "apply_quantile_method",
    "evaluate_quantile_predictions",
]


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or min(value.shape) < 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite two-dimensional array")
    return value


def _targets(value, name="y"):
    value = np.asarray(value, dtype=float)
    if value.ndim == 1:
        value = value[:, None]
    return _matrix(value, name)


def _training_arrays(X, y):
    X, y = _matrix(X, "X_train"), _targets(y, "y_train")
    if len(X) != len(y):
        raise ValueError("X_train and y_train must have the same number of rows")
    return X, y


def _probability(value, name):
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between zero and one")


def fit_quantile_models(
    X_train, y_train, *, lower_quantile, upper_quantile,
    model="gradient_boosting", model_params=None, random_state=42,
    standardize_targets=False, linear_init=False,
):
    """Fit two models per target and return a dictionary of fitted models.

    X_train is (n,p), y_train is (n,d) or (n,).  Quantile probabilities are
    explicit, usually base_alpha/2 and 1-base_alpha/2.  They are distinct from
    the later conformal miscoverage alpha.  Supported model names are
    ``gradient_boosting`` and ``linear_quantile``.

    Boosting defaults reproduce the ordinary source helper: 100 trees, depth
    3, learning rate .05 and minimum leaf size 5.  The retained core uses
    explicit parameters (800 trees, depth 3, leaf size 20, early stopping),
    standardize_targets=True and linear_init=True.  Linear quantiles default
    to alpha=0, solver='highs-ipm', fit_intercept=True, as in the matched design.
    Scaling is fit on this training input only.  Seeds are seed+2*j+k modulo
    2**32, following the current core convention.
    """
    X_train, y_train = _training_arrays(X_train, y_train)
    _probability(lower_quantile, "lower_quantile")
    _probability(upper_quantile, "upper_quantile")
    if lower_quantile >= upper_quantile:
        raise ValueError("lower_quantile must be less than upper_quantile")
    if model not in ("gradient_boosting", "linear_quantile"):
        raise ValueError("model must be gradient_boosting or linear_quantile")
    if not isinstance(random_state, (int, np.integer)):
        raise TypeError("random_state must be an integer model seed")
    supplied = dict(model_params or {})
    controlled = {"quantile"} if model == "linear_quantile" else {
        "loss", "alpha", "random_state", "init",
    }
    if controlled.intersection(supplied):
        raise ValueError("Use explicit function arguments for quantiles, seeds and initialization")
    if model == "linear_quantile" and linear_init:
        raise ValueError("linear_init applies only to gradient boosting")
    params = (dict(n_estimators=100, max_depth=3, learning_rate=.05,
                   min_samples_leaf=5) if model == "gradient_boosting" else
              dict(alpha=0., solver="highs-ipm", fit_intercept=True))
    params.update(supplied)
    lower_models, upper_models = [], []
    for j in range(y_train.shape[1]):
        for k, (q, destination) in enumerate(
            ((lower_quantile, lower_models), (upper_quantile, upper_models))
        ):
            if model == "gradient_boosting":
                estimator = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    random_state=(int(random_state) + 2*j + k) % (2**32),
                    init=LinearRegression() if linear_init else None, **params,
                )
            else:
                estimator = QuantileRegressor(quantile=q, **params)
            if standardize_targets:
                estimator = TransformedTargetRegressor(
                    regressor=estimator, transformer=StandardScaler(),
                )
            estimator.fit(X_train, y_train[:, j])
            destination.append(estimator)
    return dict(lower_models=lower_models, upper_models=upper_models,
                quantiles=(float(lower_quantile), float(upper_quantile)),
                model=model, model_params=params,
                standardize_targets=bool(standardize_targets),
                linear_init=bool(linear_init))


def predict_quantile_intervals(fitted, X):
    """Return raw/ordered lower and upper predictions, each (m,d).

    Crossed predictions are ordered coordinatewise, preserving the retained
    convention.  ``crossings`` records raw_lower > raw_upper; its overall mean
    is ``crossing_rate``.  Raw predictions remain available for diagnostics.
    """
    X = _matrix(X, "X")
    lower_models, upper_models = fitted["lower_models"], fitted["upper_models"]
    if not lower_models or len(lower_models) != len(upper_models):
        raise ValueError("fitted must contain equally sized nonempty model lists")
    raw_lower = np.column_stack([estimator.predict(X) for estimator in lower_models])
    raw_upper = np.column_stack([estimator.predict(X) for estimator in upper_models])
    raw_lower, raw_upper = _matrix(raw_lower, "raw_lower"), _matrix(raw_upper, "raw_upper")
    crossings = raw_lower > raw_upper
    return dict(lower=np.minimum(raw_lower, raw_upper),
                upper=np.maximum(raw_lower, raw_upper), raw_lower=raw_lower,
                raw_upper=raw_upper, crossings=crossings,
                crossing_rate=float(crossings.mean()))


def quantile_scores(y, lower, upper):
    """Return signed CQR scores and base lengths, both (m,d).

    Inputs are ordered lower/upper bounds and observations of matching shape.
    Scores are max(lower-y, y-upper), including negative scores inside the
    base interval.  There is no truncation, shift or calibration in this step.
    """
    y, lower, upper = _targets(y), _matrix(lower, "lower"), _matrix(upper, "upper")
    if lower.shape != y.shape or upper.shape != y.shape:
        raise ValueError("y, lower and upper must have matching shapes")
    if np.any(lower > upper):
        raise ValueError("lower and upper must already be ordered")
    return np.maximum(lower-y, y-upper), upper-lower


def fit_residual_quantile_intervals(
    X_train, y_train, width_covariates, *, base_alpha,
    coordinate_factors=None, model_params=None,
):
    """Fit the retained OLS-center/training-residual-width interval learner.

    width_covariates is positive (n,d).  Each multiplier is the ordinary
    empirical (1-base_alpha) quantile (NumPy's linear interpolation) of
    abs(y-OLS(X))/width_covariates, using training observations only.
    Optional positive coordinate_factors (d,) reproduce width misspecification.
    Returns the fitted center model and the (d,) width multiplier in a dict.
    """
    X_train, y_train = _training_arrays(X_train, y_train)
    widths = _matrix(width_covariates, "width_covariates")
    if widths.shape != y_train.shape or np.any(widths <= 0):
        raise ValueError("width_covariates must be positive and match y_train")
    _probability(base_alpha, "base_alpha")
    factors = np.ones(y_train.shape[1]) if coordinate_factors is None else np.asarray(
        coordinate_factors, dtype=float,
    )
    if factors.shape != (y_train.shape[1],) or not np.isfinite(factors).all() or np.any(factors <= 0):
        raise ValueError("coordinate_factors must be a positive finite (d,) array")
    center_model = LinearRegression(**dict(model_params or {})).fit(X_train, y_train)
    multiplier = np.quantile(
        np.abs(y_train-center_model.predict(X_train))/widths,
        1-base_alpha, axis=0, method="linear",
    ) * factors
    return dict(center_model=center_model, width_multiplier=multiplier,
                base_alpha=float(base_alpha), coordinate_factors=factors.copy())


def predict_residual_quantile_intervals(fitted, X, width_covariates):
    """Return center and ordered lower/upper (m,d) interval predictions."""
    X = _matrix(X, "X")
    widths = _matrix(width_covariates, "width_covariates")
    center = _targets(fitted["center_model"].predict(X), "center")
    if widths.shape != center.shape or np.any(widths <= 0):
        raise ValueError("width_covariates must be positive and match predictions")
    half_width = widths*np.asarray(fitted["width_multiplier"])
    return dict(center=center, lower=center-half_width, upper=center+half_width,
                base_lengths=2*half_width)


def quantile_diagnostics(
    y, raw_lower, raw_upper, *, lower_quantile, upper_quantile,
    oracle_lower=None, oracle_upper=None, noise_scales=None,
):
    """Evaluate supplied validation predictions without fitting or selection.

    Inputs are (m,d); probabilities are those fitted, not conformal alpha.
    Return per-tail/per-coordinate tail rates, pinball losses, crossing rate,
    ordered base-interval coverage and lengths.  Optional oracle predictions
    add RMSE and oracle pinball losses; positive (d,) noise_scales normalize
    losses and RMSE, matching the Gaussian-core diagnostic convention.
    """
    y = _targets(y)
    raw_lower, raw_upper = _matrix(raw_lower, "raw_lower"), _matrix(raw_upper, "raw_upper")
    if raw_lower.shape != y.shape or raw_upper.shape != y.shape:
        raise ValueError("y and both predictions must have matching shapes")
    _probability(lower_quantile, "lower_quantile")
    _probability(upper_quantile, "upper_quantile")
    if lower_quantile >= upper_quantile:
        raise ValueError("lower_quantile must be less than upper_quantile")
    scales = np.ones(y.shape[1]) if noise_scales is None else np.asarray(noise_scales, dtype=float)
    if scales.shape != (y.shape[1],) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError("noise_scales must be a positive finite (d,) array")
    lower, upper = np.minimum(raw_lower, raw_upper), np.maximum(raw_lower, raw_upper)
    result = dict(crossing_rate=float(np.mean(raw_lower > raw_upper)),
                  coordinate_crossing_rate=np.mean(raw_lower > raw_upper, axis=0),
                  coordinate_base_coverage=np.mean((y >= lower) & (y <= upper), axis=0),
                  coordinate_base_lengths=np.mean(upper-lower, axis=0),
                  minimum_base_length=float(np.min(upper-lower)))
    for name, q, pred, oracle in (
        ("lower", lower_quantile, raw_lower, oracle_lower),
        ("upper", upper_quantile, raw_upper, oracle_upper),
    ):
        result[name+"_tail_rate"] = np.mean(y <= pred, axis=0)
        result[name+"_pinball"] = np.array([
            mean_pinball_loss(y[:, j], pred[:, j], alpha=q)
            for j in range(y.shape[1])
        ])
        result[name+"_normalized_pinball"] = result[name+"_pinball"]/scales
        if oracle is not None:
            oracle = _matrix(oracle, "oracle_"+name)
            if oracle.shape != y.shape:
                raise ValueError("oracle predictions must match y")
            result[name+"_oracle_rmse"] = np.sqrt(np.mean((pred-oracle)**2, axis=0))
            result[name+"_normalized_oracle_rmse"] = result[name+"_oracle_rmse"]/scales
            result[name+"_oracle_pinball"] = np.array([
                mean_pinball_loss(y[:, j], oracle[:, j], alpha=q)
                for j in range(y.shape[1])
            ])
            result[name+"_normalized_oracle_pinball"] = result[name+"_oracle_pinball"]/scales
    return result


def apply_quantile_method(
    raw_scores_cal, base_lengths_cal, base_lengths_test, *, method, alpha,
    random_state=42, split_random_state=42,
    coordinate_keys=None, scalar_keys=None, test_keys=None,
    reference_dim=0, min_length=1e-12, search="backward",
    mu=None, std=None,
):
    """Apply one reviewed method and return per-test score adjustments.

    Calibration scores/lengths are (n,d), test lengths are (m,d).  Supported
    methods are Base, Unscaled, EC, Bonferroni, Naive, Oracle,
    Split_Standardized, CQHR, Signed_GWC (or GWC), and Envelope. Point CHR
    remains restricted to nonnegative residuals and is not available here.
    ``alpha`` is conformal miscoverage, independent of the learner's base alpha.
    Returned ``adjustments`` and ``upper_closed`` are (m,d), ``empty`` is (m,)
    and ``objects`` retains a native region per test row, except CQHR returns
    one CQHRResult and Base returns None.

    All native regions use the attainable test-score lower bound -length/2.
    Negative adjustments shrink intervals; cutoffs below that lower bound
    produce empty regions. Calibration scores are never clipped to a test
    observation's domain. Coordinate calibration keys (n,d) for EC/Bonferroni
    or scalar keys (n,) for other methods are shared across tests. Independent
    test keys have shape (m,d) for EC/Bonferroni and (m,) for other methods.
    Missing keys are drawn from a caller-controlled NumPy Generator/seed and
    explicitly passed to the methods.  An exact unresolved key collision raises
    through the method interface; supply fresh keys rather than perturb scores.

    Split_Standardized uses the separate integer split_random_state in
    [0, 2**32), reusing the same split for every test row; None, booleans and
    mutable RNG objects are rejected for that parameter. Oracle requires
    independently supplied mu/std. ``reference_dim`` and ``min_length`` are
    CQHR-only options. Naive remains a heuristic, and EC retains its empirical
    coverage construction. No calibration algorithm is copied here.
    """
    scores = _matrix(raw_scores_cal, "raw_scores_cal")
    lengths_cal = _matrix(base_lengths_cal, "base_lengths_cal")
    lengths_test = _matrix(base_lengths_test, "base_lengths_test")
    if scores.shape != lengths_cal.shape or scores.shape[1] != lengths_test.shape[1]:
        raise ValueError("Calibration arrays must match and test target dimensions must agree")
    if np.any(lengths_cal < 0) or np.any(lengths_test < 0):
        raise ValueError("Base interval lengths must be nonnegative")
    _probability(alpha, "alpha")
    name = str(method).lower().replace("-", "_").replace(" ", "_")
    name = {"signed_gwc": "gwc", "envelope_signed": "envelope"}.get(name, name)
    functions = {
        "unscaled": methods.unscaled_prediction,
        "ec": methods.empirical_copula_prediction,
        "bonferroni": methods.bonferroni_prediction,
        "naive": methods.naive_prediction,
        "oracle": methods.data_splitting_oracle_prediction,
        "split_standardized": methods.data_splitting_standardized_prediction,
        "gwc": methods.gwc_prediction,
    }
    if name == "point_chr":
        raise ValueError("Point CHR requires nonnegative residuals; use apply_absolute_method")
    if name not in functions and name not in ("base", "cqhr", "envelope"):
        raise ValueError("Supported quantile methods are Base, Unscaled, EC, Bonferroni, "
                         "Naive, Oracle, Split_Standardized, CQHR, Signed_GWC and Envelope")
    if name == "split_standardized" and (
        isinstance(split_random_state, (bool, np.bool_))
        or not isinstance(split_random_state, (int, np.integer))
        or not 0 <= split_random_state < 2**32
    ):
        raise ValueError("split_random_state must be an integer in [0, 2**32)")
    if name == "oracle" and (mu is None or std is None):
        raise ValueError("oracle requires independently supplied mu and std")
    m, d = lengths_test.shape
    if name == "base":
        return dict(method=name, adjustments=np.zeros((m, d)),
                    upper_closed=np.ones((m, d), dtype=bool),
                    empty=np.zeros(m, dtype=bool), objects=None,
                    base_lengths_test=lengths_test.copy(),
                    coordinate_keys=None, scalar_keys=None, test_keys=None)
    rng = np.random.default_rng(random_state)
    marginal = name in ("ec", "bonferroni")
    calibration_shape = scores.shape if marginal else (len(scores),)
    supplied_keys = coordinate_keys if marginal else scalar_keys
    keys_cal = (rng.random(calibration_shape) if supplied_keys is None
                else np.asarray(supplied_keys, dtype=float))
    test_shape = (m, d) if marginal else (m,)
    keys_test = rng.random(test_shape) if test_keys is None else np.asarray(test_keys, dtype=float)
    if keys_cal.shape != calibration_shape or keys_test.shape != test_shape:
        key_name = "coordinate_keys" if marginal else "scalar_keys"
        raise ValueError(f"{key_name} and test_keys must have shapes "
                         f"{calibration_shape} and {test_shape}")
    if name == "cqhr":
        objects = methods.cqhr_adjustments(
            scores, lengths_cal, lengths_test, alpha=alpha,
            reference_dim=reference_dim, min_length=min_length,
            random_state=rng, scalar_keys=keys_cal, test_keys=keys_test,
        )
        adjustments = objects.adjustments.copy()
        closed = np.broadcast_to(objects.upper_closed[:, None], (m, d)).copy()
        attainable = -lengths_test/2
        empty = np.any((adjustments < attainable) |
                       ((adjustments == attainable) & ~closed), axis=1)
    else:
        calibration = methods.EnvelopeCalibration(
            scores, alpha=alpha, random_state=rng, scalar_keys=keys_cal,
        ) if name == "envelope" else None
        kwargs = dict(alpha=alpha,
                      random_state=split_random_state if name == "split_standardized" else rng)
        kwargs["coordinate_keys" if marginal else "scalar_keys"] = keys_cal
        if name == "oracle":
            kwargs.update(mu=mu, std=std)
        objects = []
        for length, key in zip(lengths_test, keys_test):
            if name == "envelope":
                region = calibration.predict(-length/2, search=search, test_key=key)
            else:
                key_kw = {"test_keys": key} if marginal else {"test_key": key}
                region = functions[name](scores, lower=-length/2, **kwargs, **key_kw)
            objects.append(region)
        adjustments = np.stack([region.upper for region in objects])
        closed = np.stack([region.upper_closed for region in objects])
        empty = np.array([region.empty for region in objects], dtype=bool)
    return dict(method=name, adjustments=adjustments, upper_closed=closed,
                empty=empty, objects=objects,
                coordinate_keys=keys_cal.copy() if marginal else None,
                scalar_keys=None if marginal else keys_cal.copy(),
                test_keys=keys_test.copy(), base_lengths_test=lengths_test.copy())


def evaluate_quantile_predictions(
    prediction, raw_scores_test, base_lengths_test=None, *,
    base_lower=None, base_upper=None,
):
    """Return outcome-space metrics using the reviewed methods' membership.

    raw_scores_test and base_lengths_test are (m,d).  Lengths default to those
    recorded by apply_quantile_method and must match them if supplied.  The
    scores must come from the same actual base intervals (their attainable
    coordinate minima are -base_lengths_test/2); no rounding repair is applied.
    Outcome side lengths are base_lengths+2*adjustments.  Empty regions have
    zero lengths, volume and coverage; positive-volume logarithms use base 10.
    Joint membership calls CQHRResult.contains or each score region's contains.

    Optional ordered base_lower/base_upper (m,d) additionally return inverted
    ``outcome_lower=base_lower-adjustments`` and
    ``outcome_upper=base_upper+adjustments``.  Consult the returned empty mask
    and upper_closed flags: numeric endpoints alone omit randomized boundaries.
    """
    scores = _matrix(raw_scores_test, "raw_scores_test")
    lengths = prediction["base_lengths_test"] if base_lengths_test is None else _matrix(
        base_lengths_test, "base_lengths_test",
    )
    if scores.shape != prediction["adjustments"].shape or lengths.shape != scores.shape:
        raise ValueError("Test scores, lengths and adjustments must have matching shapes")
    if not np.array_equal(lengths, prediction["base_lengths_test"]):
        raise ValueError("Test lengths must match those used to apply the method")
    adjustments, closed = prediction["adjustments"], prediction["upper_closed"]
    empty = prediction["empty"].copy()
    outcome_lengths = lengths+2*adjustments
    empty |= np.any(outcome_lengths < 0, axis=1)
    name, objects = prediction["method"], prediction["objects"]
    if name == "cqhr":
        joint = objects.contains(scores)
        # Hold other coordinates at their attainable minima.  For a nonempty
        # rectangle these candidates isolate each coordinate while retaining
        # CQHR's exact score conversion and secondary-key comparison.
        coordinate = np.empty(scores.shape, dtype=bool)
        for j in range(scores.shape[1]):
            candidates = -lengths/2
            candidates[:, j] = scores[:, j]
            coordinate[:, j] = objects.contains(candidates)
    elif name != "base":
        joint = np.array([region.contains(row) for region, row in zip(objects, scores)])
        coordinate = np.stack([
            (row >= region.lower) & ((row < region.upper) |
                                     ((row == region.upper) & region.upper_closed))
            for region, row in zip(objects, scores)
        ])
    else:
        coordinate = scores <= 0
        joint = coordinate.all(axis=1)
    result = summarize_predictions(
        coordinate, np.maximum(outcome_lengths, 0), empty,
        adjustments=adjustments, joint_covered=joint,
    )
    result["upper_closed"] = closed.copy()
    if (base_lower is None) != (base_upper is None):
        raise ValueError("Supply both base_lower and base_upper, or neither")
    if base_lower is not None:
        base_lower, base_upper = _matrix(base_lower, "base_lower"), _matrix(base_upper, "base_upper")
        if base_lower.shape != scores.shape or base_upper.shape != scores.shape:
            raise ValueError("Base bounds must match the test scores")
        if np.any(base_lower > base_upper) or not np.array_equal(base_upper-base_lower, lengths):
            raise ValueError("Base bounds must be ordered and have the supplied lengths")
        result["outcome_lower"] = base_lower-adjustments
        result["outcome_upper"] = base_upper+adjustments
        result["outcome_lower_closed"] = closed.copy()
        result["outcome_upper_closed"] = closed.copy()
    return result

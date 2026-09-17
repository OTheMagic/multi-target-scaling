"""Data generators for the retained synthetic regression designs.

Only NumPy and the standard library are used. No data are generated at import.
The four ``make_*`` functions preserve the draw order and distributions of
their source helpers. In particular, their coefficients have shape (d, p),
whereas the two ``draw_*_regression`` functions use matrix coefficients (p, d).
"""

import hashlib
import json
from numbers import Integral

import numpy as np


__all__ = [
    "make_multitarget_regression",
    "make_multitarget_regression_dependent_noise",
    "make_partial_heteroskedastic_regression",
    "make_contaminated_regression",
    "draw_matched_regression",
    "draw_fitted_interval_stress_regression",
    "draw_regression_splits",
]


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _regression_inputs(n_samples, n_features, n_targets, n_informative,
                       noise_list, coef):
    for value, name in ((n_samples, "n_samples"), (n_features, "n_features"),
                        (n_targets, "n_targets")):
        _positive_int(value, name)
    if (isinstance(n_informative, bool)
            or not isinstance(n_informative, Integral)
            or not 0 <= n_informative <= n_features):
        raise ValueError("n_informative must be an integer from 0 to n_features.")
    scales = np.ones(n_targets) if noise_list is None else np.asarray(noise_list, float)
    if scales.shape != (n_targets,) or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("noise_list must contain one finite nonnegative scale per target.")
    if coef is not None:
        coefficients = np.asarray(coef)
        if coefficients.shape != (n_targets, n_features) or not np.isfinite(coefficients).all():
            raise ValueError("coef must be a finite (n_targets, n_features) array.")
    return scales


def _design_and_coefficients(rng, n_samples, n_features, n_targets,
                             n_informative, coef):
    # Yielding is intentional: the independent generator interleaves each
    # coordinate's coefficient draw and noise draw, exactly as its source does.
    x = rng.standard_normal(size=(n_samples, n_features))
    coef_list = [] if coef is None else None
    for j in range(n_targets):
        if coef is None:
            coef_j = np.zeros(n_features)
            idx = rng.choice(n_features, size=n_informative, replace=False)
            coef_j[idx] = rng.uniform(-10, 10, size=n_informative)
            coef_list.append(coef_j)
        else:
            coef_j = np.asarray(coef[j])
        yield x, coef_j, coef_list


def _independent_noise(rng, noise_type, scale, n_samples, j, d, df):
    if noise_type == "gaussian":
        return rng.normal(scale=scale, size=n_samples)
    if noise_type == "laplace":
        return rng.laplace(scale=scale, size=n_samples)
    if noise_type == "gamma":
        return rng.gamma(shape=j, scale=scale, size=n_samples)
    if noise_type == "mixed":
        if j < d // 2:
            return rng.normal(scale=scale, size=n_samples)
        return rng.laplace(scale=scale, size=n_samples)
    if noise_type == "cauchy":
        return rng.standard_cauchy(size=n_samples)
    if noise_type == "t":
        if df is None:
            raise ValueError("df must be provided when noise_type='t'.")
        return rng.standard_t(df=df, size=n_samples)
    raise ValueError(f"Unknown noise_type: {noise_type}")


def make_multitarget_regression(
    n_samples=100, n_features=10, n_targets=3, n_informative=10,
    noise_type="Gaussian", noise_list=None, df=None, coef=None, random_state=42,
):
    """Draw independent-coordinate linear regression data.

    Extracted from ``utility/data_generator.py:make_multitarget_regression``.
    X is standard normal (n,p); y is (n,d). When coef is None, each target
    selects n_informative positions without replacement and draws coefficients
    uniformly on [-10,10]; return (X,y,coefs), with coefs a list of d (p,)
    arrays. With supplied coef (d,p), return (X,y). Coefficients are unnormalized.

    noise_type is Gaussian, Laplace, Gamma, mixed (first d//2 Gaussian, rest
    Laplace), Cauchy, or t. noise_list defaults to ones. Preserved source quirks:
    Cauchy/t ignore noise_list scales, and Gamma shape is the zero-based target
    index, so its first noise coordinate is zero. t requires df.

    random_state accepts a NumPy seed, Generator, or None. Integer-seed draw
    order is preserved, including dependence of fresh coefficients on n_samples.
    """
    scales = _regression_inputs(n_samples, n_features, n_targets, n_informative,
                                noise_list, coef)
    rng = np.random.default_rng(random_state)
    y = np.zeros((n_samples, n_targets))
    for j, (x, coef_j, coef_list) in enumerate(_design_and_coefficients(
            rng, n_samples, n_features, n_targets, n_informative, coef)):
        y[:, j] = x @ coef_j + _independent_noise(
            rng, noise_type.lower(), scales[j], n_samples, j, n_targets, df)
    return (x, y, coef_list) if coef is None else (x, y)


def make_multitarget_regression_dependent_noise(
    n_samples=100, n_features=10, n_targets=3, n_informative=10,
    noise_type="Gaussian", noise_list=None, df=None, coef=None, random_state=42,
    correlation=0.5, correlation_structure="equicorrelated",
):
    """Draw linear regression with Gaussian noise covariance diag(scales) R diag(scales).

    Source: ``utility/data_generator.py:make_multitarget_regression_dependent_noise``.
    Input/output and coefficient contracts match make_multitarget_regression.
    R is equicorrelated or AR(1). noise_type must be Gaussian; df is unused and
    must be None. Preserve the source draw order: joint noise precedes X and
    coefficients. random_state accepts a seed, Generator, or None.
    """
    if noise_type.lower() != "gaussian" or df is not None:
        raise ValueError("Dependent noise requires Gaussian noise_type and df=None.")
    scales = _regression_inputs(n_samples, n_features, n_targets, n_informative,
                                noise_list, coef)
    structure = correlation_structure.lower()
    if structure == "equicorrelated":
        lower = -1 / (n_targets - 1) if n_targets > 1 else -1
        if not lower < correlation < 1:
            raise ValueError(f"Equicorrelation must lie strictly between {lower} and 1.")
        corr = np.full((n_targets, n_targets), correlation)
        np.fill_diagonal(corr, 1.0)
    elif structure == "ar1":
        if not -1 < correlation < 1:
            raise ValueError("AR(1) correlation must lie strictly between -1 and 1.")
        idx = np.arange(n_targets)
        corr = correlation ** np.abs(idx[:, None] - idx[None, :])
    else:
        raise ValueError("correlation_structure must be equicorrelated or ar1.")
    rng = np.random.default_rng(random_state)
    noise = rng.multivariate_normal(np.zeros(n_targets), np.outer(scales, scales) * corr,
                                   size=n_samples)
    y = np.zeros((n_samples, n_targets))
    for j, (x, coef_j, coef_list) in enumerate(_design_and_coefficients(
            rng, n_samples, n_features, n_targets, n_informative, coef)):
        y[:, j] = x @ coef_j + noise[:, j]
    return (x, y, coef_list) if coef is None else (x, y)


def _linear_signal(rng, n_samples, n_features, n_targets, n_informative, coef):
    # These DGPs draw all coefficients before noise, unlike the independent DGP.
    columns = []
    for x, coef_j, coef_list in _design_and_coefficients(
            rng, n_samples, n_features, n_targets, n_informative, coef):
        columns.append(x @ np.asarray(coef_j, dtype=float))
    return x, np.column_stack(columns), coef_list


def make_partial_heteroskedastic_regression(
    n_samples=100, n_features=10, n_targets=10, n_informative=10,
    noise_type="Gaussian", noise_list=None, coef=None, random_state=42,
    heteroskedastic_fraction=0.5, heteroskedastic_strength=1.5,
):
    """Draw Gaussian linear data with input-dependent variance in selected targets.

    Source: ``reviewer_update/build_experiment_update.py``'s same-named pure
    helper and _linear_signal. The first round(d*fraction) noise coordinates
    are multiplied by sqrt((1+strength*X[:,0]**2)/(1+strength)). noise_list
    defaults to [d,...,1]. Return (X,y,coefs) if coef is None, otherwise (X,y);
    X:(n,p), y:(n,d), supplied coef:(d,p). Draw order is preserved.
    """
    if noise_type.lower() != "gaussian":
        raise ValueError("Partial heteroskedasticity requires Gaussian noise.")
    if not np.isfinite(heteroskedastic_fraction) or not 0 <= heteroskedastic_fraction <= 1:
        raise ValueError("heteroskedastic_fraction must lie in [0,1].")
    if not np.isfinite(heteroskedastic_strength) or heteroskedastic_strength < 0:
        raise ValueError("heteroskedastic_strength must be finite and nonnegative.")
    if noise_list is None:
        noise_list = np.arange(n_targets, 0, -1, dtype=float)
    scales = _regression_inputs(n_samples, n_features, n_targets, n_informative,
                                noise_list, coef)
    rng = np.random.default_rng(random_state)
    x, signal, coef_list = _linear_signal(
        rng, n_samples, n_features, n_targets, n_informative, coef)
    noise = rng.standard_normal(size=(n_samples, n_targets)) * scales
    n_hetero = int(round(n_targets * heteroskedastic_fraction))
    factor = np.sqrt((1 + heteroskedastic_strength * x[:, [0]] ** 2)
                     / (1 + heteroskedastic_strength))
    noise[:, :n_hetero] *= factor
    y = signal + noise
    return (x, y, coef_list) if coef is None else (x, y)


def make_contaminated_regression(
    n_samples=100, n_features=10, n_targets=10, n_informative=10,
    noise_type="Gaussian", noise_list=None, coef=None, random_state=42,
    contamination_fraction=0.0, contamination_multiplier=10.0,
):
    """Draw Gaussian linear data with exchangeable row-wise scale contamination.

    Source: ``reviewer_update/build_experiment_update.py``'s same-named pure
    helper and _linear_signal. A Bernoulli(contamination_fraction) draw selects
    each row; its entire noise vector is scaled by contamination_multiplier.
    noise_list defaults to [d,...,1]. Shapes, returns, and coefficient convention
    match make_partial_heteroskedastic_regression. Draw order is preserved.
    """
    if noise_type.lower() != "gaussian":
        raise ValueError("Contamination requires Gaussian noise.")
    if not np.isfinite(contamination_fraction) or not 0 <= contamination_fraction <= 1:
        raise ValueError("contamination_fraction must lie in [0,1].")
    if not np.isfinite(contamination_multiplier) or contamination_multiplier < 0:
        raise ValueError("contamination_multiplier must be finite and nonnegative.")
    if noise_list is None:
        noise_list = np.arange(n_targets, 0, -1, dtype=float)
    scales = _regression_inputs(n_samples, n_features, n_targets, n_informative,
                                noise_list, coef)
    rng = np.random.default_rng(random_state)
    x, signal, coef_list = _linear_signal(
        rng, n_samples, n_features, n_targets, n_informative, coef)
    multiplier = np.where(rng.random(n_samples) < contamination_fraction,
                          contamination_multiplier, 1.0)
    noise = rng.standard_normal(size=(n_samples, n_targets)) * scales * multiplier[:, None]
    y = signal + noise
    return (x, y, coef_list) if coef is None else (x, y)


def _matrix_coefficients(n_samples, coef):
    _positive_int(n_samples, "n_samples")
    coef = np.asarray(coef, dtype=float)
    if coef.ndim != 2 or min(coef.shape) < 1 or not np.isfinite(coef).all():
        raise ValueError("coef must be a finite, nonempty (n_features,n_targets) array.")
    return coef


def _named_seed(*parts):
    payload = json.dumps(parts, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def draw_matched_regression(n_samples, coef, *, noise_design="gaussian", random_state=42):
    """Return (X,y) with matched standard Gaussian or Gaussian/uniform target noise.

    Source: ``envelope_method/cqr10/matched_comparison.py:_draw``. coef is (p,d),
    X is (n,p), y is (n,d). noise_design is 'gaussian' or 'normal_uniform'; the
    latter uses Uniform(-sqrt(3),sqrt(3)) in odd-indexed target coordinates.
    X and each noise coordinate use independent named hash-derived streams.
    The source coefficient choice is arange(1,p*d+1).reshape(p,d)/(p*d), supplied
    explicitly by the caller here.

    Integer random_state preserves the source's named streams. A Generator
    (or None) first supplies one integer root seed, a new convenience behavior;
    it does not reproduce the legacy integer-seed schedule automatically.
    """
    coef = _matrix_coefficients(n_samples, coef)
    if noise_design not in ("gaussian", "normal_uniform"):
        raise ValueError("noise_design must be gaussian or normal_uniform.")
    if isinstance(random_state, Integral) and not isinstance(random_state, bool):
        seed = int(random_state)
        if seed < 0:
            raise ValueError("random_state must be nonnegative.")
    else:
        seed = int(np.random.default_rng(random_state).integers(0, 2**63))
    p, d = coef.shape
    x = np.random.default_rng(_named_seed(seed, "features")).normal(size=(n_samples, p))
    noise = np.empty((n_samples, d))
    for j in range(d):
        rng = np.random.default_rng(_named_seed(seed, "noise", j))
        if noise_design == "normal_uniform" and j % 2:
            noise[:, j] = rng.uniform(-np.sqrt(3), np.sqrt(3), size=n_samples)
        else:
            noise[:, j] = rng.normal(size=n_samples)
    return x, x @ coef + noise


def draw_fitted_interval_stress_regression(
    n_samples, coef, *, noise_scales, design="gaussian_matched", random_state=42,
    width_log_sd=0.9, tail_probability=0.1, nonlinear_strength=2.0,
):
    """Return (X,y,width_covariate) for the retained fitted-interval stress DGPs.

    Source: ``envelope_method/cqr10/pilot_misspecification.py:draw_split``.
    coef is (p,d); outputs have shapes (n,p), (n,d), (n,d). The source fixed
    coefficient choice is arange(1,p*d+1).reshape(p,d)/(p*d), supplied explicitly.
    noise_scales is (d,). Integer seeds and source draw order are preserved.

    gaussian_matched and coordinate_width_misspecified share Gaussian data and
    unit width covariates; the latter's width distortion belongs to the learner.
    nonlinear_mean_misspecified adds strength*(X[:,0]**2-1)*noise_scales to the
    mean. covariate_tail_mismatch draws independent unit-mean lognormal widths,
    an inside/outside Bernoulli indicator, uniform inside magnitudes or width
    plus exponential outside magnitudes, and independent random signs. The
    returned observed widths are available to the interval learner.
    """
    coef = _matrix_coefficients(n_samples, coef)
    p, d = coef.shape
    scales = np.asarray(noise_scales, dtype=float)
    if scales.shape != (d,) or not np.isfinite(scales).all() or np.any(scales < 0):
        raise ValueError("noise_scales must be finite nonnegative values with shape (d,).")
    if design not in ("gaussian_matched", "coordinate_width_misspecified",
                      "covariate_tail_mismatch", "nonlinear_mean_misspecified"):
        raise ValueError("Unknown fitted-interval stress design.")
    if not np.isfinite(width_log_sd) or width_log_sd < 0:
        raise ValueError("width_log_sd must be finite and nonnegative.")
    if not np.isfinite(tail_probability) or not 0 <= tail_probability <= 1:
        raise ValueError("tail_probability must lie in [0,1].")
    if not np.isfinite(nonlinear_strength):
        raise ValueError("nonlinear_strength must be finite.")
    rng = np.random.default_rng(random_state)
    x = rng.normal(size=(n_samples, p))
    mean = x @ coef
    if design == "nonlinear_mean_misspecified":
        mean += nonlinear_strength * (x[:, [0]] ** 2 - 1) * scales
    if design == "covariate_tail_mismatch":
        widths = np.exp(width_log_sd * rng.normal(size=(n_samples, d))
                        - 0.5 * width_log_sd * width_log_sd)
        outside = rng.random((n_samples, d)) < tail_probability
        magnitude = np.where(
            outside, widths + rng.exponential(scale=scales, size=(n_samples, d)),
            widths * rng.random((n_samples, d)))
        noise = magnitude * rng.choice([-1.0, 1.0], size=(n_samples, d))
    else:
        widths = np.ones((n_samples, d))
        noise = rng.normal(size=(n_samples, d)) * scales
    return x, mean + noise, widths


def draw_regression_splits(
    generator, *, coef, n_train, n_cal, n_test, train_random_state,
    calibration_random_state, test_random_state, calibration_pool_size=None,
    n_validation=None, validation_random_state=None, generator_kwargs=None,
):
    """Draw one cohort using fixed coefficients and caller-controlled split streams.

    Extracts the independent-draw/calibration-prefix computation in
    ``synthetic_absolute.py:draw_data`` and ``cqr10/matched_comparison.py:draw_data``.
    generator accepts n_samples, coef, random_state plus generator_kwargs; it
    returns (X,y), or (X,y,width_covariate) for the stress generator. coef uses
    that generator's documented orientation. No model is fitted.

    Return a dict containing dgp_coef, X_train/y_train, X_cal/y_cal, X_test/y_test,
    optionally X_validation/y_validation, and width_covariate_<split> when
    supplied by the generator. Row counts follow n_<split>; X is (n,p), y and
    widths are (n,d). Calibration is copied from the first n_cal rows of a pool
    of calibration_pool_size (default n_cal), so an unchanged pool and seed
    give nested calibration prefixes. Feature/noise stream consumption can
    depend on draw size; a common seed alone does not guarantee nested draws.

    Supply independent seeds or independent Generators for each split. Shared
    Generator objects are consumed in train, validation (if present), cal, test
    order; independence is the caller's responsibility. No legacy trial hashing
    is performed and fixed integer seeds are repeatable. n_validation=None
    omits validation; None as a random_state delegates entropy to NumPy.
    """
    if not callable(generator):
        raise TypeError("generator must be callable.")
    if coef is None:
        raise ValueError("Supply fixed coefficients shared across all splits.")
    sizes = {"train": _positive_int(n_train, "n_train"),
             "cal": _positive_int(n_cal, "n_cal"),
             "test": _positive_int(n_test, "n_test")}
    pool_size = sizes["cal"] if calibration_pool_size is None else _positive_int(
        calibration_pool_size, "calibration_pool_size")
    if pool_size < sizes["cal"]:
        raise ValueError("calibration_pool_size must be at least n_cal.")
    kwargs = dict(generator_kwargs or {})
    if {"n_samples", "coef", "random_state"}.intersection(kwargs):
        raise ValueError("generator_kwargs cannot override n_samples, coef, or random_state.")
    streams = [("train", train_random_state)]
    if n_validation is not None:
        sizes["validation"] = _positive_int(n_validation, "n_validation")
        streams.append(("validation", validation_random_state))
    streams.extend((("cal", calibration_random_state), ("test", test_random_state)))
    arrays = {"dgp_coef": np.array(coef, copy=True)}
    expected_shapes = None
    has_widths = None
    for split, seed in streams:
        count = pool_size if split == "cal" else sizes[split]
        result = generator(n_samples=count, coef=coef, random_state=seed, **kwargs)
        if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
            raise ValueError("generator must return (X,y) or (X,y,width_covariate).")
        x, y = np.asarray(result[0]), np.asarray(result[1])
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != count or y.shape[0] != count:
            raise ValueError("generator must return two-dimensional arrays with n_samples rows.")
        shape = (x.shape[1], y.shape[1])
        if min(shape) < 1 or (expected_shapes is not None and expected_shapes != shape):
            raise ValueError("Feature/target dimensions must be positive and consistent across splits.")
        if has_widths is not None and has_widths != (len(result) == 3):
            raise ValueError("The generator must return width covariates consistently across splits.")
        expected_shapes, has_widths = shape, len(result) == 3
        take = sizes[split]
        arrays["X_" + split], arrays["y_" + split] = x[:take].copy(), y[:take].copy()
        if has_widths:
            widths = np.asarray(result[2])
            if widths.shape != y.shape:
                raise ValueError("width_covariate must have the same (n,d) shape as y.")
            arrays["width_covariate_" + split] = widths[:take].copy()
    return arrays

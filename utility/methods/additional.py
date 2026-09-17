"""Additional baselines extracted from the two original baseline modules.

All inputs ``scores`` are finite, possibly signed residuals of shape (n, d),
with n and d positive. Outputs are ``Rectangle`` objects on the score domain
specified by ``lower``. Use their membership methods to honor randomized upper
boundaries. A region holds its test key fixed across candidate outcomes;
independent test observations need independent test keys.

Optional keys lie in [0, 1). Omitted keys are drawn from one local Generator
seeded by ``random_state`` (default 42), calibration keys before test keys.
Coverage arguments require independent uniform calibration and test keys,
including when supplied explicitly; arbitrary supplied values are for replay
or deterministic checks and do not by themselves justify coverage claims.
Keys replace the legacy additive jitter without perturbing numeric scores,
means, or scales. Exact score/key collisions raise ValueError. The shared
quantile helpers use the exact decimal/rational conformal-rank convention.

Each predictor accepts keyword-only ``lower``: a scalar or (d,) vector, finite
or -inf, giving the attainable lower score bound for one test observation.
The default zero is appropriate for absolute residuals. For signed CQR scores
use ``lower=-test_interval_widths/2``; use -inf for scores with no finite lower
bound. Calibration scores are not clipped or checked against this test domain.
An upper bound below lower gives an empty rectangle. Rectangle volume measures
score space, not the corresponding outcome intervals.
"""

from fractions import Fraction

import numpy as np
from sklearn.model_selection import train_test_split

from ..rectangle import Rectangle
from ..tie_breaking import (
    conformal_quantile_pair, coordinate_quantile_pairs, draw_keys, lex_le,
)


def _scores_array(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("scores must have shape (n, d) with n and d positive")
    if np.any(~np.isfinite(scores)):
        raise ValueError("scores must be finite")
    return scores


def _standardization_parameters(mean, scale, d):
    """Broadcast scalar or coordinate parameters; reject undefined inverses."""
    try:
        mean = np.broadcast_to(np.asarray(mean, dtype=float), (d,))
        scale = np.broadcast_to(np.asarray(scale, dtype=float), (d,))
    except ValueError as exc:
        raise ValueError("mean and scale must be scalars or have shape (d,)") from exc
    if np.any(~np.isfinite(mean)):
        raise ValueError("mean must be finite")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale must be finite and strictly positive in every coordinate")
    return mean, scale


def _standardized_rectangle(scores, mean, scale, alpha, scalar_keys, test_key,
                            lower):
    """Shared max-score calibration and positive affine inverse."""
    mean, scale = _standardization_parameters(mean, scale, scores.shape[1])
    standardized_scores = (scores - mean) / scale
    max_scores = np.max(standardized_scores, axis=1)
    quantile, cutoff_key = conformal_quantile_pair(max_scores, scalar_keys, alpha)
    upper = quantile * scale + mean
    upper_closed = bool(lex_le(quantile, test_key, quantile, cutoff_key))
    return Rectangle(upper=upper, lower=lower, upper_closed=upper_closed,
                     allow_empty=True)


def bonferroni_prediction(scores, alpha=0.2, random_state=42, *,
                          lower=0.0, coordinate_keys=None, test_keys=None):
    """Calibrate each residual coordinate at miscoverage alpha/d.

    Source: ``utility/unscaled.py::bonferroni_prediction``; project definition:
    ``supplementary.tex``, algorithm ``alg:bonferroni``. For d coordinates,
    each marginal cutoff uses rank ceil((n + 1) * (1 - alpha/d)); an excessive
    rank gives +inf. The alpha/d division is exact before rank calculation.
    With exchangeable calibration/test residuals and independent uniform tie
    keys, the marginal conformal bounds and union bound give simultaneous
    coverage at least 1-alpha.

    ``coordinate_keys`` has shape (n, d), and ``test_keys`` has shape (d,).
    Equal raw residuals are ranked with their coordinate keys. Each returned
    upper face is included when its test key precedes that cutoff's key.
    There is no scalar-score calibration stage. ``alpha`` is in (0, 1).
    ``lower`` specifies the test score domain as described in the module
    docstring; signed scores and empty returned rectangles are supported.
    """
    scores = _scores_array(scores)
    _, d = scores.shape
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    alpha_corrected = Fraction(str(alpha)) / d
    rng = np.random.default_rng(random_state)
    coordinate_keys = draw_keys(scores.shape, rng, coordinate_keys)
    test_keys = draw_keys((d,), rng, test_keys)
    upper, cutoff_keys = coordinate_quantile_pairs(
        scores, coordinate_keys, alpha_corrected,
    )
    upper_closed = lex_le(upper, test_keys, upper, cutoff_keys)
    return Rectangle(upper=upper, lower=lower, upper_closed=upper_closed,
                     allow_empty=True)


def naive_prediction(scores, alpha=0.2, random_state=42, *,
                     lower=0.0, scalar_keys=None, test_key=None):
    """Fit and calibrate standardized max scores on the same observations.

    Source: ``utility/data_splitting.py::naive_prediction``; project definition:
    ``supplementary.tex``, algorithm ``alg:std-plug-in``. Compute coordinate
    means and sample standard deviations (ddof=1), calibrate the row maxima
    of (scores - mean)/scale, and invert to upper = quantile*scale + mean.
    This remains a heuristic: fitting the transformation on the same rows
    used for calibration does not supply the usual split-conformal guarantee.

    ``scalar_keys`` has shape (n,); ``test_key`` is scalar. Keys order final
    scalar scores only; raw coordinates enter arithmetic, not rank operations.
    At least two rows and positive finite sample scales are required. These
    conditions now raise ValueError explicitly rather than dividing by zero
    or propagating NaNs from the legacy implementation. ``alpha`` is in (0, 1).
    ``lower`` specifies the test score domain as described in the module
    docstring; signed scores and empty returned rectangles are supported.
    """
    scores = _scores_array(scores)
    if len(scores) < 2:
        raise ValueError("naive standardization requires at least two rows")
    scale = np.std(scores, axis=0, ddof=1)
    mean = np.mean(scores, axis=0)
    rng = np.random.default_rng(random_state)
    scalar_keys = draw_keys((len(scores),), rng, scalar_keys)
    test_key = draw_keys((), rng, test_key)
    return _standardized_rectangle(scores, mean, scale, alpha, scalar_keys,
                                    test_key, lower)


def data_splitting_oracle_prediction(scores, mu, std, alpha=0.2,
                                     random_state=42, *, lower=0.0, scalar_keys=None,
                                     test_key=None):
    """Calibrate standardized max scores with fixed oracle mean and scale.

    Source: ``utility/data_splitting.py::data_splitting_oracle_prediction``.
    Project definition: ``body.tex``, section ``sec:pop-oracle`` and equations
    ``eq:residual-transformation`` / ``eq:set-valued-map``.
    Despite its historical name, this method does not split the input. All
    n rows calibrate max_j((scores_j - mu_j)/std_j); the upper bounds are
    quantile*std + mu. The oracle parameters must be fixed independently of
    calibration/test residuals for the usual conformal argument to apply.

    ``mu`` and ``std`` can each be scalar or shape (d,); mu must be finite
    and std finite and strictly positive. Scalar parameters are now explicitly
    broadcast to d coordinates, correcting the legacy scalar-output error.
    Invalid scales now raise ValueError instead of producing invalid scores.
    ``scalar_keys`` has shape (n,), ``test_key`` is scalar, and ``alpha`` is in
    (0, 1). Only final scalar scores are ranked; raw coordinates are unchanged.
    ``lower`` specifies the test score domain as described in the module
    docstring; signed scores and empty returned rectangles are supported.
    """
    scores = _scores_array(scores)
    rng = np.random.default_rng(random_state)
    scalar_keys = draw_keys((len(scores),), rng, scalar_keys)
    test_key = draw_keys((), rng, test_key)
    return _standardized_rectangle(scores, mu, std, alpha, scalar_keys,
                                    test_key, lower)


def data_splitting_standardized_prediction(scores, alpha=0.2, random_state=42,
                                           *, lower=0.0, scalar_keys=None,
                                           test_key=None):
    """Fit standardization on one half, calibrate max scores on the other.

    Source: ``utility/data_splitting.py::data_splitting_standardized_prediction``.
    Project definition: ``supplementary.tex``, algorithm ``alg:std-ds``.
    Preserve train_test_split(test_size=0.5, random_state=random_state): the
    first floor(n/2) rows in the returned split fit coordinate means and sample
    standard deviations (ddof=1); the remaining ceil(n/2) calibrate the maxima
    of standardized residuals. Return upper = quantile*scale + mean.

    ``scalar_keys`` has shape (n,) in original input order. Its entries follow
    rows into the calibration split; fit-half keys are unused. ``test_key`` is
    scalar and controls all upper-face inclusions. Raw coordinates enter
    fitting arithmetic, not ranking. ``alpha`` is in (0, 1).

    At least four rows (two in the fitting half) and positive finite fitted
    scales are required. Undefined scales now raise ValueError explicitly;
    no variance floor or replacement scale is introduced.
    ``lower`` specifies the test score domain as described in the module
    docstring; signed scores and empty returned rectangles are supported.
    """
    scores = _scores_array(scores)
    if len(scores) < 4:
        raise ValueError("split standardization requires at least four rows")
    first, second = train_test_split(
        np.arange(len(scores)), test_size=0.5, random_state=random_state,
    )
    scale = np.std(scores[first], axis=0, ddof=1)
    mean = np.mean(scores[first], axis=0)
    rng = np.random.default_rng(random_state)
    scalar_keys = draw_keys((len(scores),), rng, scalar_keys)
    test_key = draw_keys((), rng, test_key)
    return _standardized_rectangle(
        scores[second], mean, scale, alpha, scalar_keys[second], test_key, lower,
    )

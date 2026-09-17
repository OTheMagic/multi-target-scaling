"""CQHR extracted from ``utility/cqhr.py``, with random scalar tie keys.

The input validation, reference coordinate, width floor, and signed-score
conformal adjustment are preserved. Secondary keys never perturb these values.
Verified against Sampson and Chan, Algorithm 2, for lengths unaffected by
the existing numerical floor: https://arxiv.org/pdf/2406.04498.
"""

from dataclasses import dataclass, field

import numpy as np

from ..tie_breaking import conformal_quantile_pair, draw_keys, lex_le


@dataclass
class CQHRResult:
    """Numeric adjustments plus the keys needed for randomized membership.

    ``adjustments`` has shape (n_test, d), ``calibration_scores`` has shape
    (n_cal,), and ``test_keys`` / ``upper_closed`` have shape (n_test,).
    Numeric adjustments alone omit the boundary decision. Use ``contains``
    with signed raw test scores computed by the same CQR scoring rule.
    """

    adjustments: np.ndarray
    quantile_adjustment: float
    calibration_scores: np.ndarray
    quantile_key: float
    test_keys: np.ndarray
    upper_closed: np.ndarray
    _safe_test_lengths: np.ndarray = field(repr=False)
    _reference_test_lengths: np.ndarray = field(repr=False)

    def contains(self, raw_scores_test):
        """Return (n_test,) randomized acceptance for signed scores (n_test,d).

        The candidate scalar key remains fixed for each test covariate while
        checking different candidate outcomes. Unequal scores are compared
        numerically; only exact scalar ties use the random secondary key.
        """
        raw_scores_test = np.asarray(raw_scores_test, dtype=float)
        if raw_scores_test.shape != self.adjustments.shape:
            raise ValueError("raw_scores_test must have shape (n_test, d) matching adjustments")
        converted = (raw_scores_test * self._reference_test_lengths
                     / self._safe_test_lengths)
        scalar_scores = np.max(converted, axis=1)
        return lex_le(scalar_scores, self.test_keys,
                      self.quantile_adjustment, self.quantile_key)


def _validate_cqhr_inputs(raw_scores_cal, base_lengths_cal, base_lengths_test):
    """Validate CQHR array shapes and return float arrays."""
    raw_scores_cal = np.asarray(raw_scores_cal, dtype=float)
    base_lengths_cal = np.asarray(base_lengths_cal, dtype=float)
    base_lengths_test = np.asarray(base_lengths_test, dtype=float)

    if raw_scores_cal.ndim != 2:
        raise ValueError("raw_scores_cal must be a 2D array.")
    if base_lengths_cal.shape != raw_scores_cal.shape:
        raise ValueError("base_lengths_cal must have the same shape as raw_scores_cal.")
    if base_lengths_test.ndim != 2:
        raise ValueError("base_lengths_test must be a 2D array.")
    if base_lengths_test.shape[1] != raw_scores_cal.shape[1]:
        raise ValueError(
            "base_lengths_test must have the same number of coordinates as raw_scores_cal."
        )
    if np.any(base_lengths_cal < 0) or np.any(base_lengths_test < 0):
        raise ValueError("CQHR base interval lengths must be nonnegative.")

    return raw_scores_cal, base_lengths_cal, base_lengths_test


def cqhr_adjustments(
    raw_scores_cal,
    base_lengths_cal,
    base_lengths_test,
    alpha=0.1,
    reference_dim=0,
    min_length=1e-12,
    *,
    random_state=42,
    scalar_keys=None,
    test_keys=None,
):
    """
    Compute CQHR test-point-specific coordinate adjustments.

    This implements the calibration stage of Algorithm 2 from Sampson and Chan
    (2024), with the existing additional interval-length floor. The first
    coordinate is the default reference dimension. The calibration CQR score is

        E_ij = max{q_low_j(X_i) - Y_ij, Y_ij - q_high_j(X_i)}.

    Each coordinate score is converted to the reference coordinate's interval
    length scale, then one conformal quantile is taken over the row-wise maxima.
    The resulting scalar adjustment is mapped back to every test point using
    that test point's base interval lengths.

    Parameters
    ----------
    raw_scores_cal : array-like of shape (n_cal, n_targets)
        Signed CQR calibration scores.
    base_lengths_cal : array-like of shape (n_cal, n_targets)
        Base quantile interval lengths on calibration covariates.
    base_lengths_test : array-like of shape (n_test, n_targets)
        Base quantile interval lengths on test covariates.
    alpha : float, default=0.1
        Target miscoverage level.
    reference_dim : int, default=0
        Coordinate used as the reference scale.
    min_length : float, default=1e-12
        Numerical floor for interval lengths used in scale ratios; this
        regularization is an implementation addition, not specified in the paper.
    random_state : int, default=42
        Seed for one Generator drawing independent calibration and test keys.
    scalar_keys : array-like of shape (n_cal,), optional
        Calibration secondary keys in [0, 1), used only for exact scalar ties.
    test_keys : array-like of shape (n_test,), optional
        Independent candidate secondary keys, fixed for each test covariate.
        No coordinate keys are needed: CQHR does not rank raw coordinates.

    Returns
    -------
    CQHRResult
        Holds numeric adjustments, scalar calibration scores and quantile,
        random keys, and per-test upper_closed boundary flags. Its contains
        method includes the randomized test comparison; raw <= comparisons
        against the numeric adjustments would lose the tie-breaking rule.
    """
    raw_scores_cal, base_lengths_cal, base_lengths_test = _validate_cqhr_inputs(
        raw_scores_cal=raw_scores_cal,
        base_lengths_cal=base_lengths_cal,
        base_lengths_test=base_lengths_test,
    )
    _, n_targets = raw_scores_cal.shape
    if not 0 <= reference_dim < n_targets:
        raise ValueError("reference_dim must index one of the target coordinates.")
    if min_length <= 0:
        raise ValueError("min_length must be positive.")

    safe_cal_lengths = np.maximum(base_lengths_cal, min_length)
    safe_test_lengths = np.maximum(base_lengths_test, min_length)

    reference_cal_lengths = safe_cal_lengths[:, [reference_dim]]
    converted_scores = raw_scores_cal * reference_cal_lengths / safe_cal_lengths
    calibration_scores = np.max(converted_scores, axis=1)
    rng = np.random.default_rng(random_state)
    scalar_keys = draw_keys(calibration_scores.shape, rng, scalar_keys)
    test_keys = draw_keys((len(base_lengths_test),), rng, test_keys)
    quantile_adjustment, quantile_key = conformal_quantile_pair(
        calibration_scores, scalar_keys, alpha,
    )

    reference_test_lengths = safe_test_lengths[:, [reference_dim]]
    adjustments = quantile_adjustment * safe_test_lengths / reference_test_lengths

    upper_closed = lex_le(quantile_adjustment, test_keys,
                          quantile_adjustment, quantile_key)
    return CQHRResult(
        adjustments=adjustments,
        quantile_adjustment=quantile_adjustment,
        calibration_scores=calibration_scores,
        quantile_key=quantile_key,
        test_keys=test_keys,
        upper_closed=upper_closed,
        _safe_test_lengths=safe_test_lengths,
        _reference_test_lengths=reference_test_lengths,
    )

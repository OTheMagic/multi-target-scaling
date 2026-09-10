import numpy as np

from utility.conformal_utils import conformal_quantile


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
):
    """
    Compute CQHR test-point-specific coordinate adjustments.

    This implements Algorithm 2 from Sampson and Chan (2024), using the first
    coordinate as the default reference dimension. The calibration CQR score is

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
        Numerical floor for interval lengths used in scale ratios.

    Returns
    -------
    adjustments : np.ndarray of shape (n_test, n_targets)
        Coordinate-wise outcome-space adjustments for each test covariate.
    quantile_adjustment : float
        The conformal adjustment on the reference coordinate's scale.
    calibration_scores : np.ndarray of shape (n_cal,)
        Row-wise maximum rescaled calibration scores.
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
    quantile_adjustment = conformal_quantile(calibration_scores, alpha)

    reference_test_lengths = safe_test_lengths[:, [reference_dim]]
    adjustments = quantile_adjustment * safe_test_lengths / reference_test_lengths

    return adjustments, quantile_adjustment, calibration_scores

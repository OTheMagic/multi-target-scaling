"""Empirical copula calibration from Messoudi, Destercke and Rousseau (2021).

https://arxiv.org/pdf/2101.12002, equations (25)--(27), Remark 4.1.
Extracted from utility/copula.py, then aligned with the paper's empirical
marginals, common probability threshold, and generalized empirical inverse.
Independent uniform coordinate keys extend tied residuals to ordered pairs;
the empirical-CDF construction is applied to those pairs without jitter.
This implementation has no independent conformal calibration split or scalar
conformal-score calibration stage.
"""

import numpy as np

from ..rectangle import Rectangle
from ..tie_breaking import draw_keys, lex_le, lex_order


def _scores_array(scores):
    """Validate finite signed margins; retain the empty-fit +inf fallback."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] == 0 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite (n, d) array with d positive")
    return scores


class EmpiricalCopula:
    """Empirical copula of coordinatewise (residual, random key) pairs.

    After fitting, U has shape (n, d), with each column containing the
    distinct empirical ranks 1/n, ..., 1. coordinate_keys stores the
    keys used for that ordering; reuse them for empirical inversion.
    """

    def __init__(self):
        self.U = None
        self.coordinate_keys = None
        self.n = 0
        self.d = 0

    def fit(self, X, *, coordinate_keys=None, random_state=42):
        """Fit finite signed residuals X of shape (n, d), assigning keys.

        Optional coordinate_keys must have shape (n, d) and values in
        [0, 1). Otherwise independent uniform keys are drawn from a local
        Generator seeded by random_state. Only equal numeric residuals
        are reordered. Duplicate (residual, key) pairs raise ValueError.
        """
        X = _scores_array(X)
        self.n, self.d = X.shape
        rng = np.random.default_rng(random_state)
        self.coordinate_keys = draw_keys(X.shape, rng, coordinate_keys)
        self.U = self._to_uniform(X)

    def _to_uniform(self, X):
        """Return empirical pair-CDF values, i.e. lexicographic ranks / n."""
        order = lex_order(X, self.coordinate_keys, axis=0)
        ranks = np.empty(X.shape, dtype=float)
        if self.n:
            np.put_along_axis(ranks, order,
                              np.arange(1, self.n + 1)[:, None], axis=0)
            ranks /= self.n
        return ranks

    def cdf(self, u):
        """Evaluate C_n(u) = mean(all(U <= u)) for u of shape (d,)."""
        u = np.asarray(u)
        assert u.shape[0] == self.d, "Dimensionality mismatch"
        return np.mean(np.all(self.U <= u, axis=1))

    def quantile_box(self, alpha):
        """Return the first common marginal level reaching 1-alpha coverage.

        Scan the empirical copula's finite jump grid for the smallest tau
        with C_n(tau, ..., tau) >= 1-alpha, then return tau repeated d times.
        Exact equality with the target need not be attainable. If no level
        is found, retain the existing fallback of ones (including empty fits).
        This step does not add an independent scalar conformal quantile.
        """
        for level in np.unique(self.U):
            thresholds = np.full(self.d, level)
            if self.cdf(thresholds) >= 1 - alpha:
                return thresholds
        return np.ones(self.d)


def inverse_ecdf_transform(U_thresholds, scores, *, coordinate_keys=None,
                           return_keys=False):
    """Invert empirical margins at probability thresholds of shape (d,).

    scores has shape (n, d) with finite, possibly signed values and d positive.
    Supplying coordinate_keys of the same
    shape orders tied scores by those keys. return_keys=True returns
    (upper, upper_keys), each of shape (d,), and requires the keys used
    in the forward fit. The default returns only numeric upper bounds.
    Empty margins or thresholds outside (0, 1] retain the existing +inf
    fallback; the corresponding key is also +inf (whole-domain sentinel).
    """
    scores = _scores_array(scores)
    n, d = scores.shape
    if return_keys and coordinate_keys is None:
        raise ValueError("return_keys=True requires the coordinate_keys used in fit")
    if coordinate_keys is None:
        scores_sorted = np.sort(scores, axis=0, kind="mergesort")
        keys_sorted = None
    else:
        coordinate_keys = draw_keys(scores.shape, None, coordinate_keys)
        order = lex_order(scores, coordinate_keys, axis=0)
        scores_sorted = np.take_along_axis(scores, order, axis=0)
        keys_sorted = np.take_along_axis(coordinate_keys, order, axis=0)
    U_thresholds = np.asarray(U_thresholds)
    # Search the forward probability grid directly; multiplying a rounded
    # fraction by n and rounding up can spuriously advance an order statistic.
    probability_grid = np.arange(1, n + 1, dtype=float) / n
    indices = np.searchsorted(probability_grid, U_thresholds, side="left")
    upper = np.full(d, np.inf)
    upper_keys = np.full(d, np.inf)
    valid = (U_thresholds > 0) & (U_thresholds <= 1) & (indices < n)
    columns = np.arange(d)[valid]
    upper[valid] = scores_sorted[indices[valid], columns]
    if return_keys:
        upper_keys[valid] = keys_sorted[indices[valid], columns]
        return upper, upper_keys
    return upper


def empirical_copula_prediction(scores, alpha=0.2, random_state=42, *,
                                lower=0.0, coordinate_keys=None, test_keys=None):
    """Return an empirical-copula rectangle for one test observation.

    scores is a finite, possibly signed calibration-score array of shape
    (n, d), with d positive. An empty calibration array retains the existing
    +inf upper-bound fallback.
    Fit coordinatewise pair-CDF ranks divided by n, find the paper's common
    marginal level with empirical coverage at least 1-alpha, and invert
    each margin using the same score/key pairs. This empirical coverage
    target is not a new split-conformal finite-sample guarantee.

    Optional coordinate_keys (n, d) and test_keys (d,) supply uniform
    secondary keys in [0, 1). Otherwise calibration keys and then test keys
    are drawn from one local Generator seeded by random_state. Reuse
    these test keys across candidate outcomes for one test observation;
    distinct test observations need independent test keys. Duplicate pairs
    in ranked margins or at an inverse boundary raise ValueError.

    The returned Rectangle stores numeric bounds and per-coordinate
    upper_closed flags determined by the fixed test keys. Membership
    uses contains/contain_points; numeric bounds alone discard the
    randomized boundary rule. Keys never numerically perturb residuals.

    lower is the known attainable lower score bound for this test observation:
    a scalar or (d,) vector, finite or -inf. It defaults to zero for absolute
    residuals. For signed CQR scores use lower = -test_interval_widths / 2;
    use -inf for scores with no finite lower bound. Calibration scores are
    neither clipped nor checked against this test-specific domain. If an
    upper bound is below lower, the returned rectangle is empty. Its volume
    measures score space, not the corresponding outcome intervals.
    """
    scores = _scores_array(scores)
    _, d = scores.shape
    rng = np.random.default_rng(random_state)
    coordinate_keys = draw_keys(scores.shape, rng, coordinate_keys)
    test_keys = draw_keys((d,), rng, test_keys)
    cop = EmpiricalCopula()
    cop.fit(scores, coordinate_keys=coordinate_keys)
    thresholds = cop.quantile_box(alpha=alpha)
    upper, upper_keys = inverse_ecdf_transform(
        thresholds, scores, coordinate_keys=coordinate_keys, return_keys=True)
    upper_closed = lex_le(upper, test_keys, upper, upper_keys)
    return Rectangle(upper=upper, lower=lower, upper_closed=upper_closed,
                     allow_empty=True)

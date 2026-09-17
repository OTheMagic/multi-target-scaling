"""Unscaled maximum-score calibration, extracted from utility/unscaled.py.

The original numeric score and order-statistic convention are preserved.
Independent uniform secondary keys break scalar-score ties without numerical
jitter. There is no coordinate ranking step in this method.
"""

import numpy as np

from ..rectangle import Rectangle
from ..tie_breaking import conformal_quantile_pair, draw_keys, lex_le


def unscaled_prediction(scores, alpha=0.2, random_state=42, *,
                        lower=0.0, scalar_keys=None, test_key=None):
    """Return a randomized unscaled rectangle for one test observation.

    scores has shape (n, d), with n and d positive, and contains finite
    calibration scores, which may be signed. Each row's numeric maximum is
    the scalar conformity score. Its conformal order statistic is selected using
    (score, independent uniform key) in lexicographic order.

    alpha is the miscoverage level in (0, 1). random_state seeds a
    local Generator. Optional scalar_keys (n,) and scalar test_key
    supply keys explicitly; all must be in [0, 1). Otherwise calibration
    keys and then the test key are drawn from that Generator. Keys never
    change unequal numeric scores. Exact pair collisions raise ValueError.

    The returned Rectangle has a shared upper bound in all d coordinates.
    Its upper faces are included exactly when the fixed test key precedes
    the selected calibration key. Use contains/contain_points to
    honor those open or closed faces; comparing only upper loses the
    randomization. Reuse the test key for all candidate outcomes belonging
    to this test observation; use independent test keys for other tests.

    lower is the known attainable lower score bound for this test observation:
    a scalar or (d,) vector, finite or -inf. It defaults to zero for absolute
    residuals. For signed CQR scores use lower = -test_interval_widths / 2;
    use -inf for scores with no finite lower bound. Calibration scores are
    neither clipped nor checked against this test-specific domain. If an
    upper bound is below lower, the returned rectangle is empty. Its volume
    measures score space, not the corresponding outcome intervals.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2 or 0 in scores.shape or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite (n, d) array with n and d positive")
    _, d = scores.shape
    rng = np.random.default_rng(random_state)
    max_norm = np.max(scores, axis=1)
    scalar_keys = draw_keys(max_norm.shape, rng, scalar_keys)
    test_key = draw_keys((), rng, test_key)
    quantile_threshold, cutoff_key = conformal_quantile_pair(
        max_norm, scalar_keys, alpha)
    upper_closed = bool(lex_le(quantile_threshold, test_key,
                               quantile_threshold, cutoff_key))
    return Rectangle(upper=np.repeat(quantile_threshold, d),
                     lower=lower, upper_closed=upper_closed, allow_empty=True)

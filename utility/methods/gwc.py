"""Signed GWC extracted from utility/envelope.py:EnvelopeCalibration.predict.

The user selected this signed construction, rather than the original
nonnegative GWC in utility/res_rescaled.py. Its defaults come from Envelope.
"""
import numpy as np

from ..conformal_utils import conformal_rank
from ..signed_geometry import EnvelopeRegion, interval_sup, link, score_at
from ..tie_breaking import conformal_quantile_pair, draw_keys, lex_le


def _gwc_from_statistics(scores, alpha, lower, rank, mean, scale, scalar_keys, test_key):
    """Global stage shared with Envelope, using already fitted statistics."""
    n, d = scores.shape
    lower = np.broadcast_to(np.asarray(lower, float), (d,)).copy()
    if np.isnan(lower).any() or np.isposinf(lower).any():
        raise ValueError("lower must be finite or -inf")
    if rank > n or np.any(scale == 0):
        return EnvelopeRegion(
            lower, np.full(d, np.inf), np.full(d, np.inf), np.inf,
            evaluations=(0,) * d,
            fallback="infinite_rank" if rank > n else "zero_calibration_scale",
            test_key=test_key,
        )
    global_scores = np.column_stack([
        interval_sup(scores[:, j], mean[j], scale[j], n, lower[j], np.inf)
        for j in range(d)
    ])
    qg, qkey = conformal_quantile_pair(global_scores.max(axis=1), scalar_keys, alpha)
    qg, qkey = float(qg), float(qkey)
    closed = np.full(d, bool(lex_le(qg, test_key, qg, qkey)))
    upper_g = link(qg, mean, scale, n)
    # Preserve the original self-score guard against inverse cancellation.
    accepted_lower = np.isfinite(lower) & lex_le(
        score_at(lower, mean, scale, n, lower), test_key, qg, qkey)
    closed = np.where(accepted_lower & (upper_g <= lower), True, closed)
    upper_g = np.where(accepted_lower, np.maximum(upper_g, lower), upper_g)
    empty = bool(np.any(upper_g < lower) or np.any(np.isneginf(upper_g))
                 or np.any((upper_g == lower) & ~closed))
    return EnvelopeRegion(
        lower, upper_g.copy(), upper_g, qg, empty=empty, evaluations=(0,) * d,
        upper_closed=closed, gwc_upper_closed=closed, q_gwc_key=qkey, test_key=test_key,
    )


def gwc_prediction(scores, alpha=0.1, lower=0.0, *, random_state=42,
                   scalar_keys=None, test_key=None):
    """Return the signed global-worst-case score-domain rectangle.

    scores is a nonempty finite (n,d) calibration array. lower is a scalar
    or (d,) attainable score-domain bound (finite or -inf); signed CQR uses
    minus the test interval half-width. alpha is the miscoverage level.
    The returned EnvelopeRegion has (d,) bounds, an empty flag, q_gwc,
    and the original infinite-rank/zero-scale fallback diagnostics.
    Independent uniform scalar_keys (n,) order equal row scores; test_key is
    one independent uniform held fixed for this test instance. Defaults draw
    both sequentially from random_state. Use region membership methods to
    honor open upper boundaries. Raw residuals are not ranked or perturbed:
    their original numeric values determine means, scales, and suprema.
    """
    scores = np.asarray(scores, float)
    if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty, finite (n,d) array")
    n, _ = scores.shape
    rank = conformal_rank(n, alpha)
    mean = scores.mean(axis=0)
    scale = scores.std(axis=0)
    rng = np.random.default_rng(random_state)
    scalar_keys = draw_keys((n,), rng, scalar_keys)
    test_key = float(draw_keys((), rng, test_key))
    return _gwc_from_statistics(scores, alpha, lower, rank, mean, scale,
                                scalar_keys, test_key)

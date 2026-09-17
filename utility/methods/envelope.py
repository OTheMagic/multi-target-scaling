"""Signed surface-envelope TSCP extracted from utility/envelope.py.

Definition: envelope_method/signed_envelope.tex.

Cells retain the source's distinct numeric residual knots. Random secondary
keys order scalar calibration scores; the same row keys and candidate key are
reused throughout global and local calibration. Boundaries can therefore be open.
"""
import numpy as np

from ..conformal_utils import conformal_quantile, conformal_rank
from ..tie_breaking import conformal_quantile_pair, draw_keys, lex_le


from ..signed_geometry import (
    EnvelopeRegion, interval_sup, link, score_at,
)
from .gwc import _gwc_from_statistics


def link_values(q, mean, scale, n):
    """Vectorized signed inverse for rowwise rank localization."""
    q = np.asarray(q, float)
    limit = n / np.sqrt(n + 1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denom = np.sqrt((n - np.sqrt(n + 1)*q) * (n + np.sqrt(n + 1)*q))
        value = mean + scale*q*(n+1)/denom
    return np.where(q <= -limit, -np.inf, np.where(q >= limit, np.inf, value))


class EnvelopeCalibration:
    """Reuse calibration statistics across test-dependent residual domains."""

    def __init__(self, scores, alpha=0.1, *, random_state=42, scalar_keys=None):
        scores = np.asarray(scores, float)
        if scores.ndim != 2 or min(scores.shape) < 1 or not np.isfinite(scores).all():
            raise ValueError("scores must be a nonempty, finite (n,d) array")
        self.n, self.d = scores.shape
        self.rank = conformal_rank(self.n, alpha)
        self.alpha = alpha
        self.scores = scores
        self.mean = scores.mean(axis=0)
        self.scale = scores.std(axis=0)
        self.sorted = [np.unique(scores[:, j]) for j in range(self.d)]
        self.rng = np.random.default_rng(random_state)
        self.scalar_keys = draw_keys((self.n,), self.rng, scalar_keys)

    def predict(self, lower=0.0, *, search="backward", return_cells=False, test_key=None):
        """Return one test instance's region, drawing its key once per call.

        Supply the same test_key to compare search modes or to replay a region.
        contain_points tests multiple candidates against this one fixed key;
        it does not draw independent keys for separate test observations.
        """
        if search not in ("backward", "exhaustive", "rank"):
            raise ValueError("search must be backward, exhaustive, or rank")
        n, d, m, s, t = self.n, self.d, self.mean, self.scale, self.scores
        test_key = float(draw_keys((), self.rng, test_key))
        global_region = _gwc_from_statistics(t, self.alpha, lower, self.rank, m, s,
                                             self.scalar_keys, test_key)
        lower = global_region.lower
        if global_region.fallback or global_region.empty:
            return (global_region, [[] for _ in range(d)]) if return_cells else global_region
        upper_g, qg = global_region.gwc_upper, global_region.q_gwc
        local_global = np.column_stack([
            interval_sup(t[:, j], m[j], s[j], n, lower[j], upper_g[j]) for j in range(d)])
        # Leave-one-coordinate maxima in O(nd), including ties between maxima.
        left = np.maximum.accumulate(local_global, axis=1)
        right = np.maximum.accumulate(local_global[:, ::-1], axis=1)[:, ::-1]
        bounds = np.full(d, -np.inf)
        closed = np.zeros(d, dtype=bool)
        evaluations, all_cells = [], []
        for j in range(d):
            off = np.full(n, -np.inf)
            if j:
                off = np.maximum(off, left[:, j - 1])
            if j + 1 < d:
                off = np.maximum(off, right[:, j + 1])
            points = self.sorted[j]
            points = points[(points > lower[j]) & (points < upper_g[j])]
            points = np.concatenate(([lower[j]], points, [upper_g[j]]))
            cells = []
            count = 0
            start = len(points) - 2
            if search == "rank" and n > 1:
                crossing = np.maximum(t[:, j], link_values(off, m[j], s[j], n))
                cutoff = float(conformal_quantile(crossing, self.alpha))
                # Numeric pruning only: random acceptance is a subset of the
                # closed set used by the original pruning certificate.
                certified = m[j] - s[j] / np.sqrt(n - 1)
                # Only discard cells proved to fail; keep the uncertified range.
                # An outward guard retains cells near floating-point ties.
                if np.isfinite(cutoff):
                    cutoff += 64*np.finfo(float).eps*(1 + abs(cutoff) + abs(m[j]) + s[j])
                keep_through = max(cutoff, certified)
                start = min(start, max(0, np.searchsorted(points[:-1], keep_through, side="right") - 1))
            # First surviving cell from the right has the largest endpoint.
            for k in range(start, -1, -1):
                lo, hi = points[k:k + 2]
                active = interval_sup(t[:, j], m[j], s[j], n, lo, hi)
                q, qkey = conformal_quantile_pair(
                    np.maximum(active, off), self.scalar_keys, self.alpha)
                q, qkey = float(q), float(qkey)
                inverse = float(link(q, m[j], s[j], n))
                bound = min(hi, inverse)
                bound_closed = bool(hi < inverse or lex_le(q, test_key, q, qkey))
                if np.isfinite(lo) and lex_le(
                        score_at(lo, m[j], s[j], n, lo), test_key, q, qkey):
                    if bound <= lo:
                        bound_closed = True
                    bound = max(lo, bound)
                if bound == upper_g[j] and not global_region.upper_closed[j]:
                    bound_closed = False
                survives = bool((bound > lo or (bound == lo and bound_closed))
                                and not np.isneginf(bound))
                count += 1
                if return_cells:
                    cells.append(dict(k=k, lo=float(lo), hi=float(hi), q=q,
                                      q_key=qkey, bound=bound, upper_closed=bound_closed,
                                      survives=survives))
                if survives:
                    if bound > bounds[j]:
                        closed[j] = bound_closed
                    elif bound == bounds[j]:
                        closed[j] |= bound_closed
                    bounds[j] = max(bounds[j], bound)
                    if search != "exhaustive":
                        break
            evaluations.append(count)
            all_cells.append(sorted(cells, key=lambda c: c['k']))
        region = EnvelopeRegion(lower, bounds, upper_g, qg,
                                empty=bool(np.any(np.isneginf(bounds))),
                                evaluations=tuple(evaluations), upper_closed=closed,
                                gwc_upper_closed=global_region.upper_closed,
                                q_gwc_key=global_region.q_gwc_key, test_key=test_key)
        return (region, all_cells) if return_cells else region


def envelope_prediction(scores, alpha=0.1, lower=0.0, *, search="backward", return_cells=False,
                        random_state=42, scalar_keys=None, test_key=None):
    """Calibrate finite scores (n,d) and return a score-domain EnvelopeRegion.

    lower is a scalar or (d,) domain bound; signed CQR uses minus the
    test interval half-width. search selects the existing coordinate-cell
    search, and return_cells also returns the evaluated cells per coordinate.
    scalar_keys (n,) break scalar ties and are reused in every cell. Raw
    residual ties remain identical geometric knots, not ranked observations.
    Numeric means/scales and the source's geometric partition are unchanged.
    """
    return EnvelopeCalibration(scores, alpha, random_state=random_state,
                                scalar_keys=scalar_keys).predict(
        lower, search=search, return_cells=return_cells, test_key=test_key
    )

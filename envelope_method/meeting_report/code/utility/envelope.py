"""Signed surface-envelope TSCP; see envelope_method/signed_envelope.tex.

All comparisons use closed cells, so ties need neither jitter nor continuity.
Backward search is exact for any ordered partition; no prefix assumption is used.
"""
from dataclasses import dataclass

import numpy as np

from utility.conformal_utils import conformal_quantile, conformal_rank


def score_at(t, mean, scale, n, z):
    t, z = np.broadcast_arrays(np.asarray(t, float), np.asarray(z, float))
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        delta = z - mean
        sigma = np.hypot(scale, delta / np.sqrt(n + 1))
        value = (t - mean) / sigma - (delta / sigma) / (n + 1)
    value = np.where(np.isposinf(z), -1 / np.sqrt(n + 1), value)
    return np.where(np.isneginf(z), 1 / np.sqrt(n + 1), value)


def interval_sup(t, mean, scale, n, lo, hi):
    """Exact supremum over a nonempty closed interval, with infinite endpoints."""
    t = np.asarray(t, float)
    if lo > hi or scale <= 0:
        raise ValueError("Require lo <= hi and positive scale.")
    result = np.maximum(score_at(t, mean, scale, n, lo),
                        score_at(t, mean, scale, n, hi))
    above = t > mean
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        critical = mean - scale * (scale / (t - mean))
        peak = np.hypot((t - mean) / scale, 1 / np.sqrt(n + 1))
    inside = above & (critical >= lo) & (critical <= hi)
    return np.where(inside, np.maximum(result, peak), result)


def link(q, mean, scale, n):
    """Increasing inverse of the candidate self-score on the extended real line."""
    mean, scale = np.asarray(mean, float), np.asarray(scale, float)
    limit = n / np.sqrt(n + 1)
    if q >= limit:
        return np.full(np.broadcast_shapes(mean.shape, scale.shape), np.inf)
    if q <= -limit:
        return np.full(np.broadcast_shapes(mean.shape, scale.shape), -np.inf)
    denom = np.sqrt((n - np.sqrt(n + 1) * q) * (n + np.sqrt(n + 1) * q))
    return mean + scale * q * (n + 1) / denom


def link_values(q, mean, scale, n):
    """Vectorized signed inverse for rowwise rank localization."""
    q = np.asarray(q, float)
    limit = n / np.sqrt(n + 1)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        denom = np.sqrt((n - np.sqrt(n + 1)*q) * (n + np.sqrt(n + 1)*q))
        value = mean + scale*q*(n+1)/denom
    return np.where(q <= -limit, -np.inf, np.where(q >= limit, np.inf, value))


@dataclass
class EnvelopeRegion:
    lower: np.ndarray
    upper: np.ndarray
    gwc_upper: np.ndarray
    q_gwc: float
    empty: bool = False
    evaluations: tuple = ()
    fallback: str = ""

    def contain_points(self, points):
        points = np.asarray(points)
        return np.all((points >= self.lower) & (points <= self.upper), axis=-1) & ~np.bool_(self.empty)

    def contains(self, point):
        return bool(self.contain_points(point))

    def length_along_dimensions(self):
        return np.zeros_like(self.upper) if self.empty else self.upper - self.lower

    def volume(self):
        lengths = self.length_along_dimensions()
        if self.empty or np.any(lengths == 0):
            return 0.0
        return float(np.prod(lengths))

    def info(self):
        return np.array([self.lower, self.upper])


class EnvelopeCalibration:
    """Reuse calibration statistics across test-dependent residual domains."""

    def __init__(self, scores, alpha=0.1):
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

    def predict(self, lower=0.0, *, search="backward", return_cells=False):
        if search not in ("backward", "exhaustive", "rank"):
            raise ValueError("search must be backward, exhaustive, or rank")
        lower = np.broadcast_to(np.asarray(lower, float), (self.d,)).copy()
        if np.isnan(lower).any() or np.isposinf(lower).any():
            raise ValueError("lower must be finite or -inf")
        n, d, m, s, t = self.n, self.d, self.mean, self.scale, self.scores
        if self.rank > n or np.any(s == 0):
            region = EnvelopeRegion(lower, np.full(d, np.inf), np.full(d, np.inf), np.inf,
                                    evaluations=(0,) * d,
                                    fallback="infinite_rank" if self.rank > n else "zero_calibration_scale")
            return (region, [[] for _ in range(d)]) if return_cells else region
        global_scores = np.column_stack([
            interval_sup(t[:, j], m[j], s[j], n, lower[j], np.inf) for j in range(d)])
        qg = float(conformal_quantile(global_scores.max(axis=1), self.alpha))
        upper_g = link(qg, m, s, n)
        # Inverse cancellation can put an exactly accepted boundary just below
        # the domain, especially at a capped zero. Retain it by its self-score.
        accepted_lower = np.isfinite(lower) & (qg >= score_at(lower, m, s, n, lower))
        upper_g = np.where(accepted_lower, np.maximum(upper_g, lower), upper_g)
        if np.any(upper_g < lower) or np.any(np.isneginf(upper_g)):
            region = EnvelopeRegion(lower, upper_g.copy(), upper_g, qg, empty=True, evaluations=(0,) * d)
            return (region, [[] for _ in range(d)]) if return_cells else region
        local_global = np.column_stack([
            interval_sup(t[:, j], m[j], s[j], n, lower[j], upper_g[j]) for j in range(d)])
        # Leave-one-coordinate maxima in O(nd), including ties between maxima.
        left = np.maximum.accumulate(local_global, axis=1)
        right = np.maximum.accumulate(local_global[:, ::-1], axis=1)[:, ::-1]
        bounds = np.full(d, -np.inf)
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
                q = float(conformal_quantile(np.maximum(active, off), self.alpha))
                bound = min(hi, float(link(q, m[j], s[j], n)))
                if np.isfinite(lo) and q >= score_at(lo, m[j], s[j], n, lo):
                    bound = max(lo, bound)
                survives = bool(bound >= lo and not np.isneginf(bound))
                count += 1
                if return_cells:
                    cells.append(dict(k=k, lo=float(lo), hi=float(hi), q=q,
                                      bound=bound, survives=survives))
                if survives:
                    bounds[j] = max(bounds[j], bound)
                    if search != "exhaustive":
                        break
            evaluations.append(count)
            all_cells.append(sorted(cells, key=lambda c: c['k']))
        region = EnvelopeRegion(lower, bounds, upper_g, qg,
                                empty=bool(np.any(np.isneginf(bounds))),
                                evaluations=tuple(evaluations))
        return (region, all_cells) if return_cells else region


def envelope_prediction(scores, alpha=0.1, lower=0.0, **kwargs):
    return EnvelopeCalibration(scores, alpha).predict(lower, **kwargs)

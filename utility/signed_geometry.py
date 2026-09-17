"""Shared signed score geometry from utility/envelope.py.

EnvelopeRegion keeps its source name and represents both signed GWC and
Envelope score-domain boxes, including empty, infinite, and open boundaries.
The numerical score, supremum, and inverse formulas remain unchanged.
"""
from dataclasses import dataclass

import numpy as np


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


@dataclass
class EnvelopeRegion:
    lower: np.ndarray
    upper: np.ndarray
    gwc_upper: np.ndarray
    q_gwc: float
    empty: bool = False
    evaluations: tuple = ()
    fallback: str = ""
    upper_closed: object = True
    gwc_upper_closed: object = True
    q_gwc_key: float = np.inf
    test_key: object = None

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=float)
        self.upper = np.asarray(self.upper, dtype=float)
        self.gwc_upper = np.asarray(self.gwc_upper, dtype=float)
        if (self.lower.ndim != 1 or self.upper.shape != self.lower.shape
                or self.gwc_upper.shape != self.lower.shape):
            raise ValueError("lower, upper, and gwc_upper must have the same (d,) shape")
        self.upper_closed = np.broadcast_to(
            np.asarray(self.upper_closed, dtype=bool), self.lower.shape
        ).copy()
        self.gwc_upper_closed = np.broadcast_to(
            np.asarray(self.gwc_upper_closed, dtype=bool), self.lower.shape
        ).copy()
        self.empty = bool(self.empty or np.any(self.lower > self.upper) or np.any(
            (self.lower == self.upper) & ~self.upper_closed
        ))

    def contain_points(self, points):
        """Test (..., d) points with the stored test key's boundary decisions."""
        points = np.asarray(points, dtype=float)
        if points.ndim < 1 or points.shape[-1] != self.lower.size:
            raise ValueError("points must have shape (..., d) matching the region")
        upper_ok = (points < self.upper) | ((points == self.upper) & self.upper_closed)
        return np.all((points >= self.lower) & upper_ok, axis=-1) & ~np.bool_(self.empty)

    def contains(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != self.lower.shape:
            raise ValueError("point must have shape (d,) matching the region")
        return bool(self.contain_points(point))

    def length_along_dimensions(self):
        return np.zeros_like(self.upper) if self.empty else self.upper - self.lower

    def volume(self):
        lengths = self.length_along_dimensions()
        if self.empty or np.any(lengths == 0):
            return 0.0
        return float(np.prod(lengths))

    def info(self):
        """Return numeric corners; membership methods also honor open boundaries."""
        return np.array([self.lower, self.upper])



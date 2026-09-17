import math
from fractions import Fraction
from typing import Optional

import numpy as np


def conformal_rank(n_samples: int, alpha: float) -> int:
    """Return ceil((n_samples + 1) * (1 - alpha)) using exact arithmetic.

    Interpret alpha through its decimal string representation, so 0.72 means
    exactly 18/25. Fraction inputs preserve an explicitly supplied rational
    probability. No decimal truncation or integer-boundary tolerance is used.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    alpha_exact = Fraction(str(alpha))
    return math.ceil((1 - alpha_exact) * (n_samples + 1))


def conformal_quantile(values, alpha: float, axis: Optional[int] = None):
    """
    Return the conformal empirical quantile, or +inf if the rank exceeds n.

    This uses np.partition instead of fully sorting the input. It is equivalent
    for the selected order statistic and is faster in repeated experiments.
    Equal scores retain their original value; this numeric-only helper does
    not implement randomized acceptance. The extracted methods use
    tie_breaking.conformal_quantile_pair for scalar calibration. Envelope
    also uses this numeric helper for its conservative search-pruning bound.
    """
    values = np.asarray(values)
    if axis is None:
        values = values.reshape(-1)
        n_samples = values.size
        rank = conformal_rank(n_samples, alpha)
        if rank > n_samples:
            return np.inf
        return np.partition(values, rank - 1)[rank - 1]

    n_samples = values.shape[axis]
    rank = conformal_rank(n_samples, alpha)
    if rank > n_samples:
        out_shape = list(values.shape)
        out_shape.pop(axis)
        return np.full(out_shape, np.inf)

    partitioned = np.partition(values, rank - 1, axis=axis)
    return np.take(partitioned, rank - 1, axis=axis)


def add_jitter(values, random_state: Optional[int] = 42, scale: float = 1e-10):
    """Add tiny deterministic jitter for reproducible tie-breaking."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_state)
    return values + scale * rng.random(values.shape)

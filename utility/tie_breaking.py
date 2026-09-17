"""Exact random secondary keys for residual and conformal-score comparisons.

The numeric score is always the primary key.  Secondary keys are independent
uniform draws and never perturb scores or fitted statistics.  Callers own the
Generator and must reuse observation keys across repeated transformations.
"""

import numpy as np

from .conformal_utils import conformal_rank


def _validate_keys(keys, shape):
    keys = np.asarray(keys, dtype=float)
    if keys.shape != shape:
        raise ValueError(f"keys must have shape {shape}, got {keys.shape}")
    if np.any(~np.isfinite(keys)) or np.any((keys < 0) | (keys >= 1)):
        raise ValueError("keys must be finite numbers in [0, 1)")
    return keys


def draw_keys(shape, rng, keys=None):
    """Return supplied keys or draw keys using the caller's Generator.

    Supplied keys must have the exact requested shape and lie in [0, 1).
    Generated floating-point collisions within this draw are redrawn, giving
    a uniform ordering without replacement on the finite random-number grid.
    A separate draw can still collide; ranked pairs and candidate comparisons
    detect such unresolved ties explicitly.
    """
    shape = (int(shape),) if np.isscalar(shape) else tuple(shape)
    if keys is not None:
        return _validate_keys(keys, shape).copy()
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    result = rng.random(shape)
    flat = result.reshape(-1)
    while flat.size:
        _, first = np.unique(flat, return_index=True)
        repeated = np.ones(flat.size, dtype=bool)
        repeated[first] = False
        if not np.any(repeated):
            break
        flat[repeated] = rng.random(np.count_nonzero(repeated))
    return result


def lex_order(values, keys, axis=0):
    """Indices sorting by numeric value, then key, along ``axis``.

    Duplicate (value, key) pairs on a ranked axis are an unresolved tie and
    raise ValueError; observation index is never an implicit third key.
    """
    values = np.asarray(values)
    keys = _validate_keys(keys, values.shape)
    if np.any(np.isnan(values)):
        raise ValueError("values must not contain NaN")
    order = np.lexsort((keys, values), axis=axis)
    sorted_values = np.moveaxis(np.take_along_axis(values, order, axis), axis, -1)
    sorted_keys = np.moveaxis(np.take_along_axis(keys, order, axis), axis, -1)
    duplicate = ((sorted_values[..., 1:] == sorted_values[..., :-1])
                 & (sorted_keys[..., 1:] == sorted_keys[..., :-1]))
    if np.any(duplicate):
        raise ValueError("duplicate (value, key) pairs; supply fresh random keys")
    return order


def conformal_quantile_pair(values, keys, alpha):
    """Return the conformal order-statistic (numeric value, secondary key).

    Inputs are one-dimensional. Use the shared exact decimal/rational rank;
    rank above the sample size returns the whole-domain sentinel (inf, inf).
    """
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    rank = conformal_rank(values.size, alpha)
    order = lex_order(values, keys)
    if rank > values.size:
        return np.inf, np.inf
    index = order[rank - 1]
    return values[index], np.asarray(keys, dtype=float)[index]


def coordinate_quantile_pairs(scores, keys, alpha):
    """Return numeric cutoffs and keys, each (d,), from scores/keys (n,d)."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("scores must have shape (n, d) with d positive")
    rank = conformal_rank(scores.shape[0], alpha)
    order = lex_order(scores, keys, axis=0)
    if rank > scores.shape[0]:
        return np.full(scores.shape[1], np.inf), np.full(scores.shape[1], np.inf)
    columns = np.arange(scores.shape[1])
    rows = order[rank - 1]
    return scores[rows, columns], np.asarray(keys, dtype=float)[rows, columns]


def lex_le(values, keys, cutoff, cutoff_key):
    """Broadcast candidate-pair <= cutoff-pair comparisons.

    Exact equality of both candidate value and key raises ValueError: the
    finite random key has collided and must be replaced. This helper is for
    candidate comparisons, not comparing a calibration row to itself.
    """
    values, keys, cutoff, cutoff_key = np.broadcast_arrays(
        np.asarray(values), np.asarray(keys, dtype=float),
        np.asarray(cutoff), np.asarray(cutoff_key, dtype=float),
    )
    _validate_keys(keys, keys.shape)
    if np.any(np.isnan(values)) or np.any(np.isnan(cutoff)):
        raise ValueError("values and cutoffs must not contain NaN")
    regular_key = np.isfinite(cutoff_key) & (cutoff_key >= 0) & (cutoff_key < 1)
    infinite_rank = np.isposinf(cutoff) & np.isposinf(cutoff_key)
    if np.any(~(regular_key | infinite_rank)):
        raise ValueError("cutoff keys must be in [0, 1) or use the (inf, inf) sentinel")
    equal_value = values == cutoff
    if np.any(equal_value & (keys == cutoff_key)):
        raise ValueError("candidate (value, key) equals cutoff; supply a fresh key")
    return (values < cutoff) | (equal_value & (keys < cutoff_key))

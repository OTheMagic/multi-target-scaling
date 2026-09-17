"""Shared outcome-space summaries for the retained experiment computations."""

import numpy as np


def _nonnegative_mean(values, axis=0):
    """Average nonnegative values without overflowing a finite sum."""
    scale = np.max(values, axis=axis)
    # A zero maximum needs no scaling; an infinite maximum must remain infinite.
    scale = np.where((scale > 0) & np.isfinite(scale), scale, 1.0)
    with np.errstate(over="ignore", under="ignore"):
        return (values / np.expand_dims(scale, axis=axis)).mean(axis=axis) * scale


def summarize_predictions(coordinate_covered, lengths, empty, *,
                          adjustments=None, joint_covered=None):
    """Summarize one set of test regions, without pooling repeated experiments.

    Inputs are coordinate membership and outcome lengths ``(m,d)``, explicit
    empty flags ``(m,)``, and optional authoritative method membership ``(m,)``.
    Empty rows have zero lengths and no covered coordinates. Zero-volume rows
    have log10 volume -inf, including rectangles with both zero/infinite sides.
    Logs are summed over side lengths to avoid product overflow/underflow.
    Ordinary finite, positive direct products are preserved; products that
    become zero or nonfinite without a zero side are recomputed from those
    logs. A positive infinite side gives infinite volume unless another side
    is zero. Nonnegative means are scaled to avoid sum overflow. These numeric
    safeguards replace naive product/mean arithmetic without dropping rows.
    Mean log10 volume is NaN when both -inf and +inf occur; neither is excluded.
    """
    lengths = np.asarray(lengths, dtype=float).copy()
    covered_coordinates = np.asarray(coordinate_covered, dtype=bool).copy()
    empty = np.asarray(empty, dtype=bool)
    if lengths.ndim != 2 or min(lengths.shape) < 1:
        raise ValueError("lengths must have nonempty shape (n_test, n_targets)")
    if covered_coordinates.shape != lengths.shape or empty.shape != (len(lengths),):
        raise ValueError("membership and empty shapes must match lengths")
    if np.isnan(lengths).any() or (lengths < 0).any():
        raise ValueError("lengths must be nonnegative and not NaN")
    lengths[empty] = 0.0
    covered_coordinates[empty] = False
    if joint_covered is None:
        covered = covered_coordinates.all(axis=1)
    else:
        covered = np.asarray(joint_covered, dtype=bool).copy()
        if covered.shape != empty.shape:
            raise ValueError("joint_covered must have shape (n_test,)")
        covered[empty] = False
    zero_volume = (lengths == 0).any(axis=1)
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        volumes = np.prod(lengths, axis=1)
        logs = np.log10(lengths).sum(axis=1)
        volumes[zero_volume] = 0.0
        logs[zero_volume] = -np.inf
        recover = ~zero_volume & ((volumes == 0) | ~np.isfinite(volumes))
        volumes[recover] = np.power(10.0, logs[recover])
        volumes[~zero_volume & np.isposinf(lengths).any(axis=1)] = np.inf
        mean_log = float(logs.mean())
    result = dict(
        outcome_lengths=lengths, outcome_volumes=volumes, log10_volumes=logs,
        coordinate_covered=covered_coordinates, covered=covered, empty=empty.copy(),
        n_test=len(lengths), covered_count=int(covered.sum()),
        test_coverage=float(covered.mean()),
        coordinate_coverage=covered_coordinates.mean(axis=0),
        outcome_volume=float(_nonnegative_mean(volumes)), mean_log10_volume=mean_log,
        coordinate_lengths=_nonnegative_mean(lengths), empty_rate=float(empty.mean()),
    )
    if adjustments is not None:
        adjustments = np.asarray(adjustments, dtype=float)
        if adjustments.shape != lengths.shape or np.isnan(adjustments).any():
            raise ValueError("adjustments must match lengths and contain no NaNs")
        result["mean_adjustments"] = adjustments.mean(axis=0)
    return result

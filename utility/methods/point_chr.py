"""Point CHR extracted from ``utility/data_splitting.py``.

The original numeric split, order statistics, and scaling are preserved.
Exact random secondary keys order marginal residuals and final scalar scores.
The absolute-residual construction matches Sampson and Chan, Algorithm 1:
https://arxiv.org/pdf/2406.04498 (positive finite base interval widths).
"""

import numpy as np
from sklearn.model_selection import train_test_split

from ..rectangle import Rectangle
from ..conformal_utils import conformal_rank
from ..tie_breaking import (
    coordinate_quantile_pairs, conformal_quantile_pair, draw_keys, lex_le,
)


def data_spliting_CHR_prediction(scores,
                                 alpha=0.2,
                                 reference_dim=0,
                                 random_state=42,
                                 *, coordinate_keys=None, scalar_keys=None,
                                 test_key=None):
    """
    Data-splitting CHR-style conformal rectangular prediction.

    This implements a CHR-type adjustment: a base rectangle is built from
    the first half of the data using a marginal (1 - alpha)-quantile per
    dimension. Then, on the second half, the excess over this base rectangle
    is rescaled relative to a reference dimension, and a further quantile
    adjustment is computed and added.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) of absolute calibration residuals from an
        independently trained point predictor; not raw signed residuals.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Targets the (1 - alpha)-quantile in
        each step.
    reference_dim : int, default=0
        Index of the reference coordinate used for scaling the excess.
    random_state : int, default=42
        Seed used for the train/test split and independent random tie keys.
    coordinate_keys : array-like of shape (n, d), optional
        Secondary keys in [0, 1) for the first-half marginal order statistics.
        These keys order equal residuals without changing the base widths.
    scalar_keys : array-like of shape (n,), optional
        Independent secondary keys; second-half entries order the final
        scalar excess scores. Keys follow the same row split as the scores.
    test_key : float in [0, 1), optional
        Independent candidate scalar key, held fixed throughout this region.
        Omitted keys are drawn from one Generator initialized by random_state.

    Returns
    -------
    Rectangle
        A Rectangle whose upper bounds are the base upper quantiles plus
        the CHR-style adjustment based on excess scores. Its upper_closed
        flags encode the randomized scalar boundary decision; use contains
        or contain_points rather than comparing only the numeric bounds.
    """
    scores = np.asarray(scores)
    rng = np.random.default_rng(random_state)
    coordinate_keys = draw_keys(scores.shape, rng, coordinate_keys)
    scalar_keys = draw_keys((len(scores),), rng, scalar_keys)
    test_key = draw_keys((), rng, test_key)
    first, second = train_test_split(
        np.arange(len(scores)), test_size=0.5, random_state=random_state,
    )
    scores1, scores2 = scores[first], scores[second]

    n, d = scores1.shape
    quantile_level = conformal_rank(n, alpha)

    # Coordinate keys select an observation within a tie; numeric base widths
    # stay unchanged. The final independent scalar key controls acceptance.
    if quantile_level <= n:
        base_upper, _ = coordinate_quantile_pairs(
            scores1, coordinate_keys[first], alpha,
        )
    else:
        base_upper = np.repeat(np.inf, d)
        return Rectangle(upper=base_upper)

    # Compute the excess length and excess scores
    excess = scores2 - base_upper
    scale = base_upper[reference_dim] / base_upper
    scaled_excess = excess * scale
    excess_scores = np.max(scaled_excess, axis=1)

    # Recalibration uses its own sample size. With an odd total, the
    # second half is larger and can require a higher conformal rank.
    adj1, adjustment_key = conformal_quantile_pair(
        excess_scores, scalar_keys[second], alpha,
    )
    adjustments = adj1 * base_upper / base_upper[reference_dim]

    upper = base_upper + adjustments

    upper_closed = bool(lex_le(adj1, test_key, adj1, adjustment_key))
    return Rectangle(upper=upper, upper_closed=upper_closed)

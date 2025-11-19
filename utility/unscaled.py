import numpy as np
import math
from utility.rectangle import Rectangle


def unscaled_prediction(scores, alpha = 0.2):
    """
    Conformal rectangular prediction using the unscaled max-norm method.

    This method:
    1. Computes the max-norm of each row of `scores`.
    2. Adds a tiny random jitter to break ties.
    3. Takes the (1 - alpha)-quantile of these max-norm values.
    4. Constructs a hyper-rectangle where every coordinate has the same
       upper bound equal to this quantile threshold.

    Parameters
    ----------
    scores : np.ndarray, shape (n, d)
        Calibration scores or residuals used to compute the conformal region.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). The region targets coverage ≥ 1 - alpha.

    Returns
    -------
    Rectangle
        A hyper-rectangle with identical upper bounds across all dimensions,
        equal to the (1 - alpha)-quantile of the max-norm.
    """
    n, d = scores.shape

    # Break ties randomly
    np.random.seed(42)
    max_norm = np.max(scores, axis = 1) + 1e-10*np.random.rand(n)

    # Sort the scores and get the quantile for each coordinate
    max_norm_sorted = np.sort(max_norm, axis=0, kind="mergesort")
    quantile_level = math.ceil((1 - alpha) * (n + 1))
    quantile_threshold = max_norm_sorted[quantile_level-1] if quantile_level <= n else np.inf
    upper = np.repeat(quantile_threshold, d)

    return Rectangle(upper=upper)


def bonferroni_prediction(scores, alpha = 0.2):
    """
    Conformal rectangular prediction using per-coordinate Bonferroni correction.

    For each coordinate j:
    1. Apply Bonferroni correction alpha/d.
    2. Add small random jitter to break ties.
    3. Compute the (1 - alpha/d)-quantile of the marginal scores_j.
    4. Construct a hyper-rectangle with these per-dimension quantile upper bounds.

    Parameters
    ----------
    scores : np.ndarray, shape (n, d)
        Calibration scores or residuals.
    alpha : float, default=0.2
        Total miscoverage level in (0, 1). Bonferroni allocates alpha/d per dimension.

    Returns
    -------
    Rectangle
        A hyper-rectangle whose j-th upper bound is the empirical
        (1 - alpha/d)-quantile of scores[:, j].
    """
    n, d = scores.shape

    # Bonferroni correction
    alpha_corrected = alpha / d

    # Break ties randomly
    np.random.seed(42)
    scores = scores + 1e-10*np.random.rand(n, d)

    # Sort the scores and get the quantile for each coordinate
    scores_sorted = np.sort(scores, axis=0, kind="mergesort")
    quantile_level = math.ceil((1 - alpha_corrected) * (n + 1))
    upper = scores_sorted[quantile_level-1] if quantile_level <= n else np.repeat(np.inf, d)

    return Rectangle(upper=upper)

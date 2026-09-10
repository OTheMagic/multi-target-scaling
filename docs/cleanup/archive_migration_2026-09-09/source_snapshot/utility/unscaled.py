import numpy as np
from utility.rectangle import Rectangle
from utility.conformal_utils import add_jitter, conformal_quantile


def unscaled_prediction(scores, alpha=0.2, random_state=42):
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
    _, d = scores.shape

    max_norm = add_jitter(np.max(scores, axis=1), random_state=random_state)
    quantile_threshold = conformal_quantile(max_norm, alpha)
    upper = np.repeat(quantile_threshold, d)

    return Rectangle(upper=upper)


def bonferroni_prediction(scores, alpha=0.2, random_state=42):
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
    _, d = scores.shape

    # Bonferroni correction
    alpha_corrected = alpha / d

    scores = add_jitter(scores, random_state=random_state)
    upper = conformal_quantile(scores, alpha_corrected, axis=0)

    return Rectangle(upper=upper)

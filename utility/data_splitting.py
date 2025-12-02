import numpy as np
import math
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split


def naive_prediction(scores,
                     alpha = 0.2, 
                     random_state = 42):
    """
    Standardized conformal prediction using sample mean and sample std directly estimated on the scores. 
    Note this method is theoratically invalid (should not have marginal coverage).

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) of calibration scores or residuals.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Targets the (1 - alpha)-quantile.
    random_state : int, default=42
        Seed used for tie-breaking jitter on the max-norms.

    Returns
    -------
    Rectangle
        A Rectangle with upper bounds quantile_threshold * std + mu.
    """
    scale = np.std(scores, axis=0, ddof=1)
    mean = np.mean(scores, axis=0)
    scores_standardized = (scores-mean)/scale
    n, d = scores.shape

    np.random.seed(random_state)
    max_norm_standardized = np.max(scores_standardized, axis = 1)+ 1e-10*np.random.randint(n)
    max_norm_standardized_sorted = np.sort(max_norm_standardized, axis=0, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_standardized_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = quantile_threshold*scale + mean

    return Rectangle(upper=upper)

def data_splitting_oracle_prediction(scores, mu, std,
                                     alpha = 0.2, 
                                     random_state = 42):
    """
    Data-splitting conformal prediction using oracle standardization.

    Assumes the true per-dimension mean and standard deviation (mu, std)
    are known. The scores are standardized, and the (1 - alpha)-quantile
    of their max-norm is translated back to the original scale to form
    an axis-aligned rectangular prediction region.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) of calibration scores or residuals.
    mu : np.ndarray or float
        True mean(s) for standardization, per dimension or scalar.
    std : np.ndarray or float
        True standard deviation(s) for standardization, per dimension or scalar.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Targets the (1 - alpha)-quantile.
    random_state : int, default=42
        Seed used for tie-breaking jitter on the max-norms.

    Returns
    -------
    Rectangle
        A Rectangle with upper bounds quantile_threshold * std + mu.
    """
    scores_standardized = (scores-mu)/std
    n, d = scores.shape

    np.random.seed(random_state)
    max_norm_standardized = np.max(scores_standardized, axis = 1)+ 1e-10*np.random.randint(n)
    max_norm_standardized_sorted = np.sort(max_norm_standardized, axis=0, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_standardized_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = quantile_threshold*std + mu

    return Rectangle(upper=upper)


def data_splitting_standardized_prediction(scores, 
                                           alpha = 0.2, 
                                           random_state = 42):
    """
    Data-splitting conformal prediction with estimated mean and scale.

    The data is split in half. The first half is used to estimate a
    per-dimension mean and standard deviation. The second half is
    standardized using these estimates, and the (1 - alpha)-quantile
    of the max-norm of the standardized scores is mapped back to the
    original scale as a rectangular prediction region.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) of calibration scores or residuals.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Targets the (1 - alpha)-quantile.
    random_state : int, default=42
        Seed used for the train/test split and tie-breaking jitter.

    Returns
    -------
    Rectangle
        A Rectangle with upper bounds quantile_threshold * scale + mean.
    """
    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    scale = np.std(scores1, axis=0, ddof=1)
    mean = np.mean(scores1, axis=0)
    scores_standardized = (scores2-mean)/scale
    n, d = scores2.shape

    np.random.seed(random_state)
    max_norm_standardized = np.max(scores_standardized, axis = 1)+ 1e-10*np.random.randint(n)
    max_norm_standardized_sorted = np.sort(max_norm_standardized, axis=0, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_standardized_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = quantile_threshold*scale + mean

    return Rectangle(upper=upper)


def data_spliting_CHR_prediction(scores, 
                                 alpha = 0.2, 
                                 reference_dim = 0,
                                 random_state = 42):
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
        Array of shape (n, d) of calibration scores or residuals.
    alpha : float, default=0.2
        Miscoverage level in (0, 1). Targets the (1 - alpha)-quantile in
        each step.
    reference_dim : int, default=0
        Index of the reference coordinate used for scaling the excess.
    random_state : int, default=42
        Seed used for the train/test split.

    Returns
    -------
    Rectangle
        A Rectangle whose upper bounds are the base upper quantiles plus
        the CHR-style adjustment based on excess scores.
    """
    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    n, d = scores1.shape
    quantile_level = math.ceil((1 - alpha) * (n + 1))

    # Compute the base rectangle
    scores_sorted = np.sort(scores1, axis=0, kind="mergesort")

    if quantile_level <= n:
        base_upper = scores_sorted[quantile_level-1] 
    else:
        base_upper = np.repeat(np.inf, d)
        return Rectangle(upper=base_upper)

    # Compute the excess length and excess scores
    excess = scores2 - base_upper
    scale = base_upper[reference_dim]/base_upper
    scaled_excess = excess*scale
    excess_scores = np.max(scaled_excess, axis = 1)

    # Compute adjustments
    excess_scores_sorted = np.sort(excess_scores, kind="mergesort")
    adj1 = excess_scores_sorted[quantile_level-1] if quantile_level <= n else np.inf
    adjustments = adj1*base_upper/base_upper[reference_dim]

    upper = base_upper+adjustments

    return Rectangle(upper=upper)

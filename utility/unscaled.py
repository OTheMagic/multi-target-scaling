import numpy as np
import math
from utility.rectangle import Rectangle

def unscaled_prediction(scores, alpha = 0.2):

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

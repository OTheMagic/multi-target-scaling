import numpy as np
import math
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split

def no_scaling_prediction_region(scores, alpha = 0.2, random_state = 42):

    max_norm = np.max(scores, axis = 1)
    max_norm_sorted = np.sort(max_norm, axis=0, kind="mergesort")

    n = len(scores)
    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = np.repeat(quantile_threshold, len(scores[0]))

    return Rectangle(upper=upper)
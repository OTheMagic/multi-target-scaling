import numpy as np
import math
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split

def data_splitting_prediction_region(scores, alpha = 0.2, random_state = 42):

    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    scale = np.std(scores1, axis=0)
    scores_scaled = scores2/scale
    max_norm_scaled = np.max(scores_scaled, axis = 1)
    max_norm_scaled_sorted = np.sort(max_norm_scaled, axis=0, kind="mergesort")

    n = len(scores2)
    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_scaled_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = quantile_threshold*scale

    return Rectangle(upper=upper)
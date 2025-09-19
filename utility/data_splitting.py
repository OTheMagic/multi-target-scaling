import numpy as np
import math
from utility.rectangle import Rectangle
from sklearn.model_selection import train_test_split

def data_splitting_scaled_prediction(scores, 
                                     alpha = 0.2, 
                                     random_state = 42):

    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    scale = np.std(scores1, axis=0)
    scores_scaled = scores2/scale
    n, d = scores2.shape

    np.random.seed(random_state)
    max_norm_scaled = np.max(scores_scaled, axis = 1)+ 1e-10*np.random.randint(n)
    max_norm_scaled_sorted = np.sort(max_norm_scaled, axis=0, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_scaled_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    upper = quantile_threshold*scale

    return Rectangle(upper=upper)

def data_splitting_standardized_prediction(scores, 
                                             alpha = 0.2, 
                                             random_state = 42):

    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    scale = np.std(scores1, axis=0)
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
    
    scores1, scores2 = train_test_split(scores, test_size=0.5, random_state=random_state)

    n, d = scores1.shape
    quantile_level = math.ceil((1 - alpha) * (n + 1))

    # Compute the base rectangle
    scores_sorted = np.sort(scores1, axis=0, kind="mergesort")
    base_upper = scores_sorted[quantile_level-1] if quantile_level <= n else np.repeat(np.inf, d)

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
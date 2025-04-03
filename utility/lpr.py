# Import necessary packages
import numpy as np
import math
from itertools import combinations, product
from utility.rectangle import Rectangle

# Basic handling tools
def mean_index_solver(scores):
        
    scores_mean = np.transpose(np.array([np.mean(scores, axis=0)]))
    scores_sorted = np.sort(np.transpose(scores), axis=1, kind="mergesort")
    abs_diff = np.abs(scores_mean - scores_sorted)
    mask = (scores_mean - scores_sorted) > 0
    abs_diff[mask] = np.inf 
    mean_index = np.argmin(abs_diff, axis=1)

    return mean_index

def check_coverage_rate(scores, regions, one_rect=True):

    """
    Computes the coverage rate of a set of scores within given regions.

    Parameters
    ----------
    scores : numpy.ndarray
        A 2D array of shape (n_samples, n_features) representing the points to evaluate.
    regions : Rectangle or list of Rectangle
        The region(s) to check for point containment. If `one_rect` is True, this is a 
        single `Rectangle` object. Otherwise, it is a list of `Rectangle` objects.
    one_rect : bool, optional
        If True, `regions` is treated as a single rectangle. If False, `regions` is
        treated as a list of rectangles. Defaults to True.

    Returns
    -------
    float
        The coverage rate, defined as the proportion of `scores` that are contained in 
        at least one region. The value ranges from 0 to 1.
    """

    evaluation = np.zeros(len(scores))
    if one_rect == True:
        evaluation += regions.contain_points(scores).astype(int)
    else:
        for region in regions:
            evaluation += region.contain_points(scores).astype(int)
    return np.sum(evaluation > 0)/(len(scores))


# Prediction region configuration for a given guess of y
def determine_point_min_max(rectangle, mean_vector):

    dim = len(mean_vector)
    point_min = np.zeros(dim)
    point_max = np.zeros(dim)
    for j in range(dim):
        interval = [rectangle.lower[j], rectangle.upper[j]]
        if (mean_vector[j] >= interval[0]) and (mean_vector[j] <= interval[1]):
            point_min[j] = mean_vector[j]
            if (np.abs(mean_vector[j] - interval[0] )>= np.abs(mean_vector[j] - interval[1])):
                point_max[j] = interval[0]
            else:
                point_max[j] = interval[1]
        elif mean_vector[j] > interval[1]:
            point_min[j] = interval[1]
            point_max[j] = interval[0]
        else: 
            point_min[j] = interval[0]
            point_max[j] = interval[1]
    
    return point_min, point_max

def upper_quadratic_solver(A, B, C):
    delta = B**2-4*A*C
    if delta > 0:
        return (-B+np.sqrt(delta))/(2**A)
    else:
        raise ValueError("Unable to solve this quadratic function")
    
def scale_with_point(scores, point, type = "min"):

    point = point.reshape(1, -1)
    scores_new = np.append(scores, point, axis=0)
    std_dev = np.std(scores_new, axis=0)
    
    if type == "min":
        scaled_scores = scores / std_dev
        return scaled_scores
    elif type == "max":
        return std_dev
    else:
        raise ValueError("Type is either min or max")
    
def compute_quantile_threshold(scores, alpha, point):

    # Scaling
    scores_scaled = scale_with_point(scores, point, type = "min")
    max_norm_scaled = np.max(scores_scaled, axis=1)
    max_norm_scaled_sorted = np.sort(max_norm_scaled, axis=0, kind="mergesort")

    # Figure out the quantie threshold
    n = len(scores)
    quantile_level = math.ceil((1 - alpha) * (n + 1))
    if quantile_level <= n:
        quantile_threshold = max_norm_scaled_sorted[quantile_level-1]
    else:
        quantile_threshold = np.inf

    return quantile_threshold

def compute_prediction_region(scores, rectangle, alpha, scores_mean):

    n = scores.shape[0]
    ratio = n/((n+1)**2)
    point_min, point_max = determine_point_min_max(rectangle, scores_mean)
    quantile_threshold = compute_quantile_threshold(scores, alpha, point_min)
    # Bounded rectangle
    if np.all(rectangle.upper < np.inf):
        scale = scale_with_point(scores, point_max, type = "max")
        region = Rectangle(quantile_threshold*scale).intersection(rectangle)
    
    ## Unbounded rectangle
    elif quantile_threshold**2 < (n+1)**2/n:
        A = 1-(n/(n+1)**2)*(quantile_threshold**2)

        inf_indices = np.where(np.isinf(rectangle.upper))[0]
        for index in inf_indices:
            B = scores_mean[index]*ratio*(quantile_threshold**2)
            K = (quantile_threshold**2)*(1/(n+1)*np.sum((np.transpose(scores)[index]-scores_mean[index])**2)+ratio*scores_mean[index]**2)
            C = upper_quadratic_solver(A, B, -K)
            if C <= rectangle.lower[index]:
                return None
            else:
                rectangle.update_upper(C, index)

        point_min, point_max = determine_point_min_max(rectangle, scores_mean)
        scale = scale_with_point(scores, point_max, type = "max")
        region = Rectangle(quantile_threshold*scale).intersection(rectangle)
    else:
        region = rectangle
        
    return region
    
def one_rect_prediction_regions_nD(scores, alpha = 0.2, short_cut = True):
    
    # Number of samples, number of dimensions
    n = scores.shape[0]
    d = scores.shape[1]

    # Appending, sorting
    scores_augmented = np.append(scores, [np.repeat(np.inf, d), np.zeros(d)], axis=0)
    scores_sorted = np.transpose(np.sort(scores_augmented, axis=0, kind="mergesort"))

    # Grab the rectangle that contains the mean
    scores_mean = np.mean(np.transpose(scores), axis=1)
    mean_index = mean_index_solver(scores)+1

    # Inner helper functions
    def create_hyper_rectangle(indices):
        upper = [scores_sorted[dim][indices[dim]] for dim in range(d)]
        lower = [scores_sorted[dim][indices[dim]-1] for dim in range(d)]
        return Rectangle(upper, lower)
    
    def binary_search_dimension(fixed_indices, dim_along, max_bounds, start, end):

        # Edge case: the leftmost or the rightmost rectangle is hit
        if start >= end-1:
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(indices)
            region = compute_prediction_region(scores, rectangle, alpha, scores_mean)
            if region:
                np.maximum(max_bounds, region.upper, out=max_bounds)
            return

        # Binary search from the middle
        mid = (start + end) // 2
        indices = np.copy(fixed_indices)
        indices[dim_along] = mid
        rectangle = create_hyper_rectangle(indices)
        region = compute_prediction_region(scores, rectangle, alpha, scores_mean)

        # If prediction region is not none, record the maximum and continue searching
        if region is not None:
            np.maximum(max_bounds, region.upper, out=max_bounds)
            binary_search_dimension(fixed_indices, dim_along, max_bounds, mid, end)
        else:
            binary_search_dimension(fixed_indices, dim_along, max_bounds, start, mid)


    if short_cut:
        max_bounds = np.zeros(d)
        for idx in range(d):
            fix_indices = np.copy(mean_index)
            fix_indices[idx] = 1
            binary_search_dimension(fix_indices, idx, max_bounds, 1, n+1)
        return Rectangle(upper=max_bounds)
    else:   
        regions = []
        max_bounds = np.zeros(d, dtype=float)
        for indices in product(range(1, n+2), repeat=d):
            rectangle = create_hyper_rectangle(indices)
            region = compute_prediction_region(scores, rectangle, alpha, scores_mean)
            if region:
                regions.append(region)
                max_bounds = np.maximum(max_bounds, region.upper)

        return regions, Rectangle(upper=max_bounds)

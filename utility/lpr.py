# Import necessary packages
import numpy as np
import math
from itertools import combinations, product
from utility.rectangle import Rectangle
from typing import Union, List, Tuple

# Basic handling tools
def mean_index_solver(scores):
    """
    Find the index of the value just below the mean along each dimension.

    Parameters
    ----------
    scores : np.ndarray
        A 2D array of shape (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Index of the score just below the mean for each dimension.
    """
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
    scores : np.ndarray
        A 2D array representing the test scores.
    regions : Rectangle or list of Rectangle
        The prediction region(s).
    one_rect : bool
        If True, treats `regions` as a single Rectangle. If False, as a list.

    Returns
    -------
    float
        Proportion of points that lie inside at least one region.
    """
    evaluation = np.zeros(len(scores))
    if one_rect:
        evaluation += regions.contain_points(scores).astype(int)
    else:
        for region in regions:
            evaluation += region.contain_points(scores).astype(int)
    return np.sum(evaluation > 0) / len(scores)


def determine_point_min_max(rectangle, mean_vector):
    """
    Determine min and max projection points from mean to rectangle edges.

    Parameters
    ----------
    rectangle : Rectangle
        The rectangle to evaluate.
    mean_vector : np.ndarray
        Mean of the calibration scores.

    Returns
    -------
    tuple
        point_min and point_max used for scaling.
    """
    dim = len(mean_vector)
    point_min = np.zeros(dim)
    point_max = np.zeros(dim)
    for j in range(dim):
        interval = [rectangle.lower[j], rectangle.upper[j]]
        if interval[0] <= mean_vector[j] <= interval[1]:
            point_min[j] = mean_vector[j]
            point_max[j] = interval[0] if abs(mean_vector[j] - interval[0]) >= abs(mean_vector[j] - interval[1]) else interval[1]
        elif mean_vector[j] > interval[1]:
            point_min[j] = interval[1]
            point_max[j] = interval[0]
        else:
            point_min[j] = interval[0]
            point_max[j] = interval[1]
    return point_min, point_max

def upper_quadratic_solver(A, B, C):
    """
    Solve quadratic equation and return upper root.

    Parameters
    ----------
    A, B, C : float
        Coefficients of the quadratic equation.

    Returns
    -------
    float
        The upper root of the quadratic equation.
    """
    delta = B**2 - 4*A*C
    if delta > 0:
        return (-B + np.sqrt(delta)) / (2*A)
    else:
        raise ValueError("Unable to solve this quadratic function")

def scale_with_point(scores, point, type="min"):
    """
    Scale scores relative to a point.

    Parameters
    ----------
    scores : np.ndarray
        The calibration scores.
    point : np.ndarray
        The reference point.
    type : str
        Type of scaling. "min" returns scaled scores; "max" returns std_dev.

    Returns
    -------
    np.ndarray
        Scaled scores or standard deviations.
    """
    point = point.reshape(1, -1)
    scores_new = np.append(scores, point, axis=0)
    std_dev = np.std(scores_new, axis=0)
    if type == "min":
        return scores / std_dev
    elif type == "max":
        return std_dev
    else:
        raise ValueError("Type is either min or max")

def compute_quantile_threshold(scores, alpha, point):
    """
    Compute threshold quantile for max-norm of scaled scores.

    Parameters
    ----------
    scores : np.ndarray
    alpha : float
    point : np.ndarray

    Returns
    -------
    float
        Threshold value from quantile.
    """
    scores_scaled = scale_with_point(scores, point, type="min")
    np.random.seed(42)
    max_norm_scaled = np.max(scores_scaled, axis=1) + 1e-10 * np.random.randint(0, 1)
    max_norm_scaled_sorted = np.sort(max_norm_scaled, kind="mergesort")
    n = len(scores)
    quantile_level = math.ceil((1 - alpha) * (n + 1))
    return max_norm_scaled_sorted[quantile_level - 1] if quantile_level <= n else np.inf

def compute_prediction_region(scores, rectangle, alpha, scores_mean):
    """
    Compute the intersection region for a given base rectangle and quantile threshold.

    Parameters
    ----------
    scores : np.ndarray
    rectangle : Rectangle
    alpha : float
    scores_mean : np.ndarray

    Returns
    -------
    Rectangle or None
        The resulting prediction region or None if empty.
    """
    n = scores.shape[0]
    ratio = n / ((n + 1)**2)
    point_min, point_max = determine_point_min_max(rectangle, scores_mean)
    quantile_threshold = compute_quantile_threshold(scores, alpha, point_min)

    if np.all(rectangle.upper < np.inf):
        scale = scale_with_point(scores, point_max, type="max")
        return Rectangle(quantile_threshold * scale).intersection(rectangle)

    elif quantile_threshold**2 < (n + 1)**2 / n:
        A = 1 - ratio * (quantile_threshold**2)
        inf_indices = np.where(np.isinf(rectangle.upper))[0]
        for index in inf_indices:
            B = scores_mean[index] * ratio * (quantile_threshold**2)
            K = (quantile_threshold**2) * (
                (1 / (n + 1)) * np.sum((scores[:, index] - scores_mean[index])**2) + ratio * scores_mean[index]**2
            )
            C = upper_quadratic_solver(A, B, -K)
            if C <= rectangle.lower[index]:
                return None
            else:
                rectangle.update_upper(C, index)
        point_min, point_max = determine_point_min_max(rectangle, scores_mean)
        scale = scale_with_point(scores, point_max, type="max")
        return Rectangle(quantile_threshold * scale).intersection(rectangle)

    else:
        return rectangle

def one_rect_prediction_regions_nD(scores, alpha=0.2, short_cut=True) -> Union[Rectangle, Tuple[List[Rectangle], Rectangle]]:
    """
    Construct conformal prediction region using rectangular partitioning.

    Parameters
    ----------
    scores : np.ndarray
    alpha : float
    short_cut : bool

    Returns
    -------
    Rectangle or (List[Rectangle], Rectangle)
        The region(s) satisfying the coverage guarantee.
    """
    n, d = scores.shape
    scores_augmented = np.append(scores, [np.repeat(np.inf, d), np.zeros(d)], axis=0)
    scores_sorted = np.transpose(np.sort(scores_augmented, axis=0, kind="mergesort"))
    scores_mean = np.mean(scores.T, axis=1)
    mean_index = mean_index_solver(scores) + 1

    def create_hyper_rectangle(indices):
        upper = [scores_sorted[dim][indices[dim]] for dim in range(d)]
        lower = [scores_sorted[dim][indices[dim] - 1] for dim in range(d)]
        return Rectangle(upper, lower)

    def binary_search_dimension(fixed_indices, dim_along, max_bounds, start, end):
        if start >= end - 1:
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(indices)
            region = compute_prediction_region(scores, rectangle, alpha, scores_mean)
            if region:
                np.maximum(max_bounds, region.upper, out=max_bounds)
            return
        mid = (start + end) // 2
        indices = np.copy(fixed_indices)
        indices[dim_along] = mid
        rectangle = create_hyper_rectangle(indices)
        region = compute_prediction_region(scores, rectangle, alpha, scores_mean)
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
            binary_search_dimension(fix_indices, idx, max_bounds, 1, n + 1)
        return Rectangle(upper=max_bounds)
    else:
        regions = []
        max_bounds = np.zeros(d)
        for indices in product(range(1, n + 2), repeat=d):
            rectangle = create_hyper_rectangle(indices)
            region = compute_prediction_region(scores, rectangle, alpha, scores_mean)
            if region:
                regions.append(region)
                max_bounds = np.maximum(max_bounds, region.upper)
        return regions, Rectangle(upper=max_bounds)
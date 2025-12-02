# Import necessary packages
import numpy as np
import math
from itertools import combinations, product
from utility.rectangle import Rectangle
from typing import Union, List, Tuple

### Basic handling tools

def mean_index_solver(scores):
    """
    Find the index (in sorted order) of the value just above the mean along each dimension.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, d)

    Returns
    -------
    np.ndarray
        Index of the score just below the mean in the sorted column.
    """
    # Column means
    scores_mean = scores.mean(axis=0)  # (d,)

    # Sort each column
    scores_sorted = np.sort(scores, axis=0)  # (n, d)

    # Replace values above the mean with -inf, so argmax picks the largest valid one
    candidate = np.where(scores_sorted > scores_mean, -np.inf, scores_sorted)

    # Argmax along rows gives index of the closest value <= mean
    mean_index = np.argmax(candidate, axis=0)  # (d,)

    return mean_index + 1

def create_hyper_rectangle(scores_sorted, indices):
    """
    Construct a hyper-rectangle by selecting lower and upper bounds from
    pre-sorted calibration score arrays.

    Parameters
    ----------
    scores_sorted : np.ndarray
        A list/array of length d, where each element is a sorted 1D array
        containing calibration scores for that dimension.
        That is, scores_sorted[dim] has shape (n_dim_samples,).
    indices : np.ndarray
        A 1D array of length d giving the selected rank indices for each dimension.
        For each dim, the rectangle upper bound is scores_sorted[dim][indices[dim]],
        and the lower bound is scores_sorted[dim][indices[dim] - 1].

    Returns
    -------
    Rectangle
        A Rectangle object whose `upper` and `lower` bounds are defined
        component-wise from the ranked scores.
    """
    upper = [scores_sorted[dim][indices[dim]] for dim in range(len(indices))]
    lower = [scores_sorted[dim][indices[dim] - 1] for dim in range(len(indices))]
    return Rectangle(upper, lower)


def check_coverage_rate(scores, regions, one_rect=True):
    """
    Compute the empirical coverage rate of test score points relative to
    a prediction region (or a list of regions).

    Parameters
    ----------
    scores : np.ndarray
        A 2D array of shape (n_samples, d) containing test points.
    regions : Rectangle or list of Rectangle
        A single Rectangle or a list of Rectangles to be checked.
    one_rect : bool, default=True
        If True, `regions` is treated as a single Rectangle.
        If False, `regions` is treated as a list and a point is covered
        if it lies in at least one region.

    Returns
    -------
    float
        The proportion of points in `scores` that lie inside at least one region.
    """
    evaluation = np.zeros(len(scores))
    if one_rect:
        evaluation += regions.contain_points(scores).astype(int)
    else:
        for region in regions:
            evaluation += region.contain_points(scores).astype(int)
    return np.sum(evaluation > 0) / len(scores)


def mean_clip(rectangle, mean_vector):
    """
    Project a mean vector onto the hyper-rectangle defined by its lower and upper
    bounds by performing coordinate-wise clipping.

    Parameters
    ----------
    rectangle : Rectangle
        A Rectangle with attributes `lower` and `upper`, each of shape (d,).
    mean_vector : np.ndarray
        A 1D vector of shape (d,) representing the mean of calibration scores.

    Returns
    -------
    np.ndarray
        The coordinate-wise clipped projection of `mean_vector`
        onto the rectangle.
    """
    return np.clip(mean_vector, rectangle.lower, rectangle.upper)


######################### Divider ######################### 

### Pointwise upperbounds

def scaled_transformation(scores, mu, std, clipped_mean):
    """
    Apply a scaled transformation to scores using an imputed standard deviation
    based on a clipped mean.

    Parameters
    ----------
    scores : np.ndarray, shape (n, d)
        Calibration or test scores for n samples and d dimensions.
    mu : np.ndarray or float
        Mean of the calibration scores (per-dimension or scalar).
    std : np.ndarray or float
        Standard deviation of the calibration scores (per-dimension or scalar).
    clipped_mean : np.ndarray or float
        Mean vector after clipping to a prediction rectangle (per-dimension or scalar).

    Returns
    -------
    np.ndarray, shape (n,)
        For each sample i, the maximum over dimensions of
        scores[i, j] / imputed_std[j].
    """
    imputed_var = (clipped_mean - mu)**2 / (scores.shape[0] + 1) + std**2
    imputed_std = np.sqrt(imputed_var)
    upperbound = scores / imputed_std
    return np.max(upperbound, axis=1)


def standardized_transformation(scores, mu, std, clipped_mean, global_const = None):
    """
    Apply a standardized transformation to scores based on local or global
    extremal bounds derived from Gaussian-style parameters.

    Parameters
    ----------
    scores : np.ndarray, shape (n, d)
        Calibration or test scores for n samples and d dimensions.
    mu : np.ndarray or float
        Mean of the calibration scores (per-dimension or scalar).
    std : np.ndarray or float
        Standard deviation of the calibration scores (per-dimension or scalar).
    clipped_mean : np.ndarray or float
        Mean vector after clipping to a prediction rectangle (per-dimension or scalar).
        Used only when `global_const` is not None.
    global_const : float or None, optional
        If not None, uses a simplified global upper bound:
        scores / imputed_std - global_const.
        If None, uses the more detailed local extremal construction combining
        t1 (infinity), t2 (zero), and t3 (local extrema).

    Returns
    -------
    np.ndarray, shape (n,)
        For each sample i, the maximum over dimensions of the constructed
        upper bound (either global or local, depending on `global_const`).
    """
    if global_const is not None:
        imputed_var = (clipped_mean - mu)**2 / (scores.shape[0] + 1) + std**2
        imputed_std = np.sqrt(imputed_var)
        upperbound = scores / imputed_std - global_const
        return np.max(upperbound, axis=1)
    else:
        n, d = scores.shape
        scores_centered = scores - mu

        # Infinity
        t1 = np.zeros((n, d))
        t1.fill(-1/np.sqrt(n+1))

        # Zero
        t2_num = scores_centered + mu/(n+1)
        t2_den = np.sqrt(mu**2/(n+1) + std**2)
        t2 = t2_num / t2_den

        # Local extrema
        t3_num = scores_centered**2 + std**2/(n+1)
        t3_den = np.sqrt(std**4/(n+1) + scores_centered**2*std**2)
        sign = np.where(scores_centered > 0, 1, 0) - np.where(scores_centered < 0, 1, 0)
        t3_temp = np.where(scores_centered != 0, t3_num/t3_den*sign, t2)
        t3 = np.where(mu >= std**2/scores_centered, t3_temp, t2)

        upperbound = np.maximum.reduce([t1, t2, t3])
        return np.max(upperbound, axis=1)


######################### Divider ######################### 

### Scaled prediction region

def scaled_upperbound(scores, alpha, mu, std, clipped_mean):

    n, d = scores.shape

    np.random.seed(42)
    scores_upperbound = scaled_transformation(scores, mu, std, clipped_mean) + 1e-10*np.random.rand(n)
    scores_upperbound_sorted = np.sort(scores_upperbound, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    return scores_upperbound_sorted[quantile_level - 1] if quantile_level <= n else np.inf

def scaled_threshold(upperbound, mu, std, size, dim = None):

    corr = size + 1
    control = size + 1 - upperbound**2

    mu = mu if dim is None else mu[dim]
    std = std if dim is None else std[dim]
    infs = np.repeat(np.inf, mu.shape[0]) if dim is None else np.inf

    # A > 0, quadratic, take the largest of two roots
    if control > 0:
        num = -mu * upperbound**2 + upperbound*np.sqrt(corr * (mu**2 + control * std**2))
        return num / control
            
    # A = 0, linear, take the intersection
    elif control == 0:
        return (mu**2 + corr * std**2)/(2*mu)
            
    # A < 0, quadratic, no solution
    else:
        return infs

def scaled_prediction(scores, alpha=0.2, short_cut=True) -> Union[Rectangle, Tuple[List[Rectangle], Rectangle]]:
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

    # Compute shape, mean, std, and mean indices
    n, d = scores.shape
    scores_mean = np.mean(scores, axis = 0)
    scores_std = np.std(scores, axis = 0)
    mean_index = mean_index_solver(scores) + 1

    # Append zero and inf to get ready for partition
    scores_augmented = np.append(scores, [np.repeat(np.inf, d), np.zeros(d)], axis=0)
    scores_sorted = np.transpose(np.sort(scores_augmented, axis=0, kind="mergesort"))


    # Perforem binary search along dim_along
    def binary_search_dimension(fixed_indices, dim_along, max_bounds, start, end):

        # Edge case: the last rectangle along dim_along to be evaluated
        if start >= end - 1:

            # Get the last rectangle
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(scores_sorted, indices)

            # Compute the threshold along dim_along
            clipped_mean = mean_clip(rectangle, scores_mean)
            upperbound = scaled_upperbound(scores, alpha, 
                                           scores_mean, 
                                           scores_std,
                                           clipped_mean)
            L_temp = scaled_threshold(upperbound,
                                         scores_mean,
                                         scores_std,
                                         n, dim_along)

            # Update rule
            if L_temp >= rectangle.lower[dim_along]:
                update = min(L_temp, rectangle.upper[dim_along])
                max_bounds[dim_along] = max(update, max_bounds[dim_along])
            return
        
        # Get the middle rectangle
        mid = (start + end) // 2
        indices = np.copy(fixed_indices)
        indices[dim_along] = mid
        rectangle = create_hyper_rectangle(scores_sorted, indices)

        # Compute the threshold along dim_along
        clipped_mean = mean_clip(rectangle, scores_mean)
        upperbound = scaled_upperbound(scores, alpha, 
                                           scores_mean, 
                                           scores_std,
                                           clipped_mean)
        L_temp = scaled_threshold(upperbound,
                                         scores_mean,
                                         scores_std,
                                         n, dim_along)

        # Update and search rule
        if L_temp >= rectangle.lower[dim_along]:
            update = min(L_temp, rectangle.upper[dim_along])
            max_bounds[dim_along] = max(update, max_bounds[dim_along])
            binary_search_dimension(fixed_indices, dim_along, max_bounds, mid, end)
        else:
            binary_search_dimension(fixed_indices, dim_along, max_bounds, start, mid)

    if short_cut:

        # Initialize the boundary and get the starting rectangle (mean rectangle)
        max_bounds = np.zeros(d)
        mean_rectangle = create_hyper_rectangle(scores_sorted, mean_index)
        mean_upperbound = scaled_upperbound(scores, alpha, scores_mean, scores_std, scores_mean)

        # Perform coordinte-wise binary search
        for idx in range(d):
            L_temp = scaled_threshold(mean_upperbound,
                                      scores_mean,
                                      scores_std,
                                      n, idx)
            if L_temp >= mean_rectangle.lower[idx]:
                max_bounds[idx] = min(L_temp, mean_rectangle.upper[idx])
                binary_search_dimension(mean_index, idx, max_bounds, mean_index[idx], n + 1)
            else:
                current =  mean_index[idx]
                while current >= 2:
                    current = current - 1
                    indices = np.copy(mean_index)
                    indices[idx] = current
                    rectangle = create_hyper_rectangle(scores_sorted, indices)
                    clipped_mean = mean_clip(rectangle, scores_mean)
                    upperbound = scaled_upperbound(scores, alpha,
                                                scores_mean,
                                                scores_std,
                                                clipped_mean)
                    threshold = scaled_threshold(upperbound,
                                                scores_mean,
                                                scores_std, n)
                    region = Rectangle(upper=threshold)
                    intersection = region.intersection(rectangle)
                    if intersection is not None:
                        max_bounds = np.maximum(max_bounds, intersection.upper)
                        current = 0

                #max_bounds[idx] = L_temp
        return Rectangle(upper=max_bounds)
    
    else:

        regions = []
        max_bounds = np.zeros(d)
        upper = [scores_sorted[dim][math.ceil((1 - alpha) * (n + 1))] for dim in range(d)]
        rectangle = Rectangle(upper=upper)
        regions.append(rectangle)
        max_bounds = np.maximum(max_bounds,  rectangle.upper)
        for indices in product(range(1, n + 2), repeat=d):
            if np.all(np.array(indices) <= math.ceil((1 - alpha) * (n + 1))):
                continue
            else:
                rectangle = create_hyper_rectangle(scores_sorted, indices)
                clipped_mean = mean_clip(rectangle, scores_mean)
                upperbound = scaled_upperbound(scores, alpha,
                                               scores_mean,
                                               scores_std,
                                               clipped_mean)
                threshold = scaled_threshold(upperbound,
                                             scores_mean,
                                             scores_std, n)
                region = Rectangle(upper=threshold)
                intersection = region.intersection(rectangle)
                if intersection:
                    regions.append(intersection)
                    max_bounds = np.maximum(max_bounds, intersection.upper)
        return regions, Rectangle(upper=max_bounds)
    
######################### Divider ######################### 

### Standardized prediction region
    
def standardized_upperbound(scores, alpha, mu, std, clipped_mean, global_const = None):
    """
    Compute a standardized upper bound quantile for the scores.

    This applies the `standardized_transformation` to the scores (optionally
    using a global constant correction), adds a tiny random jitter to break
    ties in a stable way, sorts the transformed scores, and returns the
    empirical (1 - alpha)-quantile.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) with calibration scores or residuals.
    alpha : float
        Miscoverage level in (0, 1). The routine targets the (1 - alpha)-quantile.
    mu : np.ndarray or float
        Mean(s) associated with the scores, e.g. per-dimension calibration means.
    std : np.ndarray or float
        Standard deviation(s) associated with the scores.
    clipped_mean : np.ndarray or float
        Mean vector (or scalar) after clipping to a rectangle, used to
        construct the imputed variance when `global_const` is not None.
    global_const : float or None, optional
        If not None, passed to `standardized_transformation` to produce a
        globally shifted upper bound; otherwise the local extremal version
        of the transformation is used.

    Returns
    -------
    float
        The (1 - alpha)-empirical quantile of the transformed scores, or
        np.inf if the requested order statistic exceeds the sample size.
    """
    n, d = scores.shape

    if global_const is not None:

        np.random.seed(42)
        scores_upperbound = standardized_transformation(scores, mu, std, clipped_mean, global_const) + 1e-10 * np.random.randint(n)
        scores_upperbound_sorted = np.sort(scores_upperbound, kind="mergesort")

    else:
        np.random.seed(42)
        scores_upperbound = standardized_transformation(scores, mu, std, clipped_mean) + 1e-10 * np.random.randint(n)
        scores_upperbound_sorted = np.sort(scores_upperbound, kind="mergesort")

    quantile_level = math.ceil((1 - alpha) * (n + 1))
    return scores_upperbound_sorted[quantile_level - 1] if quantile_level <= n else np.inf
            

def standardized_threshold(upperbound, mu, std, size, dim = None):
    """
    Translate a scalar standardized upper bound into a threshold on the
    original score scale, either per-dimension or jointly.

    This inverts the standardized transformation by solving a quadratic
    inequality, returning either a vector of thresholds (one per dimension)
    or a single scalar threshold for a specified coordinate.

    Parameters
    ----------
    upperbound : float
        Scalar upper bound on the standardized scale.
    mu : np.ndarray or float
        Mean(s) used in the standardization. If `dim` is None, this is
        assumed to be an array; otherwise, a single coordinate is used.
    std : np.ndarray or float
        Standard deviation(s) used in the standardization.
    size : int
        Sample size n used in the construction (denoted as `size` here).
    dim : int or None, optional
        If None, compute thresholds for all dimensions and return an array.
        If an integer, compute the threshold only for the specified dimension
        and return a scalar.

    Returns
    -------
    np.ndarray or float
        Threshold(s) on the original score scale. If `dim` is None, an array
        of shape (d,) is returned; otherwise a scalar is returned. Zeros or
        +inf are returned in regimes where the quadratic has no admissible
        solution.
    """
    corr = size + 1
    control = size**2 / corr - upperbound**2

    mu = mu if dim is None else mu[dim]
    std = std if dim is None else std[dim]
    zeros = np.zeros(len(std)) if dim is None else 0
    infs = np.repeat(np.inf, mu.shape[0]) if dim is None else np.inf

    # G > 0, quadratic, two roots
    if control > 0:
        
        larger_root = mu + std * upperbound * np.sqrt(corr / control)
        smaller_root = mu - std * np.abs(upperbound) * np.sqrt(corr / control)

        return larger_root if upperbound >= 0 else np.maximum(zeros, smaller_root)
                
    # G <= 0, quadratic, no solution
    else:
        return zeros if upperbound < 0 else infs


def standardized_prediction(scores, alpha=0.2, method = "LWC", short_cut=True):
    """
    Construct a conformal prediction region using standardized rectangular
    partitioning and (optionally) a local-worst-case refinement.

    Depending on the `method` and `short_cut` flag, this either returns a
    single global rectangle (GWC) or a more refined LWC region, possibly
    as a union of rectangles plus a global bounding box.

    Parameters
    ----------
    scores : np.ndarray
        Array of shape (n, d) containing calibration scores or residuals.
    alpha : float, default=0.2
        Miscoverage level in (0, 1) for the conformal region.
    method : {"LWC", "GWC"}, default="LWC"
        Type of construction:
        - "GWC": global-worst-case rectangle only.
        - "LWC": local-worst-case refinement of the global rectangle.
    short_cut : bool, default=True
        If True, uses a coordinate-wise binary search shortcut to construct
        a single LWC rectangle. If False, enumerates a collection of
        candidate rectangles and returns their union along with an overall
        bounding rectangle.

    Returns
    -------
    Rectangle or (List[Rectangle], Rectangle)
        If `method == "GWC"`, a single global Rectangle.
        If `method == "LWC"` and `short_cut` is True, a single refined
        Rectangle.
        If `method == "LWC"` and `short_cut` is False, a tuple:
        (list_of_rectangles, bounding_rectangle).
    """
    # Compute shape, mean, std, and mean indices
    n, d = scores.shape
    scores_mean = np.mean(scores, axis = 0)
    scores_std = np.std(scores, axis = 0)
    mean_index = mean_index_solver(scores)+1

    # Append zero and inf to get ready for partition
    scores_augmented = np.append(scores, [np.repeat(np.inf, d), np.zeros(d)], axis=0)
    scores_sorted = np.transpose(np.sort(scores_augmented, axis=0, kind="mergesort"))

    # Global region
    global_uppberbound = standardized_upperbound(scores, alpha,
                                                 scores_mean,
                                                 scores_std,
                                                 scores_mean)
    global_threshold = standardized_threshold(global_uppberbound,
                                              scores_mean,
                                              scores_std, n)
    global_rectangle = Rectangle(upper=global_threshold)

    # GWC method
    if method == "GWC":
        return global_rectangle
    
    zero = (n*scores_mean/(n+1)) / np.sqrt(scores_mean**2/(n+1) + scores_std**2)
    bound = np.where(global_threshold == np.inf, 1/np.sqrt(n+1), ((n*scores_mean+global_threshold)/(n+1)) / np.sqrt((global_threshold - scores_mean)**2/(n+1) + scores_std**2))
    global_const = np.minimum(zero, bound)

    # Perforem binary search along dim_along
    def binary_search_dimension(fixed_indices, dim_along, max_bounds, start, end):

        # Edge case: the last rectangle along dim_along to be evaluated
        if start >= end-1:

            # Get the last rectangle
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(scores_sorted, indices).intersection(global_rectangle)
            if rectangle is None:
                return
            
            # Compute the threshold along dim_along
            clipped_mean = mean_clip(rectangle, scores_mean)
            upperbound = standardized_upperbound(scores, alpha,
                                                 scores_mean,
                                                 scores_std,
                                                 clipped_mean,
                                                 global_const)
            L_temp = standardized_threshold(upperbound,
                                            scores_mean,
                                            scores_std,
                                            n, dim_along)
            # Update rule
            if L_temp >= rectangle.lower[dim_along]:
                update = min(L_temp, rectangle.upper[dim_along])
                max_bounds[dim_along] = max(update, max_bounds[dim_along])
            return
        
        # Get the middle rectangle
        mid = (start + end) // 2
        indices = np.copy(fixed_indices)
        indices[dim_along] = mid
        rectangle = create_hyper_rectangle(scores_sorted, indices).intersection(global_rectangle)

        if rectangle is not None:

            # Compute the threshold along dim_along
            clipped_mean = mean_clip(rectangle, scores_mean)
            upperbound = standardized_upperbound(scores, alpha,
                                                 scores_mean,
                                                 scores_std,
                                                 clipped_mean,
                                                 global_const)
            L_temp = standardized_threshold(upperbound,
                                            scores_mean,
                                            scores_std,
                                            n, dim_along)
            # Update and search rule
            if L_temp >= rectangle.lower[dim_along]:
                update = min(L_temp, rectangle.upper[dim_along])
                max_bounds[dim_along] = max(update, max_bounds[dim_along])
                binary_search_dimension(fixed_indices, dim_along, max_bounds, mid, end)
            else:
                binary_search_dimension(fixed_indices, dim_along, max_bounds, start, mid)
        else:
            binary_search_dimension(fixed_indices, dim_along, max_bounds, start, mid)

    if short_cut:

        # Initialize the boundary and get the starting rectangle (mean rectangle)
        mean_rectangle = create_hyper_rectangle(scores_sorted, mean_index)
        if mean_rectangle.intersection(global_rectangle) is None:
            return global_rectangle

        # Start searching, compute the quantile corresponds to the mean rectangle first
        max_bounds = np.zeros(d)
        mean_rectangle = mean_rectangle.intersection(global_rectangle)
        mean_clipped_mean = mean_clip(mean_rectangle, scores_mean)
        mean_upperbound = standardized_upperbound(scores, alpha,
                                                 scores_mean,
                                                 scores_std,
                                                 mean_clipped_mean,
                                                 global_const)

        for idx in range(d):
            
            L_temp = standardized_threshold(mean_upperbound,
                                            scores_mean,
                                            scores_std,
                                            n, idx)

            # Binary search
            if L_temp >= mean_rectangle.lower[idx]:
                max_bounds[idx] = min(L_temp, mean_rectangle.upper[idx])
                binary_search_dimension(mean_index, idx, max_bounds, mean_index[idx], n + 1)

            # Backward search
            else:
                current = mean_index[idx]
                while current >= 2:
                    current = current - 1
                    indices = np.copy(mean_index)
                    indices[idx] = current
                    rectangle = create_hyper_rectangle(scores_sorted, indices).intersection(global_rectangle)
                    
                    if rectangle is not None:
                        clipped_mean = mean_clip(rectangle, scores_mean)
                        upperbound = standardized_upperbound(scores, alpha,
                                                    scores_mean,
                                                    scores_std,
                                                    clipped_mean)
                        L_temp = standardized_threshold(upperbound,
                                                    scores_mean,
                                                    scores_std, n, idx)
                        if L_temp > rectangle.lower[idx]:
                            max_bounds[idx] = min(L_temp, rectangle.upper[idx])
                            current = 0
                    else:
                        continue
        return Rectangle(upper=max_bounds)
    else:

        regions = []
        max_bounds = np.zeros(d)
        upper = [scores_sorted[dim][math.ceil((1 - alpha) * (n + 1))] for dim in range(d)]
        rectangle = Rectangle(upper=upper)
        regions.append(rectangle)
        max_bounds = np.maximum(max_bounds,  rectangle.upper)
        for indices in product(range(1, n + 2), repeat=d):
            if np.all(np.array(indices) <= math.ceil((1 - alpha) * (n + 1))):
                continue
            else:
                rectangle = create_hyper_rectangle(scores_sorted, indices).intersection(global_rectangle)
                if rectangle is not None:
                    clipped_mean = mean_clip(rectangle, scores_mean)
                    upperbound = standardized_upperbound(scores, alpha,
                                                scores_mean,
                                                scores_std,
                                                clipped_mean,
                                                global_const)
                    threshold = standardized_threshold(upperbound,
                                                scores_mean,
                                                scores_std, n)
                    region = Rectangle(upper=threshold)
                    intersection = region.intersection(rectangle)
                    if intersection is not None:
                        regions.append(intersection)
                        max_bounds = np.maximum(max_bounds, intersection.upper)
        return regions, Rectangle(upper=max_bounds)

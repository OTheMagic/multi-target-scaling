# Import necessary packages
import numpy as np
import math
from itertools import combinations, product
from utility.rectangle import Rectangle
from typing import Union, List, Tuple

### Basic handling tools

def mean_index_solver(scores: np.ndarray) -> np.ndarray:
    """
    Find the index (in sorted order) of the value just above the mean along each dimension.

    Parameters
    ----------
    scores : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_features,)
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
    upper = [scores_sorted[dim][indices[dim]] for dim in range(len(indices))]
    lower = [scores_sorted[dim][indices[dim] - 1] for dim in range(len(indices))]
    return Rectangle(upper, lower)

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

def mean_clip(rectangle, mean_vector):
    """
    Determine min projection points from mean to rectangle edges.

    Parameters
    ----------
    rectangle : Rectangle
        Object with .lower and .upper arrays of shape (d,)
    mean_vector : np.ndarray, shape (d,)
        Mean of the calibration scores.

    Returns
    -------
    np.ndarray
        point_min (projection of mean_vector onto rectangle)
    """
    return np.clip(mean_vector, rectangle.lower, rectangle.upper)

######################### Divider ######################### 

### Pointwise upperbounds

def scaled_transformation(scores, mu, std, clipped_mean):

    imputed_var = (clipped_mean - mu)**2 / (scores.shape[0] + 1) + std**2
    imputed_std = np.sqrt(imputed_var)
    upperbound = scores / imputed_std
    return np.max(upperbound, axis=1)

def standardized_transformation(scores, mu, std, clipped_mean, global_const = None):

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
        t3 = np.where(scores_centered != 0, t3_num/t3_den*sign, t2)

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

def standardized_prediction(scores, alpha=0.2, short_cut=True) -> Union[Rectangle, Tuple[List[Rectangle], Rectangle]]:
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

    # Global region
    global_uppberbound = standardized_upperbound(scores, alpha,
                                                 scores_mean,
                                                 scores_std,
                                                 scores_mean)
    global_threshold = standardized_threshold(global_uppberbound,
                                              scores_mean,
                                              scores_std, n)
    global_rectangle = Rectangle(upper=global_threshold)
    
    zero = (n*scores_mean/(n+1)) / np.sqrt(scores_mean**2/(n+1) + scores_std**2)
    bound = np.where(global_threshold == np.inf, 1/np.sqrt(n+1), ((n*scores_mean+global_threshold)/(n+1)) / np.sqrt((global_threshold - scores_mean)**2/(n+1) + scores_std**2))
    global_const = np.minimum(zero, bound)

    # Perforem binary search along dim_along
    def binary_search_dimension(fixed_indices, dim_along, max_bounds, start, end):

        # Edge case: the last rectangle along dim_along to be evaluated
        if start >= end - 1:

            # Get the last rectangle
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(scores_sorted, indices)
            if global_threshold[dim_along] <= rectangle.lower[dim_along]:
                return
            else:
                rectangle.update_upper(min(global_threshold[dim_along], rectangle.upper[dim_along]), dim_along)
            
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
        rectangle = create_hyper_rectangle(scores_sorted, indices)

        if global_threshold[dim_along] >= rectangle.lower[dim_along]:
            rectangle.update_upper(min(global_threshold[dim_along], rectangle.upper[dim_along]), dim_along)

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
        max_bounds = np.zeros(d)
        mean_rectangle = create_hyper_rectangle(scores_sorted, mean_index).intersection(global_rectangle)
        if mean_rectangle is None:
            return global_rectangle

        # Perform coordinte-wise binary search
        for idx in range(d):
            if global_threshold[idx] >= mean_rectangle.lower[idx]:
                max_bounds[idx] = min(global_threshold[idx], mean_rectangle.upper[idx])
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
                    if intersection:
                        regions.append(intersection)
                        max_bounds = np.maximum(max_bounds, intersection.upper)
        return regions, Rectangle(upper=max_bounds)
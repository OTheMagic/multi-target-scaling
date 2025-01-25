import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations, product
from sklearn.model_selection import train_test_split
from rectangle import Rectangle

def calibration_split(X, y, test_cal_size=0.2, cal_size=0.5, random_state=42):
    """
    Split the dataset into training, testing, and calibration sets in specified proportions.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target matrix of shape (n_samples,).
    test_cal_size : float, optional
        Fraction of the dataset to include in the combined (test + calibration) subset. Default: 0.2.
    cal_size : float, optional
        Fraction of the combined (test + calibration) subset to allocate for calibration. Default: 0.5.
    random_state : int, optional
        Seed for random number generation to ensure reproducible results. Default: 42.

    Returns
    -------
    tuple
        X_train, X_test, X_cal, y_train, y_test, y_cal : numpy.ndarray
        Training, test, and calibration splits of the features and targets.
    """
    # Split into training and combined test/calibration set
    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, test_size=test_cal_size, random_state=random_state)
    
    # Further split the test/calibration set into test and calibration subsets
    X_test, X_cal, y_test, y_cal = train_test_split(X_test_cal, y_test_cal, test_size=cal_size, random_state=random_state)

    return X_train, X_test, X_cal, y_train, y_test, y_cal

def make_2D_score_plot(scores, dimx, dimy, ax = None, figsize=(8, 8), limits = None):
    """
    Create a 2D scatter plot for a specific pair of dimensions.

    Parameters
    ----------
    scores : numpy.ndarray
        Matrix of shape (n_samples, n_features) representing the data to plot.
    dimx : int
        Index of the dimension for the x-axis.
    dimy : int
        Index of the dimension for the y-axis.
    figsize : tuple, optional
        Size of the figure in inches. Default: (8, 8).

    Returns
    -------
    tuple
        fig, ax : matplotlib Figure and Axes object.
    """

    n = scores.shape[1]
    if not (0 <= dimx < n and 0 <= dimy < n):
        raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional data.")
    if dimx == dimy:
        raise ValueError("dimx and dimy must be different to form a valid 2D projection.")

    scores_transpose = np.transpose(scores)
    if limits is None:
        limits = np.max(scores[:, [dimx, dimy]], axis = 0)*1.1

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(scores_transpose[dimx], scores_transpose[dimy], s=1)
        ax.set_xlim(0, limits[0])
        ax.set_ylim(0, limits[1])
        ax.set_aspect('equal', adjustable = "box")
        ax.set_xlabel(f"Dimension {dimx}")
        ax.set_ylabel(f"Dimension {dimy}")
        ax.set_title(f"Projection on dims ({dimx}, {dimy})")
        plt.tight_layout()
        return fig, ax
    
    ax.scatter(scores_transpose[dimx], scores_transpose[dimy], s=1)
    ax.set_xlim(0, limits[0])
    ax.set_ylim(0, limits[1])
    ax.set_aspect('equal')
    ax.set_xlabel(f"Dimension {dimx}")
    ax.set_ylabel(f"Dimension {dimy}")
    ax.set_title(f"Projection on dims ({dimx}, {dimy})")
    plt.tight_layout()

def make_score_plot(scores, global_limit = True):
    """
    Create scatter plots of pairwise dimensions of vectorized scores.

    Parameters
    ----------
    scores : numpy.ndarray
        Matrix of shape (n_samples, n_features) representing the data to plot.
    dimx : int, optional
        Index of the dimension for the x-axis. Default: None (all pairs).
    dimy : int, optional
        Index of the dimension for the y-axis. Default: None (all pairs).
    figsize : tuple, optional
        Size of the figure in inches. Default: (12, 12).

    Returns
    -------
    tuple
        fig, axes : matplotlib Figure and Axes object(s).
    """

    n = scores.shape[1]
    # All pairwise projections
    dim_pairs = list(combinations(range(n), 2))  # Unique pairs of dimensions
    num_pairs = len(dim_pairs)

    # Determine grid layout
    cols = min(3, num_pairs)
    rows = math.ceil(num_pairs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()  # Flatten for easy iteration

    for idx, (dx, dy) in enumerate(dim_pairs):
        if global_limit:
            global_limits = [np.max(scores) * 1.1, np.max(scores) * 1.1]
            make_2D_score_plot(scores, dx, dy, axes[idx], limits = global_limits)
        else:
            make_2D_score_plot(scores, dx, dy, axes[idx])

    # Hide unused subplots
    for idx in range(num_pairs, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    return fig, axes

def scale_with_sampling(scores, rectangle, point = None):
    """
    Scales scores by appending a randomly sampled point from a rectangle region.

    Parameters
    ----------
    scores : numpy.ndarray
        2D array of shape (n_samples, n_features) containing the original scores.
    rectangle : Rectangle
        Defines the sampling range with `lower` and `upper` bounds.
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple
        - Scaled scores as a 2D numpy array.
        - Standard deviation of the scaled data along each feature.
    """
    sampled_point = np.array([point]) if point is not None else np.array([rectangle.lower])
    scores_new = np.append(scores, sampled_point, axis=0)
    std_dev = np.std(scores_new, axis=0)
    
    scaled_scores = scores / std_dev
    
    return scaled_scores, std_dev

def compute_prediction_region(scores, alpha, rect_to_scale, point = None):
    """
    Scales scores and computes a prediction region using the quantile threshold.
    
    Parameters
    ----------
    scores : numpy.ndarray
        Array of input scores.
    alpha : float
        Significance level for quantile computation.
    rect_to_scale : Rectangle
        Rectangle for scaling operations.

    Returns
    -------
    Rectangle
        A rectangle representing the prediction region.
    """
    scores_scaled, scale = scale_with_sampling(scores, rect_to_scale, point)
    max_norm_scaled = np.max(scores_scaled, axis=1)

    n = len(scores)
    quantile_level = math.ceil((1 - alpha) * (n + 1)) / n
    quantile_val = np.quantile(max_norm_scaled, quantile_level)

    return Rectangle(quantile_val * scale)


def reorder_dimensions_by_variance(scores_transpose):
    """
    Reorders scores by decreasing variance along each dimension.
    
    Parameters
    ----------
    scores_transpose : numpy.ndarray
        Transposed scores array (features as rows).
    
    Returns
    -------
    numpy.ndarray
        Reordered scores.
    """
    variances = np.var(scores_transpose, axis=1)
    sorted_indices = np.argsort(variances)[::-1]
    return scores_transpose[sorted_indices, :]


def scores_completion_sorted(scores, descending = True):
    """
    Completes scores with additional points for prediction region computations.
    
    Parameters
    ----------
    scores : numpy.ndarray
        Original scores array.
    
    Returns
    -------
    numpy.ndarray
        Sorted and augmented scores.
    """
    unique_scores = np.unique(scores, axis=0)
    max_point = np.max(scores, axis=0) * 10
    augmented_scores = np.append(unique_scores, [max_point, np.zeros_like(max_point)], axis=0)
    if descending:
        return -np.sort(-augmented_scores, axis=0, kind="mergesort")
    return np.sort(augmented_scores, axis=0, kind="mergesort")

def full_prediction_regions_2D(scores, alpha, one_rect=True, short_cut=True):
    """
    Computes the full prediction regions in 2D.
    
    Parameters
    ----------
    scores : numpy.ndarray
        2D array of scores.
    alpha : float
        Significance level for region computation.
    one_rect : bool, optional
        Whether to compute a single rectangle or multiple regions. Defaults to True.
    short_cut : bool, optional
        Whether to use the shortcut optimization. Defaults to True.
    
    Returns
    -------
    Rectangle or list of Rectangles
        Prediction region(s) based on the input parameters.
    """
    scores_modified = scores_completion_sorted(scores, False)
    scores_sorted = reorder_dimensions_by_variance(np.transpose(scores_modified))
    
    n = len(scores)
    level = math.ceil((1 - alpha) * (n + 1))

    def create_rectangle(idx, idy):
        """Helper to create a Rectangle based on index ranges."""
        height, height_prev = scores_sorted[1][idy], scores_sorted[1][idy - 1]
        upper = (scores_sorted[0][idx], height)
        lower = (scores_sorted[0][idx - 1], height_prev)
        return Rectangle(upper, lower)

    def row_binary_search(idy, regions, start, end):
        """Performs binary search to compute intersecting regions."""
        if start == end:
            rectangle = create_rectangle(start, idy)
            region = compute_prediction_region(scores, alpha, rectangle)
            if region.intersection(rectangle):
                regions.append(region.intersection(rectangle))
            return

        mid = (start + end) // 2
        rectangle = create_rectangle(mid, idy)
        region = compute_prediction_region(scores, alpha, rectangle)
        intersection = region.intersection(rectangle)

        if not intersection:
            row_binary_search(idy, regions, start, mid)
        elif intersection.same_as(rectangle):
            row_binary_search(idy, regions, mid, end)
        else:
            new_lower = (scores_sorted[0][level], scores_sorted[1][idy - 1]) if idy <= level else (0, scores_sorted[1][idy - 1])
            regions.append(region.intersection(Rectangle(rectangle.upper, new_lower)))

    def col_binary_search(idx, regions, start, end):
        """Performs binary search to compute intersecting regions."""
        if start == end:
            rectangle = create_rectangle(idx, start)
            region = compute_prediction_region(scores, alpha, rectangle)
            if region.intersection(rectangle):
                regions.append(region.intersection(rectangle))
            return

        mid = (start + end) // 2
        rectangle = create_rectangle(idx, mid)
        region = compute_prediction_region(scores, alpha, rectangle)
        intersection = region.intersection(rectangle)

        if not intersection:
            col_binary_search(idx, regions, start, mid)
        elif intersection.same_as(rectangle):
            col_binary_search(idx, regions, mid, end)
        else:
            new_lower = (scores_sorted[0][idx - 1], scores_sorted[1][level]) if idx <= level else (scores_sorted[0][idx - 1], 0)
            regions.append(region.intersection(Rectangle(rectangle.upper, new_lower)))

    if short_cut:
        regions = []
        max_right, max_top = 0, 0
        for idy in range(1, n + 2):
            row_binary_search(idy, regions, level, n+1)
            last_region = regions[-1]
            max_right = max(max_right, last_region.upper[0])
        
        for idx in range(1, n+2):
            col_binary_search(idx, regions, level, n+1)
            last_region = regions[-1]
            max_top = max(max_top, last_region.upper[1])

        Inclusion = Rectangle((scores_sorted[0][level], scores_sorted[1][level]))
        regions.append(Inclusion)     
        if one_rect:
            return Rectangle(upper=(max_right, max_top))
        return regions
    else:
        regions = []
        max_right, max_top = 0, 0
        for idy in range(1,n+2):
            for idx in range(1, n+2):
                rectangle = create_rectangle(idx, idy)
                region = compute_prediction_region(scores, alpha, rectangle)
                intersection = region.intersection(rectangle)
                if intersection:
                        regions.append(intersection)
                        max_right = max(max_right, intersection.upper[0])
                        max_top = max(max_top, intersection.upper[1])
        if one_rect:
            return Rectangle(upper=(max_right, max_top))
        return regions
    
def one_rect_prediction_regions_nD(scores, alpha = 0.2, short_cut = True):

    # Sort the transposed scores from smallest to largest coordinate-wise 
    scores_sorted = np.transpose(scores_completion_sorted(scores, False))
    
    # Number of samples, number of dimensions, conformal index
    n = scores.shape[0]
    d = scores.shape[1]

    # Grab the rectangle that contains the mean
    scores_mean = np.mean(np.transpose(scores), axis=1)
    abs_diff = np.abs(np.transpose(np.array([scores_mean])) - np.sort(np.transpose(scores), axis=1, kind="mergesort"))
    mask = (np.transpose(np.array([scores_mean])) - np.sort(np.transpose(scores), axis=1, kind="mergesort")) > 0
    abs_diff[mask] = np.inf 
    mean_index = np.argmin(abs_diff, axis=1)+1

    def create_hyper_rectangle(indices):

        upper = [scores_sorted[dim][indices[dim]] for dim in range(d)]
        lower = [scores_sorted[dim][indices[dim]-1] for dim in range(d)]

        return Rectangle(upper, lower)
    
    def binary_search_dimension(scores_sorted, fixed_indices, point, dim_along, regions, start, end):

        fix_point = np.copy(point)
        if start >= end-1:
            indices = np.copy(fixed_indices)
            indices[dim_along] = end
            rectangle = create_hyper_rectangle(indices)
            fix_point[dim_along] = rectangle.lower[dim_along]
            region = compute_prediction_region(scores, alpha, rectangle, fix_point)
            if region.intersection(rectangle):
                regions.append(region.intersection(rectangle))
            return

        mid = (start + end) // 2
        indices = np.copy(fixed_indices)
        indices[dim_along] = mid
        rectangle = create_hyper_rectangle(indices)
        fix_point[dim_along] = rectangle.lower[dim_along]
        region = compute_prediction_region(scores, alpha, rectangle, fix_point)
        intersection = region.intersection(rectangle)

        if not intersection:
            binary_search_dimension(scores_sorted, fixed_indices, point, dim_along, regions, start, mid)
        elif intersection.same_as(rectangle):
            binary_search_dimension(scores_sorted, fixed_indices, point, dim_along, regions, mid, end)
        else:
            #new_lower = [scores_sorted[dim][level] if dim == dim_along else scores_sorted[dim][fixed_indices[dim] - 1] for dim in range(d)]
            regions.append(intersection)

    if short_cut:
        regions = []
        max_bounds = np.zeros(d)
        for idx in range(d):
            fix_indices = np.copy(mean_index)
            fix_indices[idx] = 1
            binary_search_dimension(scores_sorted, fix_indices, scores_mean, idx, regions, 1, n+1)
            max_bounds = np.maximum(max_bounds, regions[-1].upper)

        return Rectangle(upper=max_bounds)
    else:   

        regions = []
        max_bounds = np.zeros(d, dtype=float)
        for indices in product(range(1, n+2), repeat=d):

            indices = np.copy(indices)
            rectangle = create_hyper_rectangle(indices)

            point = np.zeros(d)
            for i in range(d):
                point[i] = scores_mean[i] if indices[i] == mean_index[i] else rectangle.lower[i]
                    

            region = compute_prediction_region(scores, alpha, rectangle, point)


            intersection = region.intersection(rectangle)
            if intersection:

                regions.append(intersection)

                max_bounds = np.maximum(max_bounds, intersection.upper)

        return regions, Rectangle(upper=max_bounds)

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
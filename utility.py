import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from itertools import combinations
from sklearn.model_selection import train_test_split
from rectangle import Rectangle

def calibration_split(X, y, test_cal_size = 0.2, cal_size = 0.5, random_state = 42):
    """
    Split the dataset into training, testing, and calibration sets in given proportions.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : numpy.ndarray
        Target matrix of shape (n_samples, n_targets).
    test_cal_size : float, optional
        The fraction of the dataset to include in the combined (test+calibration) subset.
        Default: 0.2
    cal_size : float, optional
        The fraction of the combined (test+calibration) subset to allocate for calibration.
        Default: 0.5
    random_state : int, optional
        Controls the shuffling applied to the data before splitting. Pass an int for
        reproducible output. Default: 42

    Returns
    -------
    X_train : numpy.ndarray
        Training feature matrix.
    X_test : numpy.ndarray
        Test feature matrix.
    X_cal : numpy.ndarray
        Calibration feature matrix.
    y_train : numpy.ndarray
        Training target array.
    y_test : numpy.ndarray
        Test target array.
    y_cal : numpy.ndarray
        Calibration target array.
    """

    X_train, X_test_cal, y_train, y_test_cal = train_test_split(X, y, test_size=test_cal_size, random_state=random_state)
    X_test, X_cal, y_test, y_cal = train_test_split(X_test_cal, y_test_cal, test_size=cal_size, random_state=random_state)

    return X_train, X_test, X_cal, y_train, y_test, y_cal

def make_score_plot(scores, dimx = None, dimy = None, figsize = (12, 12)):
    '''
    Create subplots of pairwise dimensions of the vectorized scores.

    Parameters
    ----------
    scores: numpy.ndarray
        Matrix of shape (n_samples, n_features)
    dimx : int, optional
        Dimension index for the horizontal axis. Default: None
    dimy : int, optional
        Dimension index for the vertical axis. Default: None
    
    Returns
    -------
    fig, axes : matplotlib Figure and Axes object(s)
        The figure and array of axes used for plotting.
    '''

    n = scores.shape[1]
    scores_transpose = np.transpose(scores)

    # Case 1: A single pair (dimx, dimy) 
    if dimx is not None and dimy is not None:

        dimension_slice = scores_transpose[[dimx, dimy],:]
        local_min = 0
        local_max = np.max(dimension_slice, axis=1)*1.1

        if not (0 <= dimx < n and 0 <= dimy < n):
            raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
        if dimx == dimy:
            raise ValueError("dimx and dimy must be different to form a valid 2D projection.")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.scatter(scores_transpose[dimx], scores_transpose[dimy], s = 1)
        ax.set_xlim(local_min, local_max[0])
        ax.set_ylim(local_min, local_max[1])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"Dimension {dimx}")
        ax.set_ylabel(f"Dimension {dimy}")
        ax.set_title(f"Projection on dims ({dimx}, {dimy})")
        plt.tight_layout()
        #plt.show()
        return fig, ax

    # Case 2: All pairwise projections 
    elif dimx is None and dimy is None:

        global_min = 0
        global_max = np.max(scores)*1.1

        # Helper function to draw 2D plot
        def draw_2D(ax, dx, dy):
            ax.scatter(scores_transpose[dx], scores_transpose[dy], s = 1)
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f"Dimension {dx}")
            ax.set_ylabel(f"Dimension {dy}")
            ax.set_title(f"Projection on dims ({dx}, {dy})")

        dim_pairs = list(combinations(range(n), 2))  # all unique pairs
        num_plots = len(dim_pairs)

        # Decide on a subplot grid layout
        cols = min(num_plots, 3)
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
        axes = axes.flatten()  # Flatten for easy indexing

        for idx, (dx, dy) in enumerate(dim_pairs):
            ax = axes[idx]
            draw_2D(ax, dx, dy)

        # Hide any unused axes (if num_plots < rows*cols)
        for j in range(num_plots, rows*cols):
            fig.delaxes(axes[j])

        plt.tight_layout()
        #plt.show()
        return fig, axes
    
     # If only one of dimx or dimy is specified, raise an error
    else:
        raise ValueError(
            "You must either provide both dimx and dimy for a single plot "
            "or leave both as None to plot all dimension pairs."
        )

def scale_with_sampling(scores, rectangle, exclusion_rectangle=None, random_state=42):
    """
    Scales scores by appending a randomly sampled point from a rectangle region.

    Parameters
    ----------
    scores : numpy.ndarray
        2D array of shape (n_samples, n_features) containing the original scores.
    rectangle : Rectangle
        Defines the sampling range with `lower` and `upper` bounds.
    exclusion_rectangle : Rectangle, optional
        Alters the lower boundary of the sampling region if it intersects `rectangle`.
    random_state : int, optional
        Seed for the random number generator.

    Returns
    -------
    tuple
        - Scaled scores as a 2D numpy array.
        - Standard deviation of the scaled data along each feature.
    """
    np.random.seed(random_state)
    
    if exclusion_rectangle and rectangle.intersects(exclusion_rectangle):
        lower, upper = exclusion_rectangle.upper, rectangle.upper
    else:
        lower, upper = rectangle.lower, rectangle.upper

    sampled_point = np.random.uniform(lower, upper, size=(1, scores.shape[1]))
    scores_new = np.append(scores, sampled_point, axis=0)
    
    std_dev = np.std(scores_new, axis=0)
    scaled_scores = scores / std_dev
    
    return scaled_scores, std_dev


def compute_prediction_region(scores, alpha, rect_to_scale, rect_exclusion=None):
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
    rect_exclusion : Rectangle, optional
        Exclusion rectangle.

    Returns
    -------
    Rectangle
        A rectangle representing the prediction region.
    """
    scores_scaled, scale = scale_with_sampling(scores, rect_to_scale, rect_exclusion)
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


def scores_completion_sorted(scores):
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
    max_point = np.max(scores, axis=0) * 1.2
    augmented_scores = np.append(unique_scores, [max_point, np.zeros_like(max_point)], axis=0)
    return np.sort(augmented_scores, axis=0)

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
    scores_modified = scores_completion_sorted(scores)
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
        if one_rect:
            max_right, max_top = 0, 0
            for idy in range(1,n+2):
                max_idx = 1
                for idx in range(max_idx, n+2):
                    rectangle = create_rectangle(idx, idy)
                    region = compute_prediction_region(scores, alpha, rectangle)
                    intersection = region.intersection(rectangle)
                    if intersection:
                            max_right = max(max_right, intersection.upper[0])
                            max_top = max(max_top, intersection.upper[1])
            return Rectangle(upper=(max_right, max_top))
        
        regions = []
        for idy in range(1,n+2):
            for idx in range(1, n+2):
                rectangle = create_rectangle(idx, idy)
                region = compute_prediction_region(scores, alpha, rectangle)
                intersection = region.intersection(rectangle)
                if intersection:
                        regions.append(intersection)
        return regions


def check_coverage_rate(scores, regions, one_rect = True):
    evaluation = np.zeros(len(scores))
    if one_rect == True:
        evaluation += regions.contain_points(scores).astype(int)
    else:
        for region in regions:
            evaluation += region.contain_points(scores).astype(int)
    return np.sum(evaluation > 0)/(len(scores))
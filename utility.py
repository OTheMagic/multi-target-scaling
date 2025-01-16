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
    Perform a scaling operation on the input scores by appending a randomly sampled point 
    from a specified rectangle region, optionally accounting for an exclusion rectangle.

    Parameters
    ----------
    scores : numpy.ndarray
        A 2D array of shape (n_samples, n_features) containing the original scores.
    rectangle : Rectangle
        A hyper-rectangle object with `lower` and `upper` attributes used to determine
        the sampling range.
    exclusion_rectangle : Rectangle, optional
        Another hyper-rectangle that, if it intersects with `rectangle`, will alter the 
        lower boundary of the sampling region. Defaults to None.
    random_state : int, optional
        Seed for the random number generator. Defaults to 42.

    Returns
    -------
    numpy.ndarray
        A 2D array of the same shape as `scores` where each value is scaled by the 
        standard deviation of the augmented data after sampling.
    """
    
    
    if (exclusion_rectangle is not None) and (rectangle.intersects(exclusion_rectangle)):
        lower = exclusion_rectangle.upper
        upper = rectangle.upper
    else:
        lower = rectangle.lower
        upper = rectangle.upper

    np.random.seed(random_state)
    point = np.array([np.random.uniform(lower, upper)])
    scores_new = np.append(scores, point, axis=0)
    scores_transpose = np.transpose(scores_new)
    return np.transpose(np.transpose(scores) / np.transpose(np.array([np.std(scores_transpose, axis = 1)]))), np.std(scores_transpose, axis = 1)

def compute_prediction_region(scores, alpha, rect_to_scale, rect_exclusion=None):
    """
    Helper function to:
      1) Scale 'scores' by sampling from 'rect_to_scale', optionally excluding 'rect_exclusion'.
      2) Compute the sup norm of the scaled data.
      3) Determine the quantile threshold and create a Rectangle(quantile * scale).
    """
    # Scale the scores
    scores_scaled, scale = scale_with_sampling(scores, rect_to_scale, rect_exclusion)

    # Compute the sup norm (max along each row)
    max_norm_scaled = np.max(scores_scaled, axis=1)

    # Determine the quantile level
    n = len(scores)
    level = math.ceil((1 - alpha) * (n + 1)) / n
    quantile_val = np.quantile(max_norm_scaled, level)

    # Return the newly created Rectangle
    return Rectangle(quantile_val * scale)

# Functions for the point method
def full_prediction_regions_2D(scores, alpha, drawings = True, short_cut = True):
    '''
    Computes the full prediction regions in 2-dim
    '''
    regions = []
    temp = np.append(scores, np.array([np.max(scores, axis=0)*1.2]), axis = 0)
    scores_modified = np.append(temp, np.array([np.zeros(2)]), axis = 0)
    scores_sorted = np.sort(np.transpose(scores_modified), axis = 1, kind = "mergesort") #O(nlogn)
    n = len(scores_modified)
    level = math.ceil((1-alpha)*(n-1))

    if short_cut == True:

        # Helper function to perform binary-sort like searching in each row
        def binary_sort_row(idy, start, end):
            '''Time complexity: O(logn)'''
            height = scores_sorted[1][idy]
            height_prev = scores_sorted[1][idy-1]
            if end == start+1:
                upper = (scores_sorted[0][start], height)
                lower = (scores_sorted[0][start-1], height_prev)
                rectangle = Rectangle(upper, lower)
                region = compute_prediction_region(scores, alpha, rectangle)
                if region.intersection(rectangle) is not None:
                   regions.append(region.intersection(rectangle))
            else: 
                mid = (end+start)//2
                upper = (scores_sorted[0][mid], height)
                lower = (scores_sorted[0][mid-1], height_prev)
                rectangle = Rectangle(upper, lower)
                region = compute_prediction_region(scores, alpha, rectangle)

                # Case 1: no intersection found
                if region.intersection(rectangle) is None:
                    binary_sort_row(idy, start, mid)
                # Case 2: full intersection
                elif region.intersection(rectangle).same_as(rectangle):
                    binary_sort_row(idy, mid, end)
                # Case 3: this is exactly the edge
                else:
                    if idy <= level:
                        new_lower = (scores_sorted[0][level], height_prev)
                    else:
                        new_lower = (0, height_prev)
                    rectangle_new = Rectangle(upper, new_lower)
                    regions.append(region.intersection(rectangle_new))

        # O(nlogn)
        for idy in range(1,n):
            if idy <= level:
                binary_sort_row(idy, level+1, n-1)
            else:
                binary_sort_row(idy, 1, n-1)
        
        Inclusion = Rectangle((scores_sorted[0][level], scores_sorted[1][level]))
        regions.append(Inclusion)     
    else:
        for idy in range(1,n):
            fix = scores_sorted[1][idy]
            fix_prev = scores_sorted[1][idy-1]
            for idx in range(1, n):
                upper = (scores_sorted[0][idx],fix)
                lower = (scores_sorted[0][idx-1],fix_prev)
                rectangle = Rectangle(upper, lower)
                region = compute_prediction_region(scores, alpha, rectangle)
                if region.intersection(rectangle) is not None:
                        regions.append(region.intersection(rectangle))

    return regions

def check_coverage_rate(scores, regions):
    evaluation = np.zeros(len(scores))
    for region in regions:
        evaluation += region.contain_points(scores).astype(int)
    return np.sum(evaluation)/(len(scores))
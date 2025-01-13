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

def make_score_plot(scores, dimx = None, dimy = None):
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
        local_max = np.max(dimension_slice)*1.1

        if not (0 <= dimx < n and 0 <= dimy < n):
            raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
        if dimx == dimy:
            raise ValueError("dimx and dimy must be different to form a valid 2D projection.")
        
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.scatter(scores_transpose[dimx], scores_transpose[dimy], s = 1)
        ax.set_xlim(local_min, local_max)
        ax.set_ylim(local_min, local_max)
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
        plt.show()
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
    
    np.random.seed(random_state)
    if (exclusion_rectangle is not None) and (rectangle.intersects(exclusion_rectangle)):
        lower = exclusion_rectangle.upper
        upper = rectangle.upper
    else:
        lower = rectangle.lower
        upper = rectangle.upper
    
    point = np.array([np.random.uniform(lower, upper)])
    scores_new = np.append(scores, point, axis=0)
    scores_transpose = np.transpose(scores_new)
    return np.transpose(np.transpose(scores) / np.transpose(np.array([np.std(scores_transpose, axis = 1)]))), np.std(scores_transpose, axis = 1)

# Functions for the point method
def rectangle_point_region(scores):
    '''
    Disect the space of calibration scores into disjoint rectangles according to coordinate-wise position.
    '''



# Functions for the norm method
def rectangle_norm_region(scores):
    """
    Dissect the space of calibration scores into sorted 1D rectangles based on the sup norm.

    Parameters
    ----------
    scores : numpy.ndarray
        A 2D array of shape (n_samples, n_features). Each row contains calibration scores
        for one sample. The sup norm is computed as the maximum value in each row.

    Returns
    -------
    rectangles : list of Rectangle
        A list of `Rectangle` objects, where each rectangle corresponds to one row's sup norm.
        The list is sorted in ascending order by the sup norm of each row.
    """

    n = scores.shape[1] # get the dimension of the rectangles
    max_norm = np.sort(np.max(scores, axis=1))
    rectangles = []
    for norm in max_norm:
        rectangle = Rectangle(np.repeat(norm, n))
        rectangles.append(rectangle)

    return rectangles

def configure_prediction_norm(scores, rectangles, alpha):
    """
    Configure prediction regions in a norm-based manner using adaptive scaling
    and quantile thresholds.

    This function iterates over a list of rectangle objects and, at each step,
    performs the following:
      1. Scales the input scores by sampling from the current rectangle, 
         optionally accounting for an exclusion rectangle (the previous one).
      2. Computes the sup norm (maximum across each sample row) on the scaled data.
      3. Calculates a quantile threshold based on (1 - alpha) and the total number
         of rectangles.
      4. Constructs a new prediction region as a `Rectangle(Quantile * scale)`.
      5. Finds intersections among the newly formed region and relevant rectangles,
         and appends non-empty intersections to `prediction_regions`.

    Parameters
    ----------
    scores : numpy.ndarray
        2D array of shape (n_samples, n_features) containing calibration or
        prediction scores.
    rectangles : list of Rectangle
        A list of `Rectangle` objects that define regions used for scaling
        and intersection checks. Each rectangle is assumed to have methods
        like `intersection()` and accessible attributes `lower` and `upper`.
    alpha : float
        The miscoverage level or risk tolerance used to compute quantiles.
        Typically a value between 0 and 1.

    Returns
    -------
    prediction_regions : list
        A list of elements, where each element is either:
        - A single intersection `[prediction_region.intersection(rectangles[idx]), 0]` 
          if `idx == 0`, or
        - A pair of intersections 
          `[prediction_region.intersection(rectangles[idx]), 
           prediction_region.intersection(rectangles[idx - 1])]`
          otherwise.
        The exact structure depends on whether it is the first rectangle, the last,
        or an intermediate one, and only non-empty intersections (based on `intersection_bound`)
        are appended.
    """

    prediction_regions = []
    for index in range(len(rectangles)+1):

        # Compute the scaled scores
        if index == 0:
            scores_scaled, scale = scale_with_sampling(scores, rectangles[index])
            
            # Compute the sup norm and corresponding quantile
            max_norm_scaled = np.max(scores_scaled, axis=1)
            level = math.ceil((1 - alpha) * (len(rectangles) + 1)) / len(rectangles)
            Quantile = np.quantile(max_norm_scaled, level)
            prediction_region = Rectangle(Quantile * scale)

            intersection_bound = np.max(
                prediction_region.intersection(rectangles[index]).upper - 0
            )
            if intersection_bound != 0:
                prediction_regions.append(
                    [prediction_region.intersection(rectangles[index]), 0]
                )

        elif index == len(rectangles):
            # Slightly enlarge the rectangle for sampling
            rectangle = Rectangle(rectangles[index - 1].upper * 1.2)
            scores_scaled, scale = scale_with_sampling(
                scores, rectangle, rectangles[index - 1]
            )

            # Compute the sup norm and corresponding quantile
            max_norm_scaled = np.max(scores_scaled, axis=1)
            level = math.ceil((1 - alpha) * (len(rectangles) + 1)) / len(rectangles)
            Quantile = np.quantile(max_norm_scaled, level)
            prediction_region = Rectangle(Quantile * scale)

            intersection_bound = np.max(
                prediction_region.intersection(rectangle).upper
                - prediction_region.intersection(rectangles[index - 1]).upper
            )
            if intersection_bound != 0:
                prediction_regions.append([
                    prediction_region.intersection(rectangle),
                    prediction_region.intersection(rectangles[index - 1])
                ])

        else:
            scores_scaled, scale = scale_with_sampling(
                scores, rectangles[index], rectangles[index - 1]
            )

            # Compute the sup norm and corresponding quantile
            max_norm_scaled = np.max(scores_scaled, axis=1)
            level = math.ceil((1 - alpha) * (len(rectangles) + 1)) / len(rectangles)
            Quantile = np.quantile(max_norm_scaled, level)
            prediction_region = Rectangle(Quantile * scale)

            # Determine and record the intersection
            intersection_bound = np.max(
                prediction_region.intersection(rectangles[index]).upper
                - prediction_region.intersection(rectangles[index - 1]).upper
            )
            if intersection_bound != 0:
                prediction_regions.append([
                    prediction_region.intersection(rectangles[index]),
                    prediction_region.intersection(rectangles[index - 1])
                ])

    return prediction_regions






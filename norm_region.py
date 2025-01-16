import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
from rectangle import Rectangle
import utility

def dissection_norm_region(scores):
    """
    Dissect the space of calibration scores into sorted rectangles based on the sup norm.

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
        A list of elements, where each element is either a single rectangle, stored as [rectangle, 0]
        or a pairs of rectangles, stored as [rectangle1, rectangle2]
    """
    prediction_regions = []
    n_rects = len(rectangles)

    for idx in range(n_rects + 1):
        if idx == 0:
            prediction_region = utility.compute_prediction_region(
                scores, alpha,
                rect_to_scale=rectangles[idx]
            )
            # Check intersection
            rectangle_current = prediction_region.intersection(rectangles[idx])
            rectangle_exclusion = 0
            intersection_bound = np.max(rectangle_current.upper- 0)
            if intersection_bound != 0:
                prediction_regions.append([
                    prediction_region.intersection(rectangles[idx]), 
                    0
                ])

        elif idx == n_rects:
            enlarged_rect = Rectangle(rectangles[idx - 1].upper * 1.2)
            prediction_region = utility.compute_prediction_region(
                scores, alpha,
                rect_to_scale=enlarged_rect,
                rect_exclusion=rectangles[idx - 1]
            )

            # Check intersection
            rectangle_current = prediction_region.intersection(enlarged_rect)
            rectangle_exclusion = prediction_region.intersection(rectangles[idx - 1])
            intersection_bound = np.max(rectangle_current.upper- rectangle_exclusion.upper)
            if intersection_bound != 0:
                if rectangle_exclusion.same_as(prediction_regions[-1][0]):
                    prediction_regions[-1][0] = rectangle_current
                else:
                    prediction_regions.append([rectangle_current,rectangle_exclusion])

        else:
            prediction_region = utility.compute_prediction_region(
                scores, alpha,
                rect_to_scale=rectangles[idx],
                rect_exclusion=rectangles[idx - 1]
            )
            # Check intersection
            rectangle_current = prediction_region.intersection(rectangles[idx])
            rectangle_exclusion = prediction_region.intersection(rectangles[idx - 1])
            intersection_bound = np.max(rectangle_current.upper- rectangle_exclusion.upper)
            if intersection_bound != 0:
                if rectangle_exclusion.same_as(prediction_regions[-1][0]):
                    prediction_regions[-1][0] = rectangle_current
                else:
                    prediction_regions.append([rectangle_current,rectangle_exclusion])

    return prediction_regions

def make_scoreRegion_plot_norm(scores, regions, dimx=None, dimy=None):
    """
    Plot the given scores and region(s) using pairwise 2D projections or a specific
    pair of dimensions.

    Parameters
    ----------
    scores : numpy.ndarray
        2D array of shape (n_samples, n_dimensions) containing the points to plot.
        Each row represents a sample, and each column a dimension.
    regions : list of tuples
        Each element is a tuple (region1, region2) or (region1, 0), representing
        one or two rectangles. A rectangle is expected to have `lower` and `upper`
        attributes and a `dimensions()` method. If the second element is 0, it
        indicates a single rectangle region.
    dimx : int, optional
        Dimension index for the x-axis if a single 2D projection is desired. Must
        be provided alongside `dimy`. Default: None.
    dimy : int, optional
        Dimension index for the y-axis if a single 2D projection is desired. Must
        be provided alongside `dimx`. Default: None.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the subplot(s).
    axes : matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
        The axes object(s). If plotting all pairs of dimensions, this is an
        array of Axes; otherwise, a single Axes object is returned.
    """

    n = regions[0][0].dimensions()
    global_min = 0
    global_max = np.max(scores)*1.1
    scores_tranpose = np.transpose(scores)

    # Helper function to draw a 2D projection for a pair of dims
    def draw_region(ax, rectangles, dx, dy):

        if rectangles[1] == 0:
            x_min, x_max = rectangles[0].lower[dx],  rectangles[0].upper[dx]
            y_min, y_max =  rectangles[0].lower[dy],  rectangles[0].upper[dy]
            x_coords = [x_min, x_max, x_max, x_min, x_min]
            y_coords = [y_min, y_min, y_max, y_max, y_min]
        else:
            x1_min, x1_max = rectangles[0].lower[dx],  rectangles[0].upper[dx]
            y1_min, y1_max =  rectangles[0].lower[dy],  rectangles[0].upper[dy]

            x2_min, x2_max = rectangles[1].lower[dx],  rectangles[1].upper[dx]
            y2_min, y2_max =  rectangles[1].lower[dy],  rectangles[1].upper[dy]

            x_coords = [x2_min, x2_max, x2_max, x1_max, x1_max, x1_min, x2_min]
            y_coords = [y2_max, y2_max, y2_min, y2_min, y1_max, y1_max, y2_max]

        # Outline the rectangle
        
        ax.plot(x_coords, y_coords, 'r-', alpha = 0.1)
        ax.fill(x_coords, y_coords, color='red', alpha=0.1)

        # Make axes the same size for easy comparison
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"Dimension {dx}")
        ax.set_ylabel(f"Dimension {dy}")
        ax.set_title(f"Projection on dims ({dx}, {dy})")

    if dimx is not None and dimy is not None:
            if not (0 <= dimx < n and 0 <= dimy < n):
                raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
            if dimx == dimy:
                raise ValueError("dimx and dimy must be different to form a valid 2D projection.")

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            for region in regions:
                draw_region(ax,region, dimx, dimy)

            ax.scatter(scores_tranpose[dimx], scores_tranpose[dimy], color = "blue", s=1)
            plt.tight_layout()
            return fig, ax
    
    # Case 2: All pairwise projections 
    elif dimx is None and dimy is None:

        dim_pairs = list(combinations(range(n), 2))  # all unique pairs
        num_plots = len(dim_pairs)

        # Decide on a subplot grid layout
        cols = min(num_plots, 3)
        rows = math.ceil(num_plots / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
        axes = axes.flatten()  # Flatten for easy indexing

        for idx, (dx, dy) in enumerate(dim_pairs):
            ax = axes[idx]
            for region in regions:
                draw_region(ax, region, dx, dy)

            ax.scatter(scores_tranpose[dx], scores_tranpose[dy], color = "blue", s=1)

        # Hide any unused axes (if num_plots < rows*cols)
        for j in range(num_plots, rows*cols):
            fig.delaxes(axes[j])

        plt.tight_layout()
        return fig, axes

        # If only one of dimx or dimy is specified, raise an error
    else:
        raise ValueError(
            "You must either provide both dimx and dimy for a single plot "
            "or leave both as None to plot all dimension pairs."
        )

def coverage_rate_norm(scores, regions):
    '''
    Compute the coverage rate in the norm-region method
    '''
    evaluation = np.zeros(len(scores))
    for index in range(len(regions)):
        if index == 0:
            evaluation += regions[index][0].contain_points(scores).astype(int)
        else:
            eval = regions[index][0].contain_points(scores).astype(int)-regions[index][1].contain_points(scores).astype(int)
            evaluation += eval
    print(evaluation)
    return np.sum(evaluation)/(len(scores))

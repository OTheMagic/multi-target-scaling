import numpy as np
from itertools import combinations
import math
import matplotlib.pyplot as plt

class Rectangle:
    """
    A class representing an n-dimensional hyper-rectangle.

    Attributes
    ----------
    lower : np.ndarray
        A 1D array of length n representing the lower (min) corner in each dimension.
    upper : np.ndarray
        A 1D array of length n representing the upper (max) corner in each dimension.
    """

    def __init__(self, upper, lower=None):
        """
        Parameters
        ----------
        lower : array-like, optional
            The lower (min) corner for each dimension. Default: the origin
        upper : array-like
            The upper (max) corner for each dimension.
        """
        self.upper = np.array(upper, dtype=float)

        if lower is None:
            self.lower = np.zeros(len(self.upper))
        else:
            self.lower = np.array(lower, dtype=float)
            if len(self.lower) != len(self.upper):
                raise ValueError("Unmatched dimension")
        if (self.lower > self.upper).any():
            raise ValueError("Each dimension of 'lower' must be less than corresponding dimension of 'upper'.")

    def info(self):
        '''
        Return the lower and upper information of this hyper-rectangle.
        '''
        return np.array([self.lower, self.upper])
    
    def dimensions(self):
        """
        Return the dimension of this hyper-rectangle.
        """
        return len(self.lower)

    def length_along_dimension(self, d):
        """
        Return the length of the dth dimensional side of hyper-rectangle.
        """
        return self.upper[d] - self.lower[d]

    def intersects(self, other):
        """
        Check whether this hyper-rectangle intersects with another.
        
        Parameters
        ----------
        other : Rectangle
        
        Returns
        -------
        bool
            True if the rectangles intersect in all dimensions.
        """
        # two hyper-rectangles has an intersection if and only if they have an intersection in every dimension
        return np.all(self.lower <= other.upper) and np.all(self.upper >= other.lower)

    def intersection(self, other):
        """
        Compute the intersection of this hyper-rectangle with another.

        Parameters
        ----------
        other : Rectangle

        Returns
        -------
        Rectangle or None
            A new Rectangle object representing the intersection,
            or None if the rectangles do not intersect.
        """
        # intersection of two hyper-rectangles is fixed by 
        # the maximum (coordinate-wise) of lower corner and minimum (coordinate-wise) of upper corner
        new_lower = np.maximum(self.lower, other.lower)
        new_upper = np.minimum(self.upper, other.upper)

        # If any dimension ends up reversed, there's no intersection.
        if np.any(new_lower > new_upper):
            return None

        return Rectangle(new_upper, lower = new_lower)

    
    def plot(self, dimx=None, dimy=None):
        """
        Create subplots showing the 2D slice of this hyper-rectangle
        either onto each pair of dimensions (the default),
        or onto a specified pair of dimensions (dimx, dimy).

        Parameters
        ----------
        dimx : int, optional
            Dimension index for the horizontal axis. Default: None
        dimy : int, optional
            Dimension index for the vertical axis. Default: None

        Returns
        -------
        fig, axes : matplotlib Figure and Axes object(s)
            The figure and array of axes used for plotting.
        """

        n = self.dimensions()
        if n < 2:
            raise ValueError("Plotting requires at least 2 dimensions.")

        # ------------------------------------------------------------
        # 1) Compute a global min and max across *all* dimensions
        #    so that each subplot has the same xlim and ylim.
        # ------------------------------------------------------------
        global_min = 0
        global_max = self.upper.max()*1.1

        # Helper function to draw a 2D projection for a pair of dims
        def draw_rectangle(ax, dx, dy):
            x_min, x_max = self.lower[dx], self.upper[dx]
            y_min, y_max = self.lower[dy], self.upper[dy]

            # Outline the rectangle
            x_coords = [x_min, x_max, x_max, x_min, x_min]
            y_coords = [y_min, y_min, y_max, y_max, y_min]
            ax.plot(x_coords, y_coords, 'b-', alpha = 0.1)
            ax.fill(x_coords, y_coords, color='blue', alpha=0.1)

            # Make axes the same size for easy comparison
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(global_min, global_max)

            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel(f"Dimension {dx}")
            ax.set_ylabel(f"Dimension {dy}")
            ax.set_title(f"Projection on dims ({dx}, {dy})")

        # Case 1: A single pair (dimx, dimy) 
        if dimx is not None and dimy is not None:
            if not (0 <= dimx < n and 0 <= dimy < n):
                raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
            if dimx == dimy:
                raise ValueError("dimx and dimy must be different to form a valid 2D projection.")

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            draw_rectangle(ax, dimx, dimy)
            plt.tight_layout()
            plt.show()
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
                draw_rectangle(ax, dx, dy)

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
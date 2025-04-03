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
        upper : array-like
            The upper (max) corner for each dimension.
        lower : array-like, optional
            The lower (min) corner for each dimension. Default: the origin
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

    def update_upper(self, new_upper, index = None):

        if index is not None:    
            self.upper[index] = new_upper
        else:
            self.upper = new_upper
    
    def update_lower(self, new_lower, index = None):

        if index is not None:    
            self.lower[index] = new_lower
        else:
            self.lower = new_lower

    # Basic information retrivers
    def info(self):
        """
        Return the lower and upper information of this hyper-rectangle.
        """
        return np.array([self.lower, self.upper])
    
    def dimensions(self):
        """
        Return the dimension of this hyper-rectangle.
        """
        return len(self.lower)

    def length_along_dimensions(self):
        """
        Return the side-lengths of hyper-rectangle.
        """
        d = self.dimensions()
        lengths = np.zeros(d)
        for i in range(d):
            lengths[i] = self.upper[i] - self.lower[i]
        
        return lengths
    
    def volume(self):
        """
        Return the volume of the hyper-rectangle
        """
        lengths = self.length_along_dimensions()
        volume = 1
        for length in lengths:
            volume = volume*length
        
        return volume


    # Intersection configurators
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
        if np.any(new_lower >= new_upper):
            return None

        return Rectangle(new_upper, new_lower)
    
    def same_as(self, other):
        """
        Check if this rectangle is exactly the same as another rectangle.

        Parameters
        ----------
        other : Rectangle
            Another Rectangle instance.

        Returns
        -------
        bool
            True if both rectangles are identical in terms of 
            their lower and upper coordinates; False otherwise.
        """

        # Optional: check if dimensionalities match
        if self.lower.shape != other.lower.shape:
            return False

        # Exact element-wise comparison
        same_lower = np.all(np.isclose(self.lower, other.lower, rtol=0, atol=1e-8))
        same_upper = np.all(np.isclose(self.upper, other.upper, rtol=0, atol=1e-8))
        return same_lower and same_upper
    
    # Coverage checkers
    def contain(self, point):
        """
        Check if a single point is contained in this rectangle.

        Parameters
        ----------
        point : array-like
            A 1D array of shape (n_dimensions,).

        Returns
        -------
        bool
            True if the point lies inside or on the boundary of the rectangle
            in every dimension; False otherwise.
        """
        if point.shape != self.lower.shape:
            raise ValueError("Point dimensionality must match the rectangle dimensionality.")

        return bool(np.all(point >= self.lower) and np.all(point <= self.upper))

    def contain_points(self, points):
        """
        Check which points from a batch of points are contained in this rectangle.

        Parameters
        ----------
        points : numpy.ndarray
            A 2D array of shape (n_points, n_dimensions). Each row is a point.

        Returns
        -------
        numpy.ndarray
            A 1D boolean array of length n_points. The i-th entry is True if
            points[i] is inside or on the boundary of this rectangle in every
            dimension, and False otherwise.
        """

        # Check shape consistency
        if points.ndim != 2:
            raise ValueError("Input 'points' must be a 2D array.")
        if points.shape[1] != self.lower.size:
            raise ValueError(
                f"Each point must have dimensionality {self.lower.size}, "
                f"but got {points.shape[1]}."
            )

        # For each row in points, check lower[d] <= points[i, d] <= upper[d]
        cond_lower = (points >= self.lower) 
        cond_upper = (points <= self.upper) 
        inside = np.all(cond_lower & cond_upper, axis=1)

        return inside
    

    # Graphing helpers
    def draw_2D(self, ax, dimx=0, dimy=1,boundary_color = None, fill_color = "red", transparancy = 0.5, min = None, max = None):
        x_min, x_max = self.lower[dimx], self.upper[dimx]
        y_min, y_max = self.lower[dimy], self.upper[dimy]

        # Outline the rectangle
        x_coords = [x_min, x_max, x_max, x_min, x_min]
        y_coords = [y_min, y_min, y_max, y_max, y_min]
        if boundary_color:
            ax.plot(x_coords, y_coords, boundary_color, alpha = transparancy)
        if fill_color:
            ax.fill(x_coords, y_coords, color=fill_color, alpha=transparancy)

        if min is not None and max is not None:
            ax.set_xlim(min, max)
            ax.set_ylim(min, max)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f"Dimension {dimx}")
        ax.set_ylabel(f"Dimension {dimy}")


    
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
            
        # Case 1: A single pair (dimx, dimy) 
        if dimx is not None and dimy is not None:
            if not (0 <= dimx < n and 0 <= dimy < n):
                raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
            if dimx == dimy:
                raise ValueError("dimx and dimy must be different to form a valid 2D projection.")

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            self.draw_2D(ax, dimx, dimy, min = global_min, max = global_max)
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
                self.draw_2D(ax, dx, dy, min = global_min, max = global_max)

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
        



import numpy as np
from itertools import combinations
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Rectangle:
    """
    A class representing an n-dimensional hyper-rectangle defined by lower and upper bounds
    in each dimension.

    Attributes
    ----------
    lower : np.ndarray
        Lower (minimum) corner of the rectangle in each dimension.
    upper : np.ndarray
        Upper (maximum) corner of the rectangle in each dimension.
    """

    def __init__(self, upper, lower=None):
        """
        Initialize a new Rectangle instance.

        Parameters
        ----------
        upper : array-like
            Upper bounds in each dimension.
        lower : array-like, optional
            Lower bounds in each dimension. Defaults to the origin (zeros).
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

    def update_upper(self, new_upper, index=None):
        """
        Update the upper bound(s) of the rectangle.

        Parameters
        ----------
        new_upper : float or array-like
            New upper value(s).
        index : int, optional
            Index of the dimension to update (if updating a single dimension).
        """
        if index is not None:
            self.upper[index] = new_upper
        else:
            self.upper = new_upper

    def update_lower(self, new_lower, index=None):
        """
        Update the lower bound(s) of the rectangle.

        Parameters
        ----------
        new_lower : float or array-like
            New lower value(s).
        index : int, optional
            Index of the dimension to update (if updating a single dimension).
        """
        if index is not None:
            self.lower[index] = new_lower
        else:
            self.lower = new_lower

    def info(self):
        """
        Get the lower and upper corners of the rectangle.

        Returns
        -------
        np.ndarray
            A 2D array with the lower and upper bounds.
        """
        return np.array([self.lower, self.upper])

    def dimensions(self):
        """
        Get the number of dimensions of the rectangle.

        Returns
        -------
        int
            Dimensionality of the rectangle.
        """
        return len(self.lower)

    def length_along_dimensions(self):
        """
        Calculate the side lengths of the rectangle along each dimension.

        Returns
        -------
        np.ndarray
            Lengths along each dimension.
        """
        return self.upper - self.lower

    def volume(self):
        """
        Compute the volume (or hypervolume) of the rectangle.

        Returns
        -------
        float
            The product of lengths in all dimensions.
        """
        return np.prod(self.length_along_dimensions())

    def intersects(self, other):
        """
        Check whether this rectangle intersects with another.

        Parameters
        ----------
        other : Rectangle
            Another rectangle instance.

        Returns
        -------
        bool
            True if the rectangles intersect in all dimensions.
        """
        return np.all(self.lower <= other.upper) and np.all(self.upper >= other.lower)

    def intersection(self, other):
        """
        Compute the intersection of this rectangle with another.

        Parameters
        ----------
        other : Rectangle
            Another rectangle instance.

        Returns
        -------
        Rectangle or None
            A new Rectangle representing the intersection, or None if no intersection exists.
        """
        new_lower = np.maximum(self.lower, other.lower)
        new_upper = np.minimum(self.upper, other.upper)

        if np.any(new_lower > new_upper):
            return None

        return Rectangle(new_upper, new_lower)

    def same_as(self, other):
        """
        Check whether this rectangle is exactly the same as another.

        Parameters
        ----------
        other : Rectangle

        Returns
        -------
        bool
            True if lower and upper bounds match within tolerance.
        """
        if self.lower.shape != other.lower.shape:
            return False

        same_lower = np.all(np.isclose(self.lower, other.lower, rtol=0, atol=1e-8))
        same_upper = np.all(np.isclose(self.upper, other.upper, rtol=0, atol=1e-8))
        return same_lower and same_upper

    def contains(self, point):
        """
        Check whether a given point lies within the rectangle.

        Parameters
        ----------
        point : array-like

        Returns
        -------
        bool
            True if the point lies within bounds.
        """
        if point.shape != self.lower.shape:
            raise ValueError("Point dimensionality must match the rectangle dimensionality.")

        return bool(np.all(point >= self.lower) and np.all(point <= self.upper))

    def contain_points(self, points):
        """
        Check which points from a batch are contained within the rectangle.

        Parameters
        ----------
        points : np.ndarray
            2D array of shape (n_points, n_dimensions).

        Returns
        -------
        np.ndarray
            Boolean array indicating point inclusion.
        """
        if points.ndim != 2:
            raise ValueError("Input 'points' must be a 2D array.")
        if points.shape[1] != self.lower.size:
            raise ValueError(f"Each point must have dimensionality {self.lower.size}, but got {points.shape[1]}.")

        cond_lower = (points >= self.lower)
        cond_upper = (points <= self.upper)
        inside = np.all(cond_lower & cond_upper, axis=1)

        return inside

    def draw_2D(self, ax, dimx=0, dimy=1, 
                boundary_color=None,
                fill_color="red",
                transparency=0.5,
                linewidth=0.5):
        """
        Draw a 2D projection of the rectangle onto a given axis using patches.Rectangle.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which to draw.
        dimx : int
            Dimension index for x-axis.
        dimy : int
            Dimension index for y-axis.
        boundary_color : str, optional
            Color of the rectangle boundary.
        fill_color : str, optional
            Fill color.
        transparency : float, optional
            Transparency level of the fill.
        linewidth : float, optional
            Line width of the boundary.
        """
        x_min, x_max = self.lower[dimx], self.upper[dimx]
        y_min, y_max = self.lower[dimy], self.upper[dimy]


        width = x_max - x_min
        height = y_max - y_min


        rect = patches.Rectangle(
        (x_min, y_min), width, height,
        linewidth=linewidth,
        edgecolor=boundary_color,
        facecolor=fill_color,
        alpha=transparency
        )
        ax.add_patch(rect)
        ax.set_aspect('equal', adjustable='box')

    def plot(self, dimx=None, dimy=None):
        """
        Plot the rectangle in 2D using projections.

        Parameters
        ----------
        dimx : int, optional
            Dimension index for x-axis.
        dimy : int, optional
            Dimension index for y-axis.

        Returns
        -------
        tuple
            matplotlib Figure and Axes objects.
        """
        n = self.dimensions()
        if n < 2:
            raise ValueError("Plotting requires at least 2 dimensions.")

        coor_max = np.max(self.upper, axis=0)*1.5

        if dimx is not None and dimy is not None:
            if not (0 <= dimx < n and 0 <= dimy < n):
                raise ValueError(f"dimx={dimx} or dimy={dimy} out of range for {n}-dimensional rectangle.")
            if dimx == dimy:
                raise ValueError("dimx and dimy must be different to form a valid 2D projection.")

            fig, ax = plt.subplots(1, 1, figsize=(6, 6))
            self.draw_2D(ax, dimx, dimy)
            ax.set(xlim = (0, coor_max[dimx]), ylim = (0, coor_max[dimy]))
            plt.tight_layout()
            return fig, ax

        elif dimx is None and dimy is None:
            dim_pairs = list(combinations(range(n), 2))
            num_plots = len(dim_pairs)

            cols = min(num_plots, 3)
            rows = math.ceil(num_plots / cols)

            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows), squeeze=False)
            axes = axes.flatten()

            for idx, (dx, dy) in enumerate(dim_pairs):
                ax = axes[idx]
                self.draw_2D(ax, dx, dy)
                ax.set(xlim = (0, coor_max[dimx]), ylim = (0, coor_max[dimy]))

            for j in range(num_plots, rows*cols):
                fig.delaxes(axes[j])

            plt.tight_layout()
            return fig, axes

        else:
            raise ValueError(
                "You must either provide both dimx and dimy for a single plot "
                "or leave both as None to plot all dimension pairs."
            )

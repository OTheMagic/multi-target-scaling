import numpy as np
from itertools import combinations
import math

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

    def __init__(self, upper, lower=None, *, upper_closed=True, allow_empty=False):
        """
        Initialize a new Rectangle instance.

        Parameters
        ----------
        upper : array-like
            Upper bounds in each dimension.
        lower : float or array-like, optional
            Closed lower bounds, scalar or shape (d,). Defaults to zeros;
            negative infinity represents an unbounded lower face.
        upper_closed : bool or array-like, optional
            Whether each upper boundary is included. Lower boundaries are closed.
        allow_empty : bool, optional
            Preserve bounds with lower > upper as an empty region instead of
            raising. Defaults to False for existing callers.
        """
        self.upper = np.array(upper, dtype=float)

        if lower is None:
            self.lower = np.zeros(len(self.upper))
        else:
            self.lower = np.array(lower, dtype=float)
            if self.lower.ndim == 0:
                self.lower = np.full(self.upper.shape, self.lower.item())
            elif self.lower.shape != self.upper.shape:
                raise ValueError("Unmatched dimension")
        if np.any(np.isnan(self.lower)) or np.any(np.isposinf(self.lower)):
            raise ValueError("Lower bounds must be finite or negative infinity")

        if not allow_empty and (self.lower > self.upper).any():
            raise ValueError("Each dimension of 'lower' must be less than corresponding dimension of 'upper'.")
        self.upper_closed = np.broadcast_to(
            np.asarray(upper_closed, dtype=bool), self.upper.shape
        ).copy()

    @property
    def empty(self):
        """Whether the bounds describe no points, including open singletons."""
        return bool(np.any(self.lower > self.upper) or np.any(
            (self.lower == self.upper) & ~self.upper_closed
        ))

    def update_upper(self, new_upper, index=None):
        """
        Update the upper bound(s), preserving the existing boundary closure.

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
            new_upper = np.asarray(new_upper, dtype=float)
            if new_upper.shape != self.upper.shape:
                raise ValueError("Unmatched dimension")
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
            new_lower = np.asarray(new_lower, dtype=float)
            if new_lower.shape != self.lower.shape:
                raise ValueError("Unmatched dimension")
            self.lower = new_lower

    def info(self):
        """
        Get numeric corners only; use membership methods to honor open boundaries.

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
        return np.zeros_like(self.upper) if self.empty else self.upper - self.lower

    def volume(self):
        """
        Compute the volume (or hypervolume) of the rectangle.

        Returns
        -------
        float
            The product of lengths in all dimensions.
        """
        lengths = self.length_along_dimensions()
        if self.empty or np.any(lengths == 0):
            return 0.0
        return np.prod(lengths)

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
        return self.intersection(other) is not None

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
        new_closed = np.where(
            self.upper < other.upper, self.upper_closed,
            np.where(other.upper < self.upper, other.upper_closed,
                     self.upper_closed & other.upper_closed),
        )

        if self.empty or other.empty or np.any(new_lower > new_upper) or np.any(
            (new_lower == new_upper) & ~new_closed
        ):
            return None

        return Rectangle(new_upper, new_lower, upper_closed=new_closed)

    def same_as(self, other):
        """
        Check whether this rectangle is exactly the same as another.

        Parameters
        ----------
        other : Rectangle

        Returns
        -------
        bool
            True if numeric bounds match within tolerance and closure matches.
        """
        if self.lower.shape != other.lower.shape:
            return False

        same_lower = np.all(np.isclose(self.lower, other.lower, rtol=0, atol=1e-8))
        same_upper = np.all(np.isclose(self.upper, other.upper, rtol=0, atol=1e-8))
        return (same_lower and same_upper and self.empty == other.empty
                and np.array_equal(self.upper_closed, other.upper_closed))

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
        point = np.asarray(point, dtype=float)
        if point.shape != self.lower.shape:
            raise ValueError("Point dimensionality must match the rectangle dimensionality.")

        upper_ok = (point < self.upper) | ((point == self.upper) & self.upper_closed)
        return bool(not self.empty and np.all(point >= self.lower) and np.all(upper_ok))

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
        points = np.asarray(points, dtype=float)
        if points.ndim != 2:
            raise ValueError("Input 'points' must be a 2D array.")
        if points.shape[1] != self.lower.size:
            raise ValueError(f"Each point must have dimensionality {self.lower.size}, but got {points.shape[1]}.")

        cond_lower = (points >= self.lower)
        cond_upper = (points < self.upper) | ((points == self.upper) & self.upper_closed)
        inside = np.all(cond_lower & cond_upper, axis=1) & ~np.bool_(self.empty)

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
        import matplotlib.patches as patches

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
        import matplotlib.pyplot as plt

        n = self.dimensions()
        if n < 2:
            raise ValueError("Plotting requires at least 2 dimensions.")

        coor_max = self.upper * 1.5

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
                ax.set(xlim=(0, coor_max[dx]), ylim=(0, coor_max[dy]))

            for j in range(num_plots, rows*cols):
                fig.delaxes(axes[j])

            plt.tight_layout()
            return fig, axes

        else:
            raise ValueError(
                "You must either provide both dimx and dimy for a single plot "
                "or leave both as None to plot all dimension pairs."
            )

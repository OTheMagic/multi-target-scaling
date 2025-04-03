from utility.rectangle import Rectangle
import matplotlib.pyplot as plt
import numpy as np

def graphing_tools_2D(ax, scores, regions = None, regions_color = "#fca3a3", rectangle = None, rectangle_fill_color = None, rectangle_boundary_color = None, label = None):

    scores_transpose = np.transpose(scores)
    ax.scatter(scores_transpose[0], scores_transpose[1], color = "#1f77b4", s = 2, zorder=10)

    if rectangle:
        rectangle.draw_2D(ax, fill_color = rectangle_fill_color, boundary_color = rectangle_boundary_color, transparancy = 0.5)
    if regions:
        for region in regions:
            region.draw_2D(ax, boundary_color = None, fill_color = regions_color, transparancy = 0.5)

    # Set up aesthetics
    limit = np.max(scores)*1.05
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_aspect('equal', adjustable = "box")
    if label: 
        ax.set_xlabel(f"Dimension {label[0]}")
        ax.set_ylabel(f"Dimension {label[1]}")
    else:
        ax.set_xlabel(f"Dimension 0")
        ax.set_ylabel(f"Dimension 1")
    plt.tight_layout()
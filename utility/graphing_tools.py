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
    plt.tight_layout()

def plot_metric_by_dim(df, method_name, metric_col, ylabel, include_title = False, include_legend = False):

    fig, ax = plt.subplots(1, 1, figsize = (6, 4), dpi=900)
    for d in sorted(df['n_dim'].unique()):
        subset = df[df['n_dim'] == d]
        ax.plot(subset['n_scores'], subset[metric_col], label=f'd={d}', marker='o')
    if include_title:
        ax.set_title(f"{ylabel} vs Sample Size (n) — {method_name}")
        ax.set_xlabel("Sample Size (n_scores)")
        ax.set_ylabel(ylabel)
    if include_legend:
        ax.legend(title="Dimension (d)", ncol=3)
    plt.grid(True)
    plt.tight_layout()

    return fig, ax

def compare_methods_by_dim(dfs, method_labels, metric_col, ylabel, title, include_title = False):
    """
    Plot a 3x3 grid comparing multiple methods across dimensions.

    Parameters:
        dfs (list of pd.DataFrame): List of dataframes [df_ds, df_lpr_o, df_npr]
        method_labels (list of str): Corresponding method names ["DS", "LPR_O", "NPR"]
        metric_col (str): Metric column to plot ("coverage_vol", "runtime", etc.)
        ylabel (str): Y-axis label
        title (str): Main figure title
    """
    unique_dims = sorted(set().union(*[df['n_dim'].unique() for df in dfs]))
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for i, d in enumerate(unique_dims):
        ax = axes[i]
        for df, label in zip(dfs, method_labels):
            subset = df[df['n_dim'] == d]
            ax.plot(subset['n_scores'], subset[metric_col], label=label, marker='o', linestyle='-')
        ax.set_title(f"Dimension d={d}")
        if i == 3:
            ax.set_ylabel(ylabel)
        ax.grid(True)
        if i == 7:
            ax.set_xlabel("Sample Size")

    if include_title:
        fig.suptitle(title, fontsize=16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(method_labels), bbox_to_anchor=(0.5, 1.03))
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    return fig, axes
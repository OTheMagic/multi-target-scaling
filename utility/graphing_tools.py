from utility.rectangle import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def graphing_tools_2D(ax, scores = None, scores_color = "#1f77b4", scores_size = 1,
                      regions = None, regions_color = "#fca3a3", linewidth = 0.5,
                      rectangle = None, rectangle_fill_color = None, rectangle_boundary_color = None):

    if scores is not None:
        scores_transpose = np.transpose(scores)
        ax.scatter(scores_transpose[0], scores_transpose[1], color = scores_color, s = scores_size, zorder=10)

    if rectangle:
        rectangle.draw_2D(ax, fill_color = rectangle_fill_color, 
                          boundary_color = rectangle_boundary_color, 
                          transparency = 0.5, linewidth=linewidth)
    if regions:
        for region in regions:
            region.draw_2D(ax, boundary_color = None, 
                           fill_color = regions_color, 
                           transparency = 0.5, linewidth=linewidth)

    # Set up aesthetics
    ax.set_aspect('equal', adjustable = "box")
    plt.tight_layout()

def short_cut_illustration(scores, alpha, bbox_to_anchor=(0.5, 1.05)):
    from utility.lpr import one_rect_prediction_regions_nD, mean_index_solver
    # Get prediction regions
    LPRO = one_rect_prediction_regions_nD(scores, alpha)
    LPR = one_rect_prediction_regions_nD(scores, alpha, False)[0]
    mean_index = mean_index_solver(scores)
    col_base = [np.sort(scores, axis = 0)[mean_index[0]-1][0], np.sort(scores, axis = 0)[mean_index[0]][0]]
    row_base = [np.sort(scores, axis = 0)[mean_index[1]-1][1], np.sort(scores, axis = 0)[mean_index[1]][1]]
    col_rect = Rectangle(upper=[col_base[1], LPRO.upper[1]], lower=[col_base[0], 0])
    row_rect = Rectangle(upper=[LPRO.upper[0], row_base[1]], lower=[0, row_base[0]])

    # Styles
    plt.rcParams.update({
    "font.size": 9, 
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    })
    search_color = "#2ca02c"     # Softer green
    LPR_color = "#fca3a3"        # Light red
    arrow_style = dict(arrowstyle="->", color='black', lw=0.5)
    score_legend = Line2D([0], [0], marker='o', color='w', label='Calibration Scores',
                      markerfacecolor="#1f77b4", markersize=6)
    LPR_legend = Patch(facecolor=LPR_color, edgecolor='none', label="LPR")
    search_legend = Patch(facecolor=search_color, edgecolor='none', label="Search area")
    LPRO_legend = Patch(facecolor="None", edgecolor="Black", label="LPR-O")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), dpi=600, sharex=True, sharey=True)
    for ax in axes:
        graphing_tools_2D(ax, scores, regions=LPR, scores_size=0.5, linewidth=0.3)
        ax.tick_params(axis='both')

    # Plot 1
    col_rect.draw_2D(axes[0], boundary_color=None, fill_color=search_color, transparency=0.7)
    row_rect.draw_2D(axes[0], boundary_color=None, fill_color=search_color, transparency=0.7)
    axes[0].axhline(LPRO.upper[1], color="black", linestyle="--", lw=0.5)
    axes[0].axvline(LPRO.upper[0], color="black", linestyle="--", lw=0.5)
    axes[0].annotate("Boundary Lines", 
        xy=(LPRO.upper[0], LPRO.upper[1]), 
        xytext=(LPRO.upper[0] -0.5, LPRO.upper[1] + 0.5),
        arrowprops=arrow_style, ha='right', fontsize = 5)
    # Plot 2
    LPRO.draw_2D(axes[1], boundary_color="black", fill_color="None", transparency=1, linewidth=1)
    
    fig.legend(handles=[score_legend, LPR_legend, search_legend, LPRO_legend], loc="upper center", ncol = 4, bbox_to_anchor=bbox_to_anchor, frameon = False)
    plt.tight_layout()

    return fig, axes

def small_sample_illustration(scores_cal, alpha, bbox_to_anchor=(0.55, 1.1)):
    from utility.lpr import one_rect_prediction_regions_nD
    from utility.npr import no_scaling_prediction_region
    from utility.ds import data_splitting_scaling_prediction_region, data_spliting_CHR_prediction_region
    scores_cal20 = scores_cal
    LPRO20 = one_rect_prediction_regions_nD(scores_cal20, alpha)
    LPR20 = one_rect_prediction_regions_nD(scores_cal20, alpha, False)[0]
    DSS20 = data_splitting_scaling_prediction_region(scores_cal20, alpha=alpha)
    DSCHR20 = data_spliting_CHR_prediction_region(scores_cal20, alpha=alpha)
    NPR20 = no_scaling_prediction_region(scores_cal20, alpha=alpha)

    vol_LPR20 = 0
    for reg in LPR20:
        vol_LPR20 += reg.volume()

    # Export table info
    table_arr = [["NPR", NPR20.volume()],
            ["DS-S", DSS20.volume()],
            ["DS-CHR", DSCHR20.volume()],   
            ["LPR-O", LPRO20.volume()],
            ["LPR", vol_LPR20]]
    column_labels = ["Method Name", "Volume"]
    table = pd.DataFrame(table_arr, columns=column_labels)
    latex_table = table.to_latex(index=False)
    with open("illustrations_in_paper/n20_d2.tex", "w") as f:
        f.write(latex_table)

    fig, ax = plt.subplots(1, 1, figsize = (3.5, 2), dpi = 900)

    # Draw test scores and LPR
    graphing_tools_2D(ax, scores=scores_cal20, regions=LPR20, linewidth=0.3)
    LPRO20.draw_2D(ax, boundary_color="Black", fill_color = None, transparency = 1)
    DSS20.draw_2D(ax, boundary_color="Blue", fill_color=None, transparency = 1)
    DSCHR20.draw_2D(ax, boundary_color="Brown", fill_color=None, transparency = 1)
    NPR20.draw_2D(ax, boundary_color="Green", fill_color=None, transparency = 1)

    # Styles
    score_legend = Line2D([0], [0], marker='o', color='w', label='Calibration Scores',
                        markerfacecolor="#1f77b4", markersize=5)
    LPR_legend = Patch(facecolor="#fca3a3", edgecolor='none', label="LPR")
    LPRO_legend = Patch(facecolor="None", edgecolor="Black", label="LPR-O")
    NPR_legend = Patch(facecolor="None", edgecolor="Green", label="NPR")
    DSS_legend = Patch(facecolor="None", edgecolor="Blue", label="DS-S")
    DSCHR_legend = Patch(facecolor="None", edgecolor="Brown", label="DS-CHR")
    fig.legend(handles = [NPR_legend, DSS_legend, DSCHR_legend, LPRO_legend, LPR_legend, score_legend], loc="upper center", ncol = 3, bbox_to_anchor=bbox_to_anchor, frameon = False)

    return fig, ax
def single_dim_comparison(df_dict, dim, runtime_key=False):
    colors = {
        'NPR': '#7f7f7f', 'DS-S': '#ff7f0e', 'DS+CHR': '#2ca02c',
        'LPR': '#d62728', 'LPR-O': '#9467bd', 'ECopula': '#8c564b',
        'DS-VCopula': '#e377c2'
    }
    markers = {
        'NPR': 'o', 'DS-S': 's', 'DS+CHR': '^',
        'LPR': 'v', 'LPR-O': 'D', 'ECopula': 'P',
        'DS-VCopula': 'X'
    }
    plt.rcParams.update({
    "font.size": 11, 
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    })

    # Safely filter data
    filtered_dict = {}
    for method, df in df_dict.items():
        filtered_dict[method] = df[df["n_dim"] == dim].copy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    axes = axes.flatten()

    for i in range(3):
        ax = axes[i]
        if i == 0:
            metric = "test_coverage_avg"
            ax.set_ylim(0.8, 1)
            ax.axhline(y=0.9, color="red", linestyle='--', linewidth=1, label=r'$\alpha = 0.9$')
            title = "Test Coverage"
        elif i == 1:
            metric = "coverage_vol_avg"
            title = "Volume (Log10)"
            ax.set_xlabel("No. of Calibration Points")
        else:
            metric = "runtime_avg"
            title = "Runtime (Log10)" if runtime_key else "Runtime"

        for method, df in filtered_dict.items():
            plot_data = df.copy()
            if (i == 2) and runtime_key:
                plot_data[metric] = np.log10(plot_data[metric])
            ax.plot(
                plot_data["n_cals"],
                plot_data[metric],
                label=method,
                marker=markers.get(method, 'o'),
                color=colors.get(method, 'black'),
                linestyle="-",
                linewidth = 1
            )

        ax.set_title(title)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.1), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes

def compare_metric_across_dims(df_dict, dim_list = [4, 10, 100], metric="test_coverage_avg", metric_label=None, logy=False):
    colors = {
        'NPR': '#7f7f7f', 'DS-S': '#ff7f0e', 'DS+CHR': '#2ca02c',
        'LPR': '#d62728', 'LPR-O': '#9467bd', 'ECopula': '#8c564b',
        'DS-VCopula': '#e377c2'
    }
    markers = {
        'NPR': 'o', 'DS-S': 's', 'DS+CHR': '^',
        'LPR': 'v', 'LPR-O': 'D', 'ECopula': 'P',
        'DS-VCopula': 'X'
    }

    plt.rcParams.update({
    "font.size": 11, 
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    })

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    axes = axes.flatten()

    for i, d in enumerate(dim_list):
        ax = axes[i]
        ax.set_title(f"Dimension $d={d}$")
        if i == 1:
            ax.set_xlabel("No. of Calibration Points")
        if i == 0 and metric_label:
            ax.set_ylabel(metric_label)
        elif i == 0:
            ax.set_ylabel(metric)

        for method, df in df_dict.items():
            df_d = df[df["n_dim"] == d]
            y = np.log10(df_d[metric]) if logy else df_d[metric]
            ax.plot(
                df_d["n_cals"],
                y,
                label=method,
                marker=markers.get(method, 'o'),
                color=colors.get(method, 'black'),
                linewidth=1,
                linestyle="-"
            )

        if metric=="test_coverage_avg":
            ax.set_ylim(0.8, 1)
            ax.axhline(y=0.9, color="red", linestyle='--', linewidth=1, label=r'$\alpha = 0.9$')
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.1), frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def single_dim_text_file(methods, dim, sample_list, trials, alpha, noise_list, log_scale = True, output_path = "results.txt"):
    from utility.exps import run_synthetic_experiment
    output = {}
    for method in methods:
        output[method] = run_synthetic_experiment(dim_list= [dim], sample_list=sample_list, alpha_list=[alpha], trials=trials, method=method, noises_list=noise_list, log_scale=log_scale)
    with open(output_path, "w") as f:
        for method, df in output.items():
            f.write(f"Method: {method}\n")
            f.write(df.to_string(index=False))
            f.write("\n\n" + "-"*60 + "\n\n")

def generate_latex_table_from_csvs(file_dict, dim, filename="table_dX.tex"):
    """
    Combine CSV files, filter by dimension, format LaTeX table, and save to .tex file.
    
    Parameters:
    - file_dict: dict, keys are method names, values are file paths
    - dim: int, dimension to filter on
    - filename: str, name of output .tex file
    """
    combined = []

    for method, path in file_dict.items():
        df = pd.read_csv(path)
        df["method"] = method
        combined.append(df)

    df_all = pd.concat(combined)
    df_all = df_all[df_all["n_dim"] == dim]
    df_all = df_all.sort_values(by=["n_cals", "method"])

    lines = []
    lines.append("\\begin{tabular}{c l r r r}")
    lines.append("\\toprule")
    lines.append("\\textbf{n Cal} & \\textbf{Method Name} & \\textbf{Volume ($\\pm$SD)} & \\textbf{Test Coverage ($\\pm$SD)} & \\textbf{Runtime (s)} \\\\")
    lines.append("\\midrule")

    for n_cals in df_all["n_cals"].unique():
        subset = df_all[df_all["n_cals"] == n_cals]
        methods = subset["method"].tolist()
        n_methods = len(methods)

        for idx, row in subset.iterrows():
            prefix = f"{int(n_cals)}" if idx == subset.index[0] else " "
            volume = f"{row['coverage_vol_avg']:.2f} ({row['coverage_vol_1std']:.2f})"
            coverage = f"{row['test_coverage_avg']:.3f} ({row['test_coverage_1std']:.3f})"
            runtime = f"{row['runtime_avg']:.3f}"
            lines.append(f"{prefix} & {row['method']} & {volume} & {coverage} & {runtime} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    filename = filename.replace("dX", f"d{dim}")
    with open(filename, "w") as f:
        f.write("\n".join(lines))

from utility.rectangle import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

## Color choices
colors = {
        'Unscaled Max': '#1f77b4',  
        'Bonferroni': '#17becf',
        'Point CHR': '#2ca02c',
        'TSCP-S': '#9467bd', 
        'TSCP-GWC': '#e377c2',
        'Emp. Copula': '#8c564b',
        'TSCP-LWC': '#7f7f7f',
        'TSCP-R': '#d62728', 
        'Pop. Oracle': '#bcbd22'
    }
markers = {
        'Unscaled Max': 'o',  
        'Bonferroni': 's',
        'Point CHR': '^',
        'TSCP-S': 'v', 
        'TSCP-GWC': 'D',
        'Emp. Copula': 'X',
        'TSCP-LWC': '*',
        'TSCP-R': '<', 
        'Pop. Oracle': '>'
    }

def method_name_coverter(method_list):
    method_name_list = {}
    for method in method_list:
        if method == "Point_CHR":
            method_name = "Point CHR"
        elif method == "TSCP_R":
            method_name = "TSCP-R"
        elif method == "TSCP_S":
            method_name = "TSCP-S"
        elif method == "Unscaled":
            method_name = "Unscaled Max"
        elif method == "Empirical_copula":
            method_name = "Emp. Copula"
        elif method == "TSCP_LWC":
            method_name = "TSCP-LWC"
        elif method == "Population_oracle":
            method_name = "Pop. Oracle"
        elif method == "Bonferroni":
            method_name = "Bonferroni"
        elif method == "TSCP_GWC":
            method_name = "TSCP-GWC"
        method_name_list[method] = method_name
    return method_name_list


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

def short_cut_illustration(scores, alpha, bbox_to_anchor=(0.5, 1.05), include_legend = False):
    from utility.res_rescaled import standardized_prediction, mean_index_solver
    # Get prediction regions
    LPRO = standardized_prediction(scores, alpha)
    LPR = standardized_prediction(scores, alpha, False)[0]
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
    LPR_legend = Patch(facecolor=LPR_color, edgecolor='none', label="Localized Prediction Regions")
    search_legend = Patch(facecolor=search_color, edgecolor='none', label="Search area")
    LPRO_legend = Patch(facecolor="None", edgecolor="Black", label="Short-cut")

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2), dpi=600, sharex=True, sharey=True)
    for ax in axes:
        graphing_tools_2D(ax, scores, regions=LPR, scores_size=0.5, linewidth=0.3)
        ax.tick_params(axis='both')

    # Plot 1
    col_rect.draw_2D(axes[0], boundary_color=None, fill_color=search_color, transparency=0.7)
    row_rect.draw_2D(axes[0], boundary_color=None, fill_color=search_color, transparency=0.7)
    axes[0].axhline(LPRO.upper[1], color="black", linestyle="--", lw=0.5)
    axes[0].axvline(LPRO.upper[0], color="black", linestyle="--", lw=0.5)
    axes[0].annotate("Upper Corner", 
        xy=(LPRO.upper[0], LPRO.upper[1]), 
        xytext=(LPRO.upper[0] -0.5, LPRO.upper[1] + 0.5),
        arrowprops=arrow_style, ha='right', fontsize = 5)
    # Plot 2
    LPRO.draw_2D(axes[1], boundary_color="black", fill_color="None", transparency=1, linewidth=1)
    
    if include_legend:
        fig.legend(handles=[score_legend, LPR_legend, search_legend, LPRO_legend], loc="upper center", ncol = 2, bbox_to_anchor=bbox_to_anchor, frameon = False)
    plt.tight_layout()

    return fig, axes

def small_sample_illustration(scores_cal, alpha, bbox_to_anchor=(0.55, 1.1), legend_loc = "center left", n_cols = 1):
    from utility.res_rescaled import one_rect_prediction_regions_nD, check_coverage_rate
    from utility.unscaled import no_scaling_prediction_region
    from utility.data_splitting import data_splitting_scaling_prediction_region, data_spliting_CHR_prediction_region
    from utility.copula import empirical_copula_prediction_region
    scores_cal20 = scores_cal
    LPRO20 = one_rect_prediction_regions_nD(scores_cal20, alpha)
    LPR20 = one_rect_prediction_regions_nD(scores_cal20, alpha, False)[0]
    DSS20 = data_splitting_scaling_prediction_region(scores_cal20, alpha=alpha)
    DSCHR20 = data_spliting_CHR_prediction_region(scores_cal20, alpha=alpha)
    NPR20 = no_scaling_prediction_region(scores_cal20, alpha=alpha)
    EMPC = empirical_copula_prediction_region(scores_cal20, alpha=alpha)

    vol_LPR20 = 0
    for reg in LPR20:
        vol_LPR20 += reg.volume()

    '''
    # Export table info
    table_arr = [["Unscaled", NPR20.volume(), check_coverage_rate(scores_test, NPR20)],
            ["Splitting baseline", DSS20.volume(), check_coverage_rate(scores_test, NPR20)],
            ["Point CHR", DSCHR20.volume(), check_coverage_rate(scores_test, NPR20)],
            ["Empirical copula", EMPC.volume(), check_coverage_rate(scores_test, NPR20)],   
            ["Scaling-based (F)", LPRO20.volume(), check_coverage_rate(scores_test, NPR20)],
            ["Scaling-based", vol_LPR20], check_coverage_rate(scores_test, NPR20)]
    column_labels = ["Method Name", "Volume"]
    table = pd.DataFrame(table_arr, columns=column_labels)
    latex_table = table.to_latex(index=False)
    with open("illustrations_in_paper/n20_d2.tex", "w") as f:
        f.write(latex_table)
    '''

    fig, ax = plt.subplots(1, 1, figsize = (5.5, 2), dpi = 900)

    # Draw test scores and LPR
    graphing_tools_2D(ax, scores=scores_cal20, regions=LPR20, linewidth=0.3)
    LPRO20.draw_2D(ax, boundary_color="Black", fill_color = None, transparency = 1)
    DSS20.draw_2D(ax, boundary_color="Gray", fill_color=None, transparency = 1)
    DSCHR20.draw_2D(ax, boundary_color="Brown", fill_color=None, transparency = 1)
    NPR20.draw_2D(ax, boundary_color="Green", fill_color=None, transparency = 1)
    EMPC.draw_2D(ax, boundary_color="Darkblue", fill_color=None, transparency = 1)

    # Styles
    score_legend = Line2D([0], [0], marker='o', color='w', label='Calibration Scores',
                        markerfacecolor="#1f77b4", markersize=5)
    LPR_legend = Patch(facecolor="#fca3a3", edgecolor='none', label="Scaling-based")
    LPRO_legend = Patch(facecolor="None", edgecolor="Black", label="Scaling-based (F)")
    NPR_legend = Patch(facecolor="None", edgecolor="Green", label="Unscaled")
    DSS_legend = Patch(facecolor="None", edgecolor="Gray", label="Splitting baseline")
    DSCHR_legend = Patch(facecolor="None", edgecolor="Brown", label="Point CHR")
    EMPC_legend = Patch(facecolor="None", edgecolor="Darkblue", label="Empirical copula")
    fig.legend(handles = [NPR_legend, DSS_legend, DSCHR_legend, EMPC_legend, LPRO_legend, LPR_legend, score_legend], loc=legend_loc, ncol = n_cols, bbox_to_anchor=bbox_to_anchor, frameon = False)

    return fig, ax

def single_dim_comparison(
    df_dict, dim, 
    include_runtime = True, 
    include_legend=True, 
    n_cols=2,
    loc = "center left",
    bbox_to_anchor=(1.02, 0.5),
    figsize = (11, 3),
    error_bar = True,
    error_bar_capsize=4
):
    plt.rcParams.update({
        "font.size": 11, 
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Safely filter data
    filtered_dict = {method: df[df["n_dim"] == dim].copy()
                     for method, df in df_dict.items()}

    if include_runtime:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        ax.set_xscale("log")
        if index == 0:
            metric = "test_coverage_avg"
            ax.set_ylim(0.7, 1)
            ax.axhline(y=0.9, color="red", linestyle='--', linewidth=1)
            title = "Test Coverage"
        elif index == 1:
            metric = "coverage_vol_avg"
            title = "Volume"
            ax.set_yscale("log")
            ax.set_xlabel("No. of Calibration Points")
        else:
            metric = "runtime_avg"
            title = r"Runtime ($\log_{10}$-Scale)"

        for method, df in filtered_dict.items():
            plot_data = df.copy()
            if index == 2:
                plot_data[metric] = np.log10(plot_data[metric])

            # Fallbacks if your dicts aren't global
            m = markers.get(method, 'o') if 'markers' in globals() else 'o'
            c = colors.get(method, 'black') if 'colors' in globals() else 'black'

            if index == 0:

                if error_bar:
                    yerr = plot_data["test_coverage_1std"].to_numpy()
                    ax.errorbar(
                        plot_data["n_cals"], plot_data[metric],
                        yerr=yerr, fmt=m + "-", color=c,
                        capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                        label=method
                    )

                ax.plot(
                    plot_data["n_cals"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
            elif index == 1:

                if error_bar:
                    yerr = plot_data["coverage_vol_1std"].to_numpy()
                    ax.errorbar(
                    plot_data["n_cals"], plot_data[metric],
                    yerr=yerr, fmt=m + "-", color=c,
                    capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                    label=method
                    )

                ax.plot(
                    plot_data["n_cals"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
            elif index == 2:
                ax.plot(
                    plot_data["n_cals"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
        ax.set_title(title)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    if include_legend:
        fig.legend(handles, labels, loc=loc, ncol=n_cols,
                   bbox_to_anchor=bbox_to_anchor, frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, axes


def heavy_t_comparison(
    df_dict, dim, sample,
    include_runtime = True, 
    include_legend=True, 
    n_cols=2,
    loc = "center left",
    bbox_to_anchor=(1.02, 0.5),
    figsize = (11, 3),
    error_bar = True,
    error_bar_capsize=4
):
    plt.rcParams.update({
        "font.size": 11, 
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Safely filter data
    filtered_dict = {
        method: df[(df["n_dim"] == dim) & (df["n_cals"] == sample)].copy()
        for method, df in df_dict.items()
    }
    if include_runtime:
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True)
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        ax.set_xscale("log")
        if index == 0:
            metric = "test_coverage_avg"
            ax.set_ylim(0.7, 1)
            ax.axhline(y=0.9, color="red", linestyle='--', linewidth=1)
            title = "Test Coverage"
        elif index == 1:
            metric = "coverage_vol_avg"
            title = "Volume"
            ax.set_yscale("log")
            ax.set_xlabel("No. of Degree of Freedom")
        else:
            metric = "runtime_avg"
            title = r"Runtime ($\log_{10}$-Scale)"

        for method, df in filtered_dict.items():
            plot_data = df[df["df"] < 100].copy()
            if index == 2:
                plot_data[metric] = np.log10(plot_data[metric])

            # Fallbacks if your dicts aren't global
            m = markers.get(method, 'o') if 'markers' in globals() else 'o'
            c = colors.get(method, 'black') if 'colors' in globals() else 'black'

            if index == 0:

                if error_bar:
                    yerr = plot_data["test_coverage_1std"].to_numpy()
                    ax.errorbar(
                        plot_data["df"], plot_data[metric],
                        yerr=yerr, fmt=m + "-", color=c,
                        capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                        label=method
                    )

                ax.plot(
                    plot_data["df"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
            elif index == 1:

                if error_bar:
                    yerr = plot_data["coverage_vol_1std"].to_numpy()
                    ax.errorbar(
                    plot_data["df"], plot_data[metric],
                    yerr=yerr, fmt=m + "-", color=c,
                    capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                    label=method
                    )

                ax.plot(
                    plot_data["df"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
            elif index == 2:
                ax.plot(
                    plot_data["df"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
        ax.set_title(title)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    if include_legend:
        fig.legend(handles, labels, loc=loc, ncol=n_cols,
                   bbox_to_anchor=bbox_to_anchor, frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, axes

def compare_across_dims(
    df_dict, sample_size, 
    include_runtime = True, 
    include_legend=True, 
    n_cols=2,
    error_bar = True,
    error_bar_capsize=4
):
    plt.rcParams.update({
        "font.size": 11, 
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    # Safely filter data
    filtered_dict = {method: df[df["n_cals"] == sample_size].copy()
                     for method, df in df_dict.items()}

    if include_runtime:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    axes = axes.flatten()

    for index, ax in enumerate(axes):
        if index == 0:
            metric = "test_coverage_avg"
            ax.set_ylim(0.7, 1)
            ax.axhline(y=0.9, color="red", linestyle='--', linewidth=1)
            title = "Test Coverage"
        elif index == 1:
            metric = "coverage_vol_avg"
            title = "Volume"
            ax.set_yscale("log")
            ax.set_xlabel("No. of Dimensions")
        else:
            metric = "runtime_avg"
            title = r"Runtime ($\log_{10}$-Scale)"

        for method, df in filtered_dict.items():
            plot_data = df.copy()
            if index == 2:
                plot_data[metric] = np.log10(plot_data[metric])

            # Fallbacks if your dicts aren't global
            m = markers.get(method, 'o') if 'markers' in globals() else 'o'
            c = colors.get(method, 'black') if 'colors' in globals() else 'black'

            if index == 0:

                if error_bar:
                    yerr = plot_data["test_coverage_1std"].to_numpy()
                    ax.errorbar(
                        plot_data["n_dim"], plot_data[metric],
                        yerr=yerr, fmt=m + "-", color=c,
                        capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                        label=method
                    )

                ax.plot(
                    plot_data["n_dim"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
            elif index == 1:

                if error_bar:
                    yerr = plot_data["coverage_vol_1std"].to_numpy()
                    ax.errorbar(
                    plot_data["n_dim"], plot_data[metric],
                    yerr=yerr, fmt=m + "-", color=c,
                    capsize=error_bar_capsize, elinewidth=1, linewidth=1,
                    label=method
                    )

                ax.plot(
                    plot_data["n_dim"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
            elif index == 2:
                ax.plot(
                    plot_data["n_dim"], plot_data[metric],
                    label=method, marker=m, color=c,
                    linestyle="-", linewidth=1
                )
                
        ax.set_title(title)
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    if include_legend:
        fig.legend(handles, labels, loc="center left", ncol=n_cols,
                   bbox_to_anchor=(1.02, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
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

def generate_latex_table_from_csvs(file_dict, dim, noise_type, filename="table_dX.tex"):
    """
    Combine CSV files, filter by dimension, format LaTeX table, and save to .tex file.
    
    Parameters:
    - file_dict: dict, keys are method names, values are file paths
    - dim: int, dimension to filter on
    - filename: str, name of output .tex file
    """
    combined = []

    for method, data in file_dict.items():
        data["method"] = method
        combined.append(data)

    df_all = pd.concat(combined)
    df_all = df_all[df_all["n_dim"] == dim]
    df_all = df_all.sort_values(by=["n_cals", "method"])

    lines = []
    lines.append("\\begin{tabular}{l l c c c}")
    lines.append("\\toprule")
    lines.append(f"(n, d, Noise) & Method & Coverage & Volume & Runtime \\\\")
    lines.append("\\midrule")

    for n_cals in df_all["n_cals"].unique():
        subset = df_all[df_all["n_cals"] == n_cals]
        n_methods = len(subset)

        for idx, (_, row) in enumerate(subset.iterrows()):
            cal_label = f"\\multirow{{{n_methods}}}{{*}}{{({int(n_cals)}, {int(dim)}, {noise_type})}}" if idx == 0 else ""
            vol_mean = row['coverage_vol_avg']
            vol_std = row['coverage_vol_1std']

            volume = f"{vol_mean:.3e}({vol_std:.3e})"
            coverage = f"{row['test_coverage_avg']:.3f} ({row['test_coverage_1std']:.3f})"
            runtime = f"{row['runtime_avg']:.3f}"
            lines.append(f"{cal_label} & {row['method']} & {coverage} & {volume} & {runtime} \\\\")

        lines.append("\\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    filename = filename.replace("dX", f"d{dim}")
    with open(filename, "w") as f:
        f.write("\n".join(lines))


def format_pm(mean, std, sci=False, highlight=False):
    """Format as mean(std), optionally in scientific notation and highlighted."""
    if pd.isna(mean) or pd.isna(std):
        return f"\\text{{inf}}"
    if sci:
        mean_str = f"{mean:.3e}"
        std_str = f"{std:.3e}"
    else:
        mean_str = f"{mean:.3f}"
        std_str = f"{std:.3f}"
    result = f"{mean_str}({std_str})"
    if highlight:
        return f"\\textcolor{{red}}{{{result}}}"
    return result


def extract_stats(df):
    """Return dict[method] = (cov, cov_std, vol, vol_std)."""
    stats = {}
    for _, row in df.iterrows():
        m = row["Methods"].strip()
        stats[m] = (
            row["test_coverage_avg"],
            row["test_coverage_1std"],
            row["coverage_vol"],
            row["coverage_vol_1std"],
        )
    return stats

def build_panel(panel_datasets, methods_order, caption):
    lines = []
    header_cols = " & ".join(
        [f"\\multicolumn{{2}}{{c}}{{{name} $(d={d},\\, n={n})$}}" for name, _, d, n in panel_datasets]
    )
    subheaders = " & ".join(["Coverage & Volume"] * len(panel_datasets))

    lines.append("\\begin{subtable}{\\textwidth}")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l" + "c c " * len(panel_datasets) + "}")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & " + header_cols + " \\\\")
    cmid = "".join([f"\\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(panel_datasets))])
    lines.append(cmid)
    lines.append("& " + subheaders + " \\\\")
    lines.append("\\midrule")

    dfs = []
    for _, file, _, _ in panel_datasets:
        df = pd.read_csv(file)
        dfs.append(extract_stats(df))

    # Find min volume per dataset for highlighting
    min_vols = []
    for data in dfs:
        vols = [v[2] for v in data.values() if pd.notna(v[2])]
        min_vols.append(min(vols) if vols else None)

    # Build each row (method)
    for method in methods_order:
        row_parts = [method]
        for j, data in enumerate(dfs):
            if method in data:
                cov, cov_std, vol, vol_std = data[method]
                cov_str = format_pm(cov, cov_std, sci=False)
                highlight = (vol == min_vols[j])
                vol_str = format_pm(vol, vol_std, sci=True, highlight=False)
            else:
                cov_str, vol_str = "--", "--"
            row_parts += [cov_str, vol_str]
        lines.append(" & ".join(row_parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\subcaption{{{caption}}}")
    lines.append("\\end{subtable}")
    return "\n".join(lines)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

sns.set_theme(
    style="whitegrid",  # beautiful light grid, best for scientific plots
    context="paper",     # good scaling
    font_scale=1.2,
    rc={
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 11,
}
)

# Color and marker order: select your own color order if needed
COLOR_ORDER = ["Unscaled Max",
               "Naive", 
               "Pop. Oracle",
               "TSCP (Ours)",
               "TSCP-S", 
               "Point CHR",
               "Bonferroni",
               "Emp. Copula",
               "TSCP-GWC", 
               "TSCP-LWC"]

# Graphing order
METHOD_ORDER = ["Emp. Copula",
                "Pop. Oracle", 
                "Unscaled Max",
                "Point CHR", 
                "Bonferroni",
                "Naive", 
                "TSCP-S", 
                "TSCP-GWC", 
                "TSCP-LWC",
                "TSCP (Ours)"]

# Consistent color palette across all figures
PALETTE = sns.color_palette("deep", n_colors=len(METHOD_ORDER))
METHOD_COLORS = dict(zip(COLOR_ORDER, PALETTE))

# Consistent markers across all figures
MARKERS = ["o", "^", "s", "D", "v", "<", ">", "P", "X", "H"]
METHOD_MARKERS = dict(zip(COLOR_ORDER, MARKERS))


def method_name_coverter(method_list):
    method_name_list = {}
    for method in method_list:
        if method == "Point_CHR":
            method_name = "Point CHR"
        elif method == "TSCP_R":
            method_name = "TSCP (Ours)"
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
        elif method == "Naive":
            method_name = "Naive"
        method_name_list[method] = method_name
    return method_name_list


def prepare_long_form(df_dict, t_dist = False):
    """
    Convert df_dict[method] (wide format) into a single long-form df
    usable by seaborn.
    """
    frames = []
    metric_pairs = [
        ("test_coverage_avg",  "test_coverage_1std"),
        ("coverage_vol_avg",   "coverage_vol_1std"),
        ("runtime_avg",        None)
    ]

    for method, df in df_dict.items():
        df_temp = df.copy()
        df_temp["method"] = method

        for metric, std_col in metric_pairs:
            tmp = df_temp[["method", "n_dim", "n_cals", metric]].copy()
            if t_dist == True:
                tmp = df_temp[["method", "n_dim", "n_cals", "df", metric]].copy()
            tmp = tmp.rename(columns={metric: "metric_value"})
            tmp["metric_name"] = metric

            if std_col is not None:
                tmp["metric_std"] = df_temp[std_col]
            else:
                tmp["metric_std"] = np.nan

            frames.append(tmp)

    long_df = pd.concat(frames, ignore_index=True)
    return long_df
    
def single_dim_comparison(
    df_dict, dim, 
    include_runtime=True,
    include_legend=True,
    figsize=(12, 3),
    legend_bbox=(1.02, 0.5),
    ncols=2,
    direction = "Horizontal",
    ylim = (0.6, 1)
):
    # Gather all data for n_dim == dim
    long_df = prepare_long_form(df_dict)
    df = long_df[long_df["n_dim"] == dim]

    # Panels to plot
    panels = ["test_coverage_avg", "coverage_vol_avg"]
    titles = ["Coverage", "Volume"]

    # Add Runtime data
    if include_runtime:
        panels.append("runtime_avg")
        titles.append("Runtime (log10-scale)")

    # Create figure
    if direction == "Horizontal":
        fig, axes = plt.subplots(
            1, len(panels),
            figsize=figsize,
            sharex=True
        )
    else:
        fig, axes = plt.subplots(
            len(panels), 1,
            figsize=figsize,
            sharex=True
        )

    # Enumerate the axes
    axes = axes.flatten()

    for ax, metric, title in zip(axes, panels, titles):

        plot_df = df[df["metric_name"] == metric].copy()

        # Runtime → log10 scale
        if metric == "runtime_avg":
            plot_df["metric_value"] = np.log10(plot_df["metric_value"])

        # Main seaborn line plot
        sns.lineplot(
            data=plot_df,
            x="n_cals",
            y="metric_value",
            hue="method",
            style="method",
            hue_order=METHOD_ORDER,
            style_order=METHOD_ORDER,
            dashes=False,
            palette=METHOD_COLORS,
            markers=METHOD_MARKERS,
            ax=ax,
            linewidth = 2,
            markersize = 8
        )

        # Add fancy gray panel title
        rect = patches.Rectangle(
            (0, 1.02), 1, 0.14,
            transform = ax.transAxes,
            color = "#E0E0E0",
            clip_on = False,
            zorder = -1
        )
        ax.add_patch(rect)

        # Axis scaling
        ax.set_xscale("log")
        ticks = sorted(plot_df["n_cals"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Calibration Sample Size")
        ax.set_ylabel("Mean")
        

        if metric == "coverage_vol_avg" and dim > 2:
            ax.set_yscale("log")
        if metric == "test_coverage_avg":
            #ax.axhline(0.90, color="green", linestyle="--", linewidth=1)
            ax.set_ylim(ylim)

        ax.set_title(title)
        ax.grid(True)

    # Legend (single shared)
    if include_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        methods_drawn = sorted(long_df["method"].unique())   # or filtered per subplot

        filtered_handles = []
        filtered_labels = []

        for h, lbl in zip(handles, labels):
            if lbl in methods_drawn:
                filtered_handles.append(h)
                filtered_labels.append(lbl)

        # Global legend
        fig.legend(
            filtered_handles, filtered_labels,
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=False,
            ncol=ncols,
            title = "Method"
        )
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, axes

def heavy_t_comparison(
    df_dict, dim, sample,
    include_runtime=True,
    include_legend=True,
    figsize=(12, 3),
    legend_bbox=(1.02, 0.5),
    ncols=2,
    direction = "Horizontal",
    ylim = (0.6, 1)
):

    # Gather all data for n_dim == dim
    long_df = prepare_long_form(df_dict, t_dist=True)
    df = long_df[(long_df["n_dim"] == dim) & (long_df["n_cals"] == sample)]

    # Panels to plot
    panels = ["test_coverage_avg", "coverage_vol_avg"]
    titles = ["Coverage", "Volume"]

    # Add Runtime data
    if include_runtime:
        panels.append("runtime_avg")
        titles.append("Runtime (log10-scale)")

    # Create figure
    if direction == "Horizontal":
        fig, axes = plt.subplots(
            1, len(panels),
            figsize=figsize,
            sharex=True
        )
    else:
        fig, axes = plt.subplots(
            len(panels), 1,
            figsize=figsize,
            sharex=True
        )

    # Enumerate the axes
    axes = axes.flatten()

    for ax, metric, title in zip(axes, panels, titles):

        plot_df = df[(df["metric_name"] == metric) & (df["df"] < 10)].copy()

        # Runtime → log10 scale
        if metric == "runtime_avg":
            plot_df["metric_value"] = np.log10(plot_df["metric_value"])


        # Main seaborn line plot
        sns.lineplot(
            data=plot_df,
            x="df",
            y="metric_value",
            hue="method",
            style="method",
            hue_order=METHOD_ORDER,
            style_order=METHOD_ORDER,
            dashes=False,
            palette=METHOD_COLORS,
            markers=METHOD_MARKERS,
            ax=ax,
            linewidth = 2,
            markersize = 8
        )

        # Add fancy gray panel title
        rect = patches.Rectangle(
            (0, 1.02), 1, 0.14,
            transform = ax.transAxes,
            color = "#E0E0E0",
            clip_on = False,
            zorder = -1
        )
        ax.add_patch(rect)

        # Axis scaling
        ax.set_xscale("log")
        ticks = sorted(plot_df["df"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Degree of Freedoms")
        ax.set_ylabel("Mean")
        

        if metric == "coverage_vol_avg" and dim > 2:
            ax.set_yscale("log")
        if metric == "test_coverage_avg":
            #ax.axhline(0.90, color="green", linestyle="--", linewidth=1)
            ax.set_ylim(ylim)

        ax.set_title(title)
        ax.grid(True)

    # Legend (single shared)
    if include_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        methods_drawn = sorted(long_df["method"].unique())   # or filtered per subplot

        filtered_handles = []
        filtered_labels = []

        for h, lbl in zip(handles, labels):
            if lbl in methods_drawn:
                filtered_handles.append(h)
                filtered_labels.append(lbl)

        # Global legend
        fig.legend(
            filtered_handles, filtered_labels,
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=False,
            ncol=ncols,
            title = "Method"
        )
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    return fig, axes

def compare_across_dims(
    df_dict, figsize=(12, 3),
    include_legend=True,
    legend_bbox = (1.02, 0.5),
    ncols=2,
    ylim = (0.7, 1)
):
    # ------------------------------------------
    # Convert df_dict into a long-form dataframe
    # ------------------------------------------
    # Gather all data for n_dim == dim
    long_df = prepare_long_form(df_dict)

    # ------------------------------------------
    # Prepare subplots
    # ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Panels to plot
    panels = ["test_coverage_avg", "coverage_vol_avg", "runtime_avg"]
    titles = ["Coverage", "Volume", "Runtime"]

    axes = axes.flatten()

    for ax, metric, title in zip(axes, panels, titles):

        plot_df = long_df[long_df["metric_name"] == metric].copy()

        # Runtime → log10 scale
        if metric == "runtime_avg":
            plot_df["metric_value"] = np.log10(plot_df["metric_value"])


        sns.lineplot(
                data=plot_df,
                x="n_dim",
                y="metric_value",
                hue="method",
                style="method",
                hue_order=METHOD_ORDER,
                style_order=METHOD_ORDER,
                dashes=False,
                palette=METHOD_COLORS,
                markers=METHOD_MARKERS,
                ax=ax,
                linewidth = 2,
                markersize = 8
            )

        # Add fancy gray panel title
        rect = patches.Rectangle(
                (0, 1.02), 1, 0.14,
                transform = ax.transAxes,
                color = "#E0E0E0",
                clip_on = False,
                zorder = -1
        )
        ax.add_patch(rect)

        # Axis scaling
        ax.set_xscale("log")
        ticks = sorted(plot_df["n_dim"].unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks)
        ax.set_xlabel("Dimensions")
        ax.set_ylabel("Mean")


        if metric == "coverage_vol_avg":
            ax.set_yscale("log")
        if metric == "test_coverage_avg":
            #ax.axhline(0.90, color="green", linestyle="--", linewidth=1)
            ax.set_ylim(ylim)
        if metric == "runtime_avg":
            ax.set_ylabel(r"$\log_{10}$(Mean)")

        ax.set_title(title)
        ax.grid(True)

    
    # Legend (single shared)
    if include_legend:
        handles, labels = axes[0].get_legend_handles_labels()
        methods_drawn = sorted(long_df["method"].unique())   # or filtered per subplot

        filtered_handles = []
        filtered_labels = []

        for h, lbl in zip(handles, labels):
            if lbl in methods_drawn:
                filtered_handles.append(h)
                filtered_labels.append(lbl)

        # Global legend
        fig.legend(
            filtered_handles, filtered_labels,
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=False,
            ncol=ncols,
            title = "Method"
        )
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    fig.tight_layout(rect=[0, 0, 0.85, 1])
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

def generate_latex_table_from_csvs(
    file_dict, dim, 
    noise_type, 
    method_exlude = ["Naive", "Emp. Copula"],
    filename="table_dX.tex"
):

    combined = []

    # Combine all CSV results
    for method, data in file_dict.items():
        df = data.copy()
        df["method"] = method
        combined.append(df)

    df_all = pd.concat(combined)

    # Filter by dimension
    df_all = df_all[df_all["n_dim"] == dim]
    df_all = df_all.sort_values(by=["n_cals", "method"])

    lines = []
    lines.append("\\begin{tabular}{l l c c c}")
    lines.append("\\toprule")
    lines.append("(n, d, Noise) & Method & Coverage & Volume & Runtime \\\\")
    lines.append("\\midrule")

    for n_cals in df_all["n_cals"].unique():

        subset = df_all[df_all["n_cals"] == n_cals]
        n_methods = len(subset)

        # --- EXCLUDE CERTAIN METHODS FROM MIN-VOLUME SEARCH ---
        subset_for_min = subset[
            ~subset["method"].isin(["Naive", "Emp. Copula"])
        ]
        if subset_for_min.empty:  # safety fallback
            subset_for_min = subset

        min_vol = subset_for_min["coverage_vol_avg"].min()
        # ------------------------------------------------------

        for idx, (_, row) in enumerate(subset.iterrows()):

            # Multirow label for n_cals
            cal_label = (
                f"\\multirow{{{n_methods}}}{{*}}{{({int(n_cals)}, {int(dim)}, {noise_type})}}"
                if idx == 0 else ""
            )

            # Coverage formatting
            coverage_avg = row['test_coverage_avg']
            coverage_std = row['test_coverage_1std']
            coverage_text = f"{coverage_avg:.3f} ({coverage_std:.3f})"

            # Volume formatting
            vol_mean = row['coverage_vol_avg']
            vol_std = row['coverage_vol_1std']
            volume_text = f"{vol_mean:.3e}({vol_std:.3e})"

            # ========= BLUE HIGHLIGHT FOR FAILED COVERAGE =========
            if coverage_avg < 0.895:
                coverage_text = f"\\textcolor{{red}}{{{coverage_text}}}"
            # ======================================================

            # ========= RED HIGHLIGHT FOR MINIMUM VOLUME ===========
            if (
                row["method"] not in ["Naive", "Emp. Copula"] 
                and vol_mean == min_vol
            ):
                volume_text = f"\\textbf{{{volume_text}}}"
            # ======================================================

            # Runtime formatting
            runtime_text = f"{row['runtime_avg']:.3f}"

            # Compose row
            lines.append(
                f"{cal_label} & {row['method']} & "
                f"{coverage_text} & {volume_text} & {runtime_text} \\\\"
            )

        lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    # Output filename
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
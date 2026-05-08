import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
from pathlib import Path

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

METHOD_DISPLAY_NAMES = {
    "Point_CHR": "Point CHR",
    "Point CHR": "Point CHR",
    "CQHR": "CQHR",
    "QCH": "CQHR",
    "Quantile CHR": "CQHR",
    "TSCP_R": "TSCP (Ours)",
    "TSCP": "TSCP (Ours)",
    "TSCP_S": "TSCP-S",
    "TSCP-S": "TSCP-S",
    "TSCP_GWC": "TSCP-GWC",
    "TSCP-GWC": "TSCP-GWC",
    "TSCP_LWC": "TSCP-LWC",
    "TSCP-LWC": "TSCP-LWC",
    "Unscaled": "Unscaled Max",
    "Unscaled Max": "Unscaled Max",
    "Empirical_copula": "Emp. Copula",
    "Emp. copula": "Emp. Copula",
    "Emp. Copula": "Emp. Copula",
    "Population_oracle": "Pop. Oracle",
    "Pop. Oracle": "Pop. Oracle",
    "Bonferroni": "Bonferroni",
    "Naive": "Naive",
}

REVIEWER_METHOD_ORDER = [
    "TSCP (Ours)",
    "CQHR",
    "Point CHR",
    "Unscaled Max",
    "Emp. Copula",
    "Bonferroni",
    "TSCP-GWC",
    "TSCP-S",
    "TSCP-LWC",
    "Pop. Oracle",
    "Naive",
]

REVIEWER_METHOD_COLORS = {
    method: METHOD_COLORS.get(method, color)
    for method, color in zip(
        REVIEWER_METHOD_ORDER,
        sns.color_palette("deep", n_colors=len(REVIEWER_METHOD_ORDER)),
    )
}

REVIEWER_METHOD_MARKERS = {
    method: METHOD_MARKERS.get(method, marker)
    for method, marker in zip(REVIEWER_METHOD_ORDER, MARKERS)
}


def method_name_coverter(method_list):
    """Backward-compatible method display-name converter."""
    return {method: METHOD_DISPLAY_NAMES.get(method, method) for method in method_list}


def normalize_experiment_columns(df):
    """
    Normalize older and newer experiment outputs to a shared column convention.

    The original plotting helpers used `n_cals`, `Methods`, and `coverage_vol`,
    while the newer experiment runners use `n_cal`, `method`, and
    `coverage_vol_avg`. This helper lets the report notebook work with both.
    """
    df = df.copy()
    rename_map = {
        "n_cals": "n_cal",
        "Methods": "method",
        "coverage_vol": "coverage_vol_avg",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    if "method" in df.columns:
        df["method_display"] = df["method"].map(METHOD_DISPLAY_NAMES).fillna(df["method"])
    return df


def order_methods(methods, preferred_order=None):
    """Return method display names in a stable, publication-friendly order."""
    preferred_order = preferred_order or REVIEWER_METHOD_ORDER
    methods = [METHOD_DISPLAY_NAMES.get(method, method) for method in methods]
    known = [method for method in preferred_order if method in methods]
    extra = sorted(method for method in methods if method not in preferred_order)
    return known + extra


def _available_methods(df, methods=None):
    df = normalize_experiment_columns(df)
    if "method_display" not in df.columns:
        return []
    available = df["method_display"].dropna().unique().tolist()
    if methods is not None:
        requested = [METHOD_DISPLAY_NAMES.get(method, method) for method in methods]
        available = [method for method in requested if method in available]
    return order_methods(available)


def _method_palette(methods):
    palette = {}
    fallback = sns.color_palette("deep", n_colors=max(len(methods), 1))
    for idx, method in enumerate(methods):
        palette[method] = REVIEWER_METHOD_COLORS.get(method, fallback[idx])
    return palette


def _method_markers(methods):
    fallback = ["o", "^", "s", "D", "v", "<", ">", "P", "X", "H"]
    return {
        method: REVIEWER_METHOD_MARKERS.get(method, fallback[idx % len(fallback)])
        for idx, method in enumerate(methods)
    }


def _format_tick_label(value):
    """Format numeric ticks plainly, avoiding 10^k labels on log-scaled axes."""
    if pd.isna(value):
        return ""
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def _set_plain_numeric_x_ticks(ax, values):
    """Use the observed x values as plain-number tick labels."""
    values = pd.Series(values).dropna().unique()
    if len(values) == 0:
        return
    try:
        values = sorted(float(value) for value in values)
    except (TypeError, ValueError):
        return
    ax.set_xticks(values)
    ax.set_xticklabels([_format_tick_label(value) for value in values])


def _set_plain_categorical_x_labels(ax):
    """Format categorical tick labels that happen to be numeric."""
    ticks = ax.get_xticks()
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        try:
            labels.append(_format_tick_label(float(text)))
        except ValueError:
            labels.append(text)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)


def _add_panel_title(ax, title):
    """Draw the gray title strip used by the existing paper-style figures."""
    rect = patches.Rectangle(
        (0, 1.02),
        1,
        0.14,
        transform=ax.transAxes,
        color="#E0E0E0",
        clip_on=False,
        zorder=-1,
    )
    ax.add_patch(rect)
    ax.set_title("")
    ax.text(
        0.5,
        1.09,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=plt.rcParams.get("axes.titlesize", 11),
    )


def _finish_shared_legend(fig, axes, legend_bbox=(1.02, 0.5), ncols=1):
    axes = np.asarray(axes).flatten()
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=legend_bbox,
            frameon=False,
            ncol=ncols,
            title="Method",
        )
    for ax in axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()


def reviewer_summary_table(
    summary_df,
    group_cols=None,
    methods=None,
    include_runtime=True,
    formatted=True,
    target_coverage=0.9,
):
    """
    Build a tidy summary table for reviewer reports.

    If `formatted=True`, coverage and volume are displayed as mean(std).
    If `formatted=False`, the returned DataFrame keeps numeric columns for
    downstream export and plotting.
    """
    df = normalize_experiment_columns(summary_df)
    if group_cols is None:
        candidate_cols = [
            "alpha",
            "n_dim",
            "n_cal",
            "noise_type",
            "correlation",
            "correlation_structure",
            "noise_ratio",
            "df",
            "score_transform",
            "base_interval_alpha",
            "shift_constant",
        ]
        group_cols = [col for col in candidate_cols if col in df.columns]
    if methods is not None:
        method_order = order_methods(methods)
        df = df[df["method_display"].isin(method_order)]
    else:
        method_order = _available_methods(df)

    cols = [
        *group_cols,
        "method_display",
        "n_trials",
        "test_coverage_avg",
        "test_coverage_1std",
        "coverage_vol_avg",
        "coverage_vol_1std",
    ]
    if include_runtime and "runtime_avg" in df.columns:
        cols.append("runtime_avg")
    cols = [col for col in cols if col in df.columns]
    out = df[cols].copy()
    out["method_display"] = pd.Categorical(
        out["method_display"],
        categories=method_order,
        ordered=True,
    )
    out = out.sort_values([*group_cols, "method_display"]).reset_index(drop=True)

    if not formatted:
        return out

    def _mean_std(mean, std, sci=False):
        if pd.isna(mean):
            return ""
        if sci or np.isinf(mean) or abs(mean) >= 1e4 or (abs(mean) < 1e-2 and mean != 0):
            mean_str = f"{mean:.3e}" if np.isfinite(mean) else "inf"
            std_str = f"{std:.3e}" if pd.notna(std) and np.isfinite(std) else ""
        else:
            mean_str = f"{mean:.3f}"
            std_str = f"{std:.3f}" if pd.notna(std) else ""
        return f"{mean_str}({std_str})" if std_str else mean_str

    display = out[[*group_cols, "method_display", "n_trials"]].copy()
    display = display.rename(columns={"method_display": "method"})
    display["coverage"] = [
        _mean_std(mean, std)
        for mean, std in zip(out["test_coverage_avg"], out["test_coverage_1std"])
    ]
    display["volume"] = [
        _mean_std(mean, std, sci=True)
        for mean, std in zip(out["coverage_vol_avg"], out["coverage_vol_1std"])
    ]
    if include_runtime and "runtime_avg" in out.columns:
        display["runtime"] = out["runtime_avg"].map(lambda value: f"{value:.3f}")
    if target_coverage is not None and "test_coverage_avg" in out.columns:
        display["meets_target"] = out["test_coverage_avg"] >= target_coverage
    return display


def pivot_metric_table(
    summary_df,
    metric,
    index_cols,
    methods=None,
    value_format=None,
):
    """Create a wide method-by-column table for one metric."""
    df = normalize_experiment_columns(summary_df)
    method_order = _available_methods(df, methods=methods)
    table = df.pivot_table(
        index=index_cols,
        columns="method_display",
        values=metric,
        aggfunc="mean",
    )
    table = table.reindex(columns=method_order)
    if value_format is not None:
        table = table.apply(
            lambda col: col.map(lambda value: "" if pd.isna(value) else value_format(value))
        )
    return table.reset_index()


def add_coordinate_noise_levels(coordinate_df, noise_levels=None, noise_col="noise_level"):
    """
    Add coordinate-wise noise levels to a coordinate summary table.

    If `noise_levels` is omitted, the helper assumes the default synthetic
    design used in the paper and experiment runner: `[d, d-1, ..., 1]`.
    Pass an explicit array for custom designs such as geometric sweeps.
    """
    df = coordinate_df.copy()
    if "coordinate" not in df.columns:
        raise ValueError("coordinate_df must contain a coordinate column.")

    if noise_levels is not None:
        noise_levels = np.asarray(noise_levels, dtype=float)
        coordinate_index = df["coordinate"].astype(int).to_numpy() - 1
        if np.any(coordinate_index < 0) or np.any(coordinate_index >= len(noise_levels)):
            raise ValueError("noise_levels length must cover all coordinate indices.")
        df[noise_col] = noise_levels[coordinate_index]
        return df

    if "n_dim" not in df.columns:
        raise ValueError("Either noise_levels or an n_dim column must be provided.")
    df[noise_col] = df["n_dim"].astype(float) - df["coordinate"].astype(float) + 1
    return df


def coordinate_length_table(
    coordinate_summary_df,
    methods=None,
    noise_levels=None,
    coordinate_scale=2.0,
    value_format=None,
):
    """
    Create a wide coordinate-wise length table with an explicit noise-level column.

    For absolute residual scores, use `coordinate_scale=2` to display full
    outcome-space interval lengths instead of residual-space half-widths.
    """
    df = normalize_experiment_columns(coordinate_summary_df)
    df = add_coordinate_noise_levels(df, noise_levels=noise_levels)
    df = df.copy()
    df["coordinate_interval_length_avg"] = df["coordinate_length_avg"] * coordinate_scale

    index_cols = [
        col
        for col in [
            "correlation",
            "correlation_structure",
            "n_cal",
            "score_transform",
            "base_interval_alpha",
            "shift_constant",
            "coordinate",
            "noise_level",
        ]
        if col in df.columns
    ]
    return pivot_metric_table(
        df,
        metric="coordinate_interval_length_avg",
        index_cols=index_cols,
        methods=methods,
        value_format=value_format,
    )


def summarize_infinite_volume_rate(trial_df):
    """Summarize how often each method returns an infinite prediction volume."""
    df = normalize_experiment_columns(trial_df)
    if "coverage_volume" not in df.columns:
        raise ValueError("trial_df must contain a coverage_volume column.")
    group_cols = [
        col
        for col in [
            "alpha",
            "n_dim",
            "n_cal",
            "method",
            "method_display",
            "noise_type",
            "correlation",
            "correlation_structure",
            "noise_ratio",
            "df",
        ]
        if col in df.columns
    ]
    out = (
        df.assign(infinite_volume=np.isinf(df["coverage_volume"]))
        .groupby(group_cols, as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            infinite_volume_rate=("infinite_volume", "mean"),
        )
    )
    method_order = _available_methods(out)
    out["method_display"] = pd.Categorical(
        out["method_display"],
        categories=method_order,
        ordered=True,
    )
    return out.sort_values([col for col in group_cols if col != "method"]).reset_index(drop=True)


def save_experiment_dataframes(dataframes, output_dir, prefix):
    """
    Save named DataFrames to stable CSV paths and return those paths.

    Parameters
    ----------
    dataframes : dict[str, pd.DataFrame]
        Mapping such as {"summary": df, "coordinate_summary": coord_df}.
    output_dir : str or Path
        Destination folder.
    prefix : str
        Filename prefix, for example "dependent_gaussian".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, df in dataframes.items():
        if df is None:
            continue
        path = output_dir / f"{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = path
    return paths


def load_experiment_dataframes(output_dir, prefix, names=None):
    """Load stable CSVs saved by `save_experiment_dataframes`."""
    output_dir = Path(output_dir)
    names = names or ["trial", "summary", "coordinate_trial", "coordinate_summary"]
    loaded = {}
    for name in names:
        path = output_dir / f"{prefix}_{name}.csv"
        if path.exists():
            loaded[name] = pd.read_csv(path)
    return loaded


def plot_reviewer_metric_grid(
    summary_df,
    x="n_cal",
    methods=None,
    target_coverage=0.9,
    include_runtime=True,
    log_x=False,
    log_volume=True,
    figsize=(12, 3.4),
    title=None,
    legend_bbox=(1.02, 0.5),
    ncols=1,
):
    """Plot coverage, volume, and optionally runtime for reviewer experiments."""
    df = normalize_experiment_columns(summary_df)
    method_order = _available_methods(df, methods=methods)
    if methods is not None:
        df = df[df["method_display"].isin(method_order)]

    panels = [
        ("test_coverage_avg", "Coverage"),
        ("coverage_vol_avg", "Volume"),
    ]
    if include_runtime and "runtime_avg" in df.columns:
        panels.append(("runtime_avg", "Runtime"))

    fig, axes = plt.subplots(1, len(panels), figsize=figsize, sharex=False)
    axes = np.asarray(axes).flatten()
    palette = _method_palette(method_order)
    markers = _method_markers(method_order)

    for ax, (metric, panel_title) in zip(axes, panels):
        plot_df = df[[x, "method_display", metric]].copy()
        plot_df[metric] = plot_df[metric].replace([np.inf, -np.inf], np.nan)
        sns.lineplot(
            data=plot_df,
            x=x,
            y=metric,
            hue="method_display",
            style="method_display",
            hue_order=method_order,
            style_order=method_order,
            dashes=False,
            palette=palette,
            markers=markers,
            linewidth=2,
            markersize=8,
            ax=ax,
        )
        if log_x:
            ax.set_xscale("log")
        _set_plain_numeric_x_ticks(ax, plot_df[x])
        if metric == "coverage_vol_avg" and log_volume:
            ax.set_yscale("log")
        if metric == "runtime_avg":
            ax.set_yscale("log")
        if metric == "test_coverage_avg" and target_coverage is not None:
            ax.axhline(target_coverage, color="black", linestyle="--", linewidth=1)
            lower = max(0, min(target_coverage - 0.25, plot_df[metric].min() - 0.02))
            ax.set_ylim(lower, 1.02)
        ax.set_xlabel(x.replace("_", " "))
        ax.set_ylabel("Mean")
        ax.grid(True)
        _add_panel_title(ax, panel_title)

    if title:
        fig.suptitle(title, y=1.08)
    _finish_shared_legend(fig, axes, legend_bbox=legend_bbox, ncols=ncols)
    fig.tight_layout(rect=[0, 0, 0.84, 1])
    return fig, axes


def plot_coordinate_length_profile(
    coordinate_summary_df,
    methods=None,
    noise_levels=None,
    coordinate_scale=2.0,
    figsize=(8, 4),
    title=None,
    legend_bbox=(1.02, 0.5),
    ncols=1,
):
    """
    Plot average coordinate-wise interval lengths.

    The experiment runner stores residual-space upper thresholds. Use
    `coordinate_scale=2` to display full outcome interval lengths for
    absolute residual scores.
    """
    df = normalize_experiment_columns(coordinate_summary_df)
    df = add_coordinate_noise_levels(df, noise_levels=noise_levels)
    method_order = _available_methods(df, methods=methods)
    if methods is not None:
        df = df[df["method_display"].isin(method_order)]
    df = df.copy()
    df["coordinate_interval_length_avg"] = df["coordinate_length_avg"] * coordinate_scale

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.barplot(
        data=df,
        x="coordinate",
        y="coordinate_interval_length_avg",
        hue="method_display",
        hue_order=method_order,
        palette=_method_palette(method_order),
        errorbar=None,
        ax=ax,
    )
    ax.set_xlabel("Coordinate")
    ax.set_ylabel("Average interval length")
    ax.grid(True)
    if "noise_level" in df.columns:
        ax2 = ax.secondary_xaxis("bottom")
        coordinates = sorted(df["coordinate"].dropna().unique())
        ax2.set_xticks(np.arange(len(coordinates)))
        noise_by_coordinate = (
            df.drop_duplicates("coordinate")
            .set_index("coordinate")
            .loc[coordinates, "noise_level"]
        )
        ax2.set_xticklabels([_format_tick_label(value) for value in noise_by_coordinate])
        ax2.set_xlabel("Noise level")
        ax2.spines["bottom"].set_position(("outward", 34))
    _add_panel_title(ax, title or "Coordinate-wise Length")
    _finish_shared_legend(fig, [ax], legend_bbox=legend_bbox, ncols=ncols)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    return fig, ax


def plot_metric_sweep(
    summary_df,
    x,
    metric="coverage_vol_avg",
    methods=None,
    target_coverage=0.9,
    log_x=False,
    log_y=None,
    plot_kind="bar",
    figsize=(7, 4),
    title=None,
    legend_bbox=(1.02, 0.5),
    ncols=1,
):
    """
    Plot one metric against a design parameter such as correlation or noise ratio.

    Defaults to grouped bars because these reviewer sweeps compare discrete
    design settings. Use `plot_kind="line"` only when the x-axis is a genuine
    sample-size progression or another ordered curve.
    """
    df = normalize_experiment_columns(summary_df)
    method_order = _available_methods(df, methods=methods)
    if methods is not None:
        df = df[df["method_display"].isin(method_order)]
    df = df.copy()
    df[metric] = df[metric].replace([np.inf, -np.inf], np.nan)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if plot_kind not in {"bar", "line"}:
        raise ValueError("plot_kind must be either 'bar' or 'line'.")

    if plot_kind == "line":
        sns.lineplot(
            data=df,
            x=x,
            y=metric,
            hue="method_display",
            style="method_display",
            hue_order=method_order,
            style_order=method_order,
            dashes=False,
            palette=_method_palette(method_order),
            markers=_method_markers(method_order),
            linewidth=2,
            markersize=8,
            ax=ax,
        )
        if log_x:
            ax.set_xscale("log")
        _set_plain_numeric_x_ticks(ax, df[x])
    else:
        sns.barplot(
            data=df,
            x=x,
            y=metric,
            hue="method_display",
            hue_order=method_order,
            palette=_method_palette(method_order),
            errorbar=None,
            ax=ax,
        )
        _set_plain_categorical_x_labels(ax)

    if log_y is None:
        log_y = metric in {"coverage_vol_avg", "runtime_avg"}
    if log_y:
        ax.set_yscale("log")
    if metric == "test_coverage_avg" and target_coverage is not None:
        ax.axhline(target_coverage, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(max(0, min(target_coverage - 0.25, df[metric].min() - 0.02)), 1.02)
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True)
    _add_panel_title(ax, title or metric.replace("_", " ").title())
    _finish_shared_legend(fig, [ax], legend_bbox=legend_bbox, ncols=ncols)
    fig.tight_layout(rect=[0, 0, 0.82, 1])
    return fig, ax


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

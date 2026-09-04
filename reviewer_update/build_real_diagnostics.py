"""Summarize completed real-data diagnostic runs and generate vector figures."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/mpl-diagnostics"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import beta

from reviewer_update.run_real_diagnostics import DATASETS, METHODS, OUT, TABLES

FIGURES = ROOT / "reviewer_update/figures"
LATEX = ROOT / "multi_target_scaling_latex"
LABELS = {"Empirical_copula": "Emp. Copula", "Unscaled": "Unscaled Max",
          "Point_CHR": "Point CHR", "TSCP_R": "TSCP (Ours)"}
COLORS = {"Empirical_copula": "#8C8C8C", "Unscaled": "#9C755F",
          "Point_CHR": "#4E79A7", "TSCP_R": "#E15759"}
DATA_COLORS = ["#4E79A7", "#E15759", "#59A14F", "#7B61A8", "#B07A26", "#3E9A9A"]
FIGURE_NAMES = ["fig_body_real_coordinate_bars", "fig_body_real_search_diagnostics",
                "fig_app_real_marginal_coverage", "fig_app_real_search_alpha"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.7, "lines.linewidth": 1.4,
    "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42})


def clean_axis(ax, title):
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#d0d0d0", linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#b0b0b0")
    ax.set_title(title, fontsize=8, pad=6, backgroundcolor="#e6e6e6")


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    shutil.copy2(path, LATEX / "figures" / path.name)


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    assert not np.isnan(values).any()
    finite = np.isfinite(values)
    mean = np.mean(values) if finite.all() else np.inf
    std = np.std(values, ddof=1) if finite.all() and len(values) > 1 else np.nan
    return mean, std, np.count_nonzero(~finite) / len(values)


def summaries():
    manifest = json.loads((OUT / "run_manifest.json").read_text())
    trials = manifest["trials_per_dataset"]
    assert manifest["datasets"] == DATASETS and trials >= 30, "Do not publish the pilot."
    joint = pd.read_csv(TABLES / "real_joint_trials.csv")
    coords = pd.read_csv(TABLES / "real_coordinate_trials.csv")
    coords = coords.drop(columns=["unscaled_full_length", "paired_length_ratio"], errors="ignore")
    branches = pd.read_csv(TABLES / "real_search_trials.csv")
    stress = pd.read_csv(TABLES / "real_search_alpha_trials.csv")
    assert len(branches) == len(DATASETS) * trials
    assert not branches.duplicated(["dataset", "trial"]).any()
    assert set(branches.trial) == set(range(trials))
    assert len(joint) == len(DATASETS) * trials * len(METHODS)
    assert len(stress) == len(branches) * len(manifest["stress_alpha"])
    denominators = coords.loc[coords.method == "Unscaled", ["dataset", "trial", "coordinate", "full_length"]]
    denominators = denominators.rename(columns={"full_length": "unscaled_full_length"})
    coords = coords.merge(denominators, on=["dataset", "trial", "coordinate"], validate="many_to_one")
    assert (coords.unscaled_full_length > 0).all() and np.isfinite(coords.unscaled_full_length).all()
    coords["paired_length_ratio"] = coords.full_length / coords.unscaled_full_length
    coords.to_csv(TABLES / "real_coordinate_trials.csv", index=False)
    coordinate_summary = []
    for key, group in coords.groupby(["dataset", "method", "coordinate", "target"], sort=False):
        assert len(group) == trials
        mean, std, inf_rate = summarize_values(group.full_length)
        ratio, ratio_std, _ = summarize_values(group.paired_length_ratio)
        coverage = group.marginal_coverage.mean()
        coverage_std = group.marginal_coverage.std(ddof=1)
        coordinate_summary.append(dict(zip(["dataset", "method", "coordinate", "target"], key)) | {
            "n_trials": trials, "full_length_mean": mean, "full_length_sd": std,
            "paired_length_ratio_mean": ratio, "paired_length_ratio_sd": ratio_std,
            "infinite_fraction": inf_rate, "marginal_coverage_mean": coverage,
            "marginal_coverage_sd": coverage_std,
            "marginal_mc95_lower": max(0, coverage - 1.96 * coverage_std / np.sqrt(trials)),
            "marginal_mc95_upper": min(1, coverage + 1.96 * coverage_std / np.sqrt(trials))})
    cs = pd.DataFrame(coordinate_summary)
    cs.to_csv(TABLES / "real_coordinate_summary.csv", index=False)
    js = joint.groupby(["dataset", "method"], sort=False).agg(
        n_trials=("trial", "size"), joint_coverage_mean=("joint_coverage", "mean"),
        joint_coverage_sd=("joint_coverage", "std"), infinite_fraction=("infinite_region", "mean")).reset_index()
    js["target_within_one_sd"] = abs(js.joint_coverage_mean - 0.9) <= js.joint_coverage_sd
    js["mc95_lower"] = js.joint_coverage_mean - 1.96 * js.joint_coverage_sd / np.sqrt(trials)
    js["mc95_upper"] = js.joint_coverage_mean + 1.96 * js.joint_coverage_sd / np.sqrt(trials)
    js.to_csv(TABLES / "real_joint_summary.csv", index=False)
    group_keys = ["dataset", "alpha"]
    aggregate = []
    for key, group in stress.groupby(group_keys, sort=False):
        regular = group.loc[group.fallback == 0]
        fallback_count = int(group.fallback.sum())
        backward_count = int(group.any_backward.sum())
        row = dict(zip(group_keys, key)) | {"n_trials": len(group), "n_cal": int(group.n_cal.iloc[0]),
            "n_dim": int(group.n_dim.iloc[0]), "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(group), "backward_run_count": backward_count,
            "backward_run_rate": backward_count / len(group),
            "nonfallback_runs": len(regular), "searched_coordinates": int(regular.n_dim.sum()),
            "backward_coordinates": int(group.backward_coordinates.sum()),
            "binary_candidates": int(group.binary_candidates.sum()),
            "backward_candidates": int(group.backward_candidates.sum()),
            "runtime_ms_mean": group.runtime_ms.mean(), "runtime_ms_sd": group.runtime_ms.std(ddof=1),
            "runtime_ms_median": group.runtime_ms.median()}
        for event, count in [("fallback", fallback_count), ("backward", backward_count)]:
            row[f"{event}_one_sided95_upper"] = 1.0 if count == len(group) else beta.ppf(0.95, count+1, len(group)-count)
        for regime in ["binary_only", "any_backward", "fallback"]:
            subset = group.loc[group.regime == regime]
            row[f"{regime}_count"] = len(subset)
            row[f"{regime}_runtime_ms_mean"] = subset.runtime_ms.mean()
            row[f"{regime}_runtime_ms_sd"] = subset.runtime_ms.std(ddof=1)
            row[f"{regime}_runtime_ms_median"] = subset.runtime_ms.median()
        aggregate.append(row)
    bs = pd.DataFrame(aggregate)
    bs.to_csv(TABLES / "real_search_alpha_summary.csv", index=False)
    bs.loc[np.isclose(bs.alpha, 0.1)].to_csv(TABLES / "real_search_summary.csv", index=False)
    return manifest, cs, js, bs


def coordinate_bars(cs):
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 6.15), squeeze=False)
    for ax, dataset in zip(axes.flat, DATASETS):
        part = cs.loc[cs.dataset == dataset]
        d = int(part.coordinate.max())
        x = np.arange(1, d+1)
        width = 0.19
        for i, method in enumerate(METHODS):
            values = part.loc[part.method == method].sort_values("coordinate").paired_length_ratio_mean.to_numpy()
            finite = np.isfinite(values)
            ax.bar(x[finite] + (i-1.5)*width, values[finite], width=width,
                   color=COLORS[method], label=LABELS[method])
        ax.axhline(1, color="#333333", linewidth=0.7, linestyle="--")
        clean_axis(ax, dataset)
        ax.set_xticks(x)
        ax.set_xlim(0.4, d+0.6)
        finite_values = part.paired_length_ratio_mean.to_numpy()
        finite_values = finite_values[np.isfinite(finite_values)]
        ax.set_ylim(0, max(1.0, finite_values.max()) * 1.22)
        if dataset == "stock":
            ax.text(0.98, 0.92, "Point CHR: infinite", ha="right", va="top",
                    transform=ax.transAxes, color=COLORS["Point_CHR"], fontsize=7)
        ax.set_xlabel("Outcome coordinate")
        ax.set_ylabel("Mean paired length ratio")
    handles = [Patch(facecolor=COLORS[method], label=LABELS[method]) for method in METHODS]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(hspace=0.65, wspace=0.28, bottom=0.10, top=0.96)
    save(fig, FIGURE_NAMES[0])


def marginal_coverage(cs, js):
    fig, axes = plt.subplots(3, 2, figsize=(7.25, 6.15), squeeze=False)
    for ax, dataset in zip(axes.flat, DATASETS):
        part = cs.loc[cs.dataset == dataset]
        for method, marker in zip(METHODS, ["+", "<", "o", "D"]):
            rows = part.loc[part.method == method].sort_values("coordinate")
            ax.plot(rows.coordinate, rows.marginal_coverage_mean, color=COLORS[method],
                    marker=marker, label=LABELS[method], clip_on=False)
        clean_axis(ax, dataset)
        ax.axhline(0.9, color="#333333", linewidth=0.7, linestyle="--")
        ax.set_ylim(0.6, 1.0)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        d = int(part.coordinate.max())
        ax.set_xlim(0.7, d+0.3)
        ax.set_xticks(np.arange(1, d+1))
        ax.set_xlabel("Outcome coordinate")
        ax.set_ylabel("Marginal coverage")
        value = js.loc[(js.dataset == dataset) & (js.method == "TSCP_R"), "joint_coverage_mean"].iloc[0]
        ax.text(0.02, 0.06, f"TSCP joint: {value:.3f}", transform=ax.transAxes, fontsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(hspace=0.65, wspace=0.28, bottom=0.10, top=0.96)
    save(fig, FIGURE_NAMES[2])


def search_figures(bs):
    main = bs.loc[np.isclose(bs.alpha, 0.1)].set_index("dataset").loc[DATASETS]
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 2.65))
    x = np.arange(len(DATASETS))
    denominator = main.searched_coordinates.to_numpy(dtype=float)
    binary = np.divide(main.binary_candidates, denominator, out=np.full(len(main), np.nan), where=denominator>0)
    backward = np.divide(main.backward_candidates, denominator, out=np.full(len(main), np.nan), where=denominator>0)
    axes[0].bar(x, binary, color="#4E79A7", label="Binary")
    axes[0].bar(x, backward, bottom=binary, color="#E15759", label="Backward")
    axes[0].set_ylabel("Mean candidate cells per coordinate")
    axes[0].legend(frameon=False, fontsize=7)
    clean_axis(axes[0], "Search work on non-fallback runs")
    colors = {"binary_only": "#4E79A7", "any_backward": "#E15759", "fallback": "#59A14F"}
    names = {"binary_only": "Binary only", "any_backward": "Any backward", "fallback": "Fallback"}
    observed = [regime for regime in colors if main[f"{regime}_runtime_ms_mean"].notna().any()]
    offsets = [0] if len(observed) == 1 else np.linspace(-0.25, 0.25, len(observed))
    for offset, regime in zip(offsets, observed):
        vals = main[f"{regime}_runtime_ms_mean"].to_numpy()
        mask = np.isfinite(vals)
        axes[1].bar(x[mask] + offset, vals[mask], width=0.6 if len(observed) == 1 else 0.24,
                    color=colors[regime], label=names[regime])
    clean_axis(axes[1], "Construction time conditional on regime")
    axes[1].set_ylabel("Mean of per-split medians (ms)")
    axes[1].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.set_xticks(x, DATASETS, rotation=25, ha="right")
        ax.set_ylim(bottom=0)
    axes[0].set_ylim(0, float(np.nanmax(binary + backward)) * 1.18)
    axes[1].set_ylim(0, float(main[[f"{regime}_runtime_ms_mean" for regime in observed]].max().max()) * 1.18)
    fig.tight_layout()
    save(fig, FIGURE_NAMES[1])

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.65))
    for dataset, color, marker in zip(DATASETS, DATA_COLORS, ["o", "s", "^", "D", "v", "P"]):
        rows = bs.loc[bs.dataset == dataset].sort_values("alpha")
        for ax, column in zip(axes, ["backward_run_rate", "fallback_rate", "runtime_ms_mean"]):
            ax.plot(rows.alpha, rows[column], marker=marker, color=color, label=dataset)
    for ax, title in zip(axes, ["Any backward search", "GWC fallback", "Overall construction time"]):
        clean_axis(ax, title)
        ax.set_xlabel("Miscoverage level")
        ax.set_xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    for ax in axes[:2]:
        ax.set_ylim(-0.015, 1.03)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Fraction of calibration splits")
    backward_top = max(0.05, float(bs.backward_run_rate.max()) * 1.2)
    axes[0].set_ylim(-backward_top * 0.02, backward_top)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1 if backward_top <= 0.1 else 0))
    axes[2].set_ylabel("Mean of per-split medians (ms)")
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, FIGURE_NAMES[3])


def write_latex_tables(js, bs):
    main = bs.loc[np.isclose(bs.alpha, 0.1)].set_index("dataset").loc[DATASETS]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Dataset & $n$ & \shortstack{Backward\\splits} & \shortstack{Backward\\coordinates} & \shortstack{Fallback\\splits} & \shortstack{Binary only\\(ms)} & \shortstack{Any backward\\(ms)} & \shortstack{Fallback\\(ms)} \\", r"\midrule"]
    for name, row in main.iterrows():
        count = int(row.n_trials)
        cells = [rf"\texttt{{{name}}}", str(int(row.n_cal)),
                 f"{int(row.backward_run_count)}/{count}",
                 f"{int(row.backward_coordinates)}/{int(row.searched_coordinates)}",
                 f"{int(row.fallback_count)}/{count}"]
        for regime in ["binary_only", "any_backward", "fallback"]:
            value = row[f"{regime}_runtime_ms_mean"]
            cells.append(f"{value:.3f}" if np.isfinite(value) else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "real_search_table.tex").write_text("\n".join(lines) + "\n")
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Dataset & Emp.\ Copula & Unscaled Max & Point CHR & TSCP \\", r"\midrule"]
    for name in DATASETS:
        cells = [rf"\texttt{{{name}}}"]
        for method in METHODS:
            row = js.loc[(js.dataset == name) & (js.method == method)].iloc[0]
            cells.append(f"${row.joint_coverage_mean:.3f}\\,({row.joint_coverage_sd:.3f})$")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (TABLES / "real_joint_table.tex").write_text("\n".join(lines) + "\n")


def main():
    manifest, cs, js, bs = summaries()
    coordinate_bars(cs)
    marginal_coverage(cs, js)
    search_figures(bs)
    write_latex_tables(js, bs)
    destination = LATEX / "experiment_data/real_diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    for path in TABLES.iterdir():
        if path.suffix not in {".csv", ".tex"}:
            continue
        shutil.copy2(path, destination / path.name)
    shutil.copy2(OUT / "run_manifest.json", destination / "run_manifest.json")
    print("JOINT COVERAGE\n", js.to_string(index=False))
    print("SEARCH AT ALPHA 0.1\n", bs.loc[np.isclose(bs.alpha, 0.1)].to_string(index=False))
    print("TSCP COORDINATES\n", cs.loc[cs.method == "TSCP_R"].to_string(index=False))
    print("Generated four PDF figures and copied all diagnostic CSV files into the manuscript folder.")


if __name__ == "__main__":
    main()

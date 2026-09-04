"""Restyle archived numerical summaries without changing or resimulating them.

Uses the same plotting helpers as the current synthetic and real-data figures.
The manifest records the exact values drawn and hashes of their source CSVs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewer_update import build_experiment_update as style
from reviewer_update import build_real_diagnostics as real
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

LATEX = ROOT / "multi_target_scaling_latex"
OUT = LATEX / "figures"
RECORDS = []

SPECS = [
    ("fig_body_abs_independent_gaussian", "baselines/gaussian_10d.pdf", "gaussian", "gaussian", 10,
     ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]),
    ("fig_body_abs_homogeneous_gaussian", "baselines/unit_gaussian_10d.pdf", "gaussian", "unit_gaussian", 10,
     ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]),
    ("fig_app_oracle_approximation", "ours/ours_approximation_2d.pdf", "laplace", "laplace", 2,
     ["Population_oracle", "Naive", "TSCP_S", "TSCP_R"]),
    ("fig_app_local_enclosure_2d", "ours/ours_containment_2d.pdf", "laplace", "laplace", 2,
     ["Population_oracle", "TSCP_GWC", "TSCP_LWC", "TSCP_R"]),
    ("fig_app_local_enclosure_10d", "ours/ours_containment_10d.pdf", "laplace", "laplace", 10,
     ["Population_oracle", "Naive", "TSCP_S", "TSCP_GWC", "TSCP_R"]),
]


def load_source(folder, suffix, methods, dimension=None):
    pieces, sources = [], []
    for method in methods:
        path = ROOT / "syn_exps" / folder / f"{method.lower()}_{suffix}.csv"
        frame = pd.read_csv(path).drop(columns=["Unnamed: 0"], errors="ignore")
        if dimension is not None:
            frame = frame.loc[frame.n_dim.eq(dimension)].copy()
        frame["method"] = method
        pieces.append(frame)
        sources.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.concat(pieces, ignore_index=True), sources


def coverage(ax, frame, x, methods, xlabel):
    style._lineplot(ax, frame, x, "test_coverage_avg", methods,
                    title="Joint coverage", xlabel=xlabel, ylabel="Mean", target=0.9)
    style._set_coverage_axis(ax)
    # Keep the common range without silently concealing values outside it.
    for method in methods:
        below = frame.loc[frame.method.eq(method) & frame.test_coverage_avg.lt(0.6)]
        if not below.empty:
            ax.scatter(below[x], np.full(len(below), 0.6), marker="v", s=28,
                       color=style.COLORS[style._display_name(method)],
                       edgecolor="white", linewidth=0.45, clip_on=False, zorder=5)


def finish(fig, axes, name, original, frame, sources, ncol=4):
    style._shared_legend(fig, axes, ncol=ncol, y=-0.1)
    path = style._save(fig, name + ".pdf", bottom=0.21)
    shutil.copy2(path, OUT / path.name)
    RECORDS.append({"figure": path.name, "replaces": original,
                    "sources": sources, "plotted_rows": json.loads(frame.to_json(orient="records", double_precision=15)),
                    "coverage_ylim": [0.6, 1.0], "error_bars": False,
                    "method_styles": {m: {"label": style._display_name(m),
                                          "color": style.COLORS[style._display_name(m)],
                                          "marker": style.MARKERS[style._display_name(m)]}
                                      for m in frame.method.unique()}})
    print(path.name, flush=True)


def sample_comparisons():
    for name, original, folder, suffix, dimension, methods in SPECS:
        frame, sources = load_source(folder, suffix, methods, dimension)
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
        coverage(axes[0], frame, "n_cals", methods, "Calibration sample size")
        style._lineplot(axes[1], frame, "n_cals", "coverage_vol_avg", methods,
                        title="Residual-space volume", xlabel="Calibration sample size",
                        ylabel="Mean", yscale="log")
        finish(fig, axes, name, original, frame, sources, ncol=3 if len(methods) > 4 else 4)


def heavy_tails():
    methods = ["Empirical_copula", "Point_CHR", "TSCP_R"]
    all_rows, sources = load_source("t", "t", methods, 10)
    for n in [30, 500]:
        frame = all_rows.loc[all_rows.n_cals.eq(n) & all_rows.df.isin([1.5, 2, 3])].copy()
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
        coverage(axes[0], frame, "df", methods, "Degrees of freedom")
        style._lineplot(axes[1], frame, "df", "coverage_vol_avg", methods,
                        title="Residual-space volume", xlabel="Degrees of freedom", ylabel="Mean", yscale="log")
        for ax in axes:
            ax.set_xticks([1.5, 2, 3])
        finish(fig, axes, f"fig_app_heavy_tail_n{n}", f"t/t_n{n}_10d.pdf", frame, sources)


def dimension_scaling():
    methods = ["Empirical_copula", "Point_CHR", "TSCP_LWC", "TSCP_R"]
    frame, sources = load_source("laplace", "laplace_30sample", methods)
    frame["runtime_ms"] = 1000 * frame.runtime_avg
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45))
    coverage(axes[0], frame, "n_dim", methods, "Output dimension")
    style._lineplot(axes[1], frame, "n_dim", "coverage_vol_avg", methods,
                    title="Residual-space volume", xlabel="Output dimension", ylabel="Mean", yscale="log")
    style._lineplot(axes[2], frame, "n_dim", "runtime_ms", methods,
                    title="Construction time", xlabel="Output dimension", ylabel="Milliseconds", yscale="log")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([2, 3, 4, 5, 10, 20, 30], [2, 3, 4, 5, 10, 20, 30])
        ax.xaxis.set_minor_locator(NullLocator())
    finish(fig, axes, "fig_app_dimension_scaling", "ours/ours_runtime.pdf", frame, sources)


def real_figures_only():
    directory = LATEX / "experiment_data/real_diagnostics"
    cs = pd.read_csv(directory / "real_coordinate_summary.csv")
    js = pd.read_csv(directory / "real_joint_summary.csv")
    bs = pd.read_csv(directory / "real_search_alpha_summary.csv")
    real.coordinate_bars(cs)
    real.marginal_coverage(cs, js)
    real.search_figures(bs)


def main():
    sample_comparisons()
    heavy_tails()
    dimension_scaling()
    real_figures_only()
    report = {"operation": "Presentation only; source CSV values are unchanged",
              "figures": RECORDS,
              "shared_style": "reviewer_update/build_experiment_update.py",
              "restyled_real_figures": real.FIGURE_NAMES}
    (LATEX / "figure_style_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

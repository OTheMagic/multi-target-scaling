"""Audit TSCP coverage and protocol metadata in reviewer-facing results.

The summary audit verifies that the relevant nominal coverage lies within one
across-repetition standard deviation of every plotted TSCP mean. It also
checks that each synthetic result came from independently redrawn train/test
samples with the experiment-specific sample counts.

Run from the repository root:

    python reviewer_update/audit_coverage.py
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reviewer_update" / "data"
TARGET = 0.9


def _summary_rows(
    path: Path,
    figure: str,
    *,
    query: str | None = None,
    x_columns: tuple[str, ...] = (),
) -> list[dict]:
    frame = pd.read_csv(path)
    if query:
        frame = frame.query(query)
    frame = frame[frame["method"].eq("TSCP_R")]
    synthetic = "real_exps" not in path.parts
    if synthetic:
        required = {"redraw_train_test", "n_train", "n_test"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Missing fresh-draw protocol metadata in {path}")
        if not frame["redraw_train_test"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Found a fixed-pool result in {path}")
    rows = []
    for _, row in frame.iterrows():
        target = 1 - float(row.get("alpha", 0.1))
        setting = ", ".join(
            f"{column}={row[column]}"
            for column in x_columns
            if column in row.index and pd.notna(row[column])
        )
        rows.append(
            {
                "figure": figure,
                "source": str(path.relative_to(ROOT)),
                "setting": setting,
                "target": target,
                "coverage_mean": float(row["test_coverage_avg"]),
                "coverage_1std": float(row["test_coverage_1std"]),
                "n_trials": int(row.get("n_trials", 200)),
                "redraw_train_test": row.get("redraw_train_test", np.nan),
                "n_train": row.get("n_train", np.nan),
                "n_test": row.get("n_test", np.nan),
            }
        )
    return rows


def build_summary_audit() -> pd.DataFrame:
    update_dir = ROOT / "reviewer_update" / "data"
    rows: list[dict] = []
    rows += _summary_rows(
        update_dir / "dependent_gaussian_summary.csv",
        "fig_body_abs_residual_overview",
        query="correlation == 0.6",
        x_columns=("n_cal", "correlation"),
    )
    rows += _summary_rows(
        update_dir / "partial_heteroskedasticity_summary.csv",
        "fig_body_partial_heteroskedasticity",
        x_columns=("n_cal",),
    )
    rows += _summary_rows(
        update_dir / "cqr_sample_size_summary.csv",
        "fig_body_cqr_comparison",
        x_columns=("n_cal",),
    )
    rows += _summary_rows(
        update_dir / "dependent_gaussian_summary.csv",
        "fig_app_dependence_sensitivity",
        query="n_cal == 100",
        x_columns=("correlation",),
    )
    rows += _summary_rows(
        update_dir / "heterogeneity_sweep_summary.csv",
        "fig_app_heterogeneity_sweep",
        x_columns=("noise_ratio",),
    )
    rows += _summary_rows(
        update_dir / "alpha_sensitivity_summary.csv",
        "fig_app_alpha_sensitivity",
        x_columns=("alpha",),
    )
    rows += _summary_rows(
        update_dir / "small_calibration_stress_summary.csv",
        "fig_app_small_calibration",
        x_columns=("n_cal",),
    )
    rows += _summary_rows(
        update_dir / "cqr_base_alpha_summary.csv",
        "fig_app_cqr_base_interval",
        x_columns=("base_interval_alpha",),
    )
    rows += _summary_rows(
        update_dir / "cqr_shift_summary.csv",
        "fig_app_cqr_shift_sensitivity",
        x_columns=("shift_constant",),
    )
    rows += _summary_rows(
        update_dir / "contamination_stress_summary.csv",
        "fig_app_contamination_stress",
        x_columns=("contamination_fraction",),
    )
    rows += _summary_rows(
        update_dir / "shape_template_standard_summary.csv",
        "fig_app_shape_template_baseline",
        x_columns=("n_cal",),
    )

    rows += _summary_rows(
        update_dir / "heavy_tail_stress_summary.csv",
        "fig_app_heavy_tail_stress",
        x_columns=("n_cal", "df"),
    )

    for dataset in ["stock", "rf2", "scm1d", "scm20d", "energy", "student"]:
        path = ROOT / "real_exps" / f"{dataset}.csv"
        frame = pd.read_csv(path)
        row = frame[frame["Methods"].eq("Standardized (Shortcut)")].iloc[0]
        rows.append(
            {
                "figure": "tab_real_data",
                "source": str(path.relative_to(ROOT)),
                "setting": f"dataset={dataset}",
                "target": TARGET,
                "coverage_mean": float(row["test_coverage_avg"]),
                "coverage_1std": float(row["test_coverage_1std"]),
                "n_trials": 200,
                "redraw_train_test": np.nan,
                "n_train": np.nan,
                "n_test": np.nan,
            }
        )

    audit = pd.DataFrame.from_records(rows)
    audit["one_sd_low"] = audit["coverage_mean"] - audit["coverage_1std"]
    audit["one_sd_high"] = audit["coverage_mean"] + audit["coverage_1std"]
    audit["target_within_one_sd"] = (
        audit["target"].ge(audit["one_sd_low"])
        & audit["target"].le(audit["one_sd_high"])
    )
    audit["naive_standard_error"] = (
        audit["coverage_1std"] / np.sqrt(audit["n_trials"])
    )
    audit["naive_ci95_low"] = (
        audit["coverage_mean"] - 1.96 * audit["naive_standard_error"]
    )
    audit["naive_ci95_high"] = (
        audit["coverage_mean"] + 1.96 * audit["naive_standard_error"]
    )
    return audit


def _draw_scores(
    rng: np.random.Generator,
    size: int,
    scenario: tuple[str, float | None],
) -> np.ndarray:
    kind, parameter = scenario
    scales = np.arange(10.0, 0.0, -1.0)
    if kind == "gaussian":
        return np.abs(rng.normal(size=(size, 10)) * scales)
    if kind == "correlated":
        rho = float(parameter)
        correlation = (1 - rho) * np.eye(10) + rho * np.ones((10, 10))
        covariance = np.diag(scales) @ correlation @ np.diag(scales)
        return np.abs(
            rng.multivariate_normal(np.zeros(10), covariance, size=size)
        )
    if kind == "partial":
        x1 = rng.normal(size=size)
        sigma = np.broadcast_to(scales, (size, 10)).copy()
        sigma[:, :5] *= np.sqrt((1 + 1.5 * x1[:, None] ** 2) / 2.5)
        return np.abs(rng.normal(size=(size, 10)) * sigma)
    if kind == "contamination":
        multiplier = np.where(
            rng.random(size) < float(parameter), 10.0, 1.0
        )
        return np.abs(
            rng.normal(size=(size, 10)) * scales * multiplier[:, None]
        )
    if kind == "heavy":
        return np.abs(rng.standard_t(float(parameter), size=(size, 10)))
    if kind == "capped":
        capped_scales = np.array([3.0, 2.0, 1.0])
        central_half_width = 0.6744897501960817 * capped_scales
        residual = np.abs(
            rng.normal(size=(size, 3)) * capped_scales
        )
        return np.maximum(residual - central_half_width, 0.0)
    raise ValueError(f"Unknown scenario: {kind}")


def run_independent_audit(n_trials: int, test_size: int) -> pd.DataFrame:
    sys.path.insert(0, str(ROOT))
    from utility.res_rescaled import check_coverage_rate, standardized_prediction

    configurations = [
        ("Gaussian", ("gaussian", None), 100),
        ("Correlated Gaussian, rho=0.6", ("correlated", 0.6), 100),
        ("Partial heteroskedasticity", ("partial", None), 100),
        ("Partial heteroskedasticity", ("partial", None), 300),
        ("Partial heteroskedasticity", ("partial", None), 500),
        ("Contamination, epsilon=0.10", ("contamination", 0.10), 100),
        ("Student t, df=1.5", ("heavy", 1.5), 500),
        ("Capped-score analogue", ("capped", None), 100),
    ]
    rows = []
    for config_index, (name, scenario, n_cal) in enumerate(configurations):
        values = {"TSCP_R": [], "TSCP_GWC": []}
        for trial in range(n_trials):
            rng = np.random.default_rng(
                2_000_000 + 10_000 * config_index + trial
            )
            calibration = _draw_scores(rng, n_cal, scenario)
            test = _draw_scores(rng, test_size, scenario)
            regions = {
                "TSCP_R": standardized_prediction(
                    calibration, alpha=0.1, method="LWC", short_cut=True
                ),
                "TSCP_GWC": standardized_prediction(
                    calibration, alpha=0.1, method="GWC", short_cut=True
                ),
            }
            for method, region in regions.items():
                values[method].append(check_coverage_rate(test, region))
        for method, coverages in values.items():
            coverages = np.asarray(coverages)
            mean = float(np.mean(coverages))
            std = float(np.std(coverages, ddof=1))
            standard_error = std / math.sqrt(n_trials)
            rows.append(
                {
                    "scenario": name,
                    "n_cal": n_cal,
                    "method": method,
                    "n_trials": n_trials,
                    "test_size_per_trial": test_size,
                    "target": TARGET,
                    "coverage_mean": mean,
                    "coverage_1std": std,
                    "standard_error": standard_error,
                    "ci95_low": mean - 1.96 * standard_error,
                    "ci95_high": mean + 1.96 * standard_error,
                    "target_within_one_sd": abs(mean - TARGET) <= std,
                }
            )
    return pd.DataFrame.from_records(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--independent-trials", type=int, default=0)
    parser.add_argument("--test-size", type=int, default=4000)
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary_audit()
    summary_path = DATA_DIR / "coverage_uncertainty_audit.csv"
    summary.to_csv(summary_path, index=False)
    failures = summary[~summary["target_within_one_sd"]]
    print(
        f"Checked {len(summary)} plotted TSCP points; "
        f"target is within one SD for {len(summary) - len(failures)}."
    )
    if not failures.empty:
        print(failures.to_string(index=False))
        raise SystemExit(1)

    if args.independent_trials > 0:
        independent = run_independent_audit(
            n_trials=args.independent_trials,
            test_size=args.test_size,
        )
        independent_path = DATA_DIR / "independent_draw_coverage_audit.csv"
        independent.to_csv(independent_path, index=False)
        print(
            independent[
                [
                    "scenario",
                    "n_cal",
                    "method",
                    "coverage_mean",
                    "coverage_1std",
                    "ci95_low",
                    "ci95_high",
                    "target_within_one_sd",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

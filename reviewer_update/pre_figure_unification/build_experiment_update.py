"""Build the reviewer-facing experiment figures and cache synthetic runs.

Run from the repository root with the bundled or project Python environment:

    python reviewer_update/build_experiment_update.py

Every synthetic repetition redraws independent training, test, and calibration
observations while holding the underlying regression coefficients fixed. The
absolute-residual experiments use 7,200 training and 800 test observations,
matching the protocol stated in the old draft. The more expensive CQR and
shape-template experiments use 2,400 training and 600 test observations.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reviewer_update"
DATA_DIR = OUT_DIR / "data"
FIGURE_DIR = OUT_DIR / "figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp" / "mpl"))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRectangle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(ROOT))

from utility.data_generator import (
    make_multitarget_regression,
    make_multitarget_regression_dependent_noise,
)
from utility.exps import (
    _stable_hash,
    run_abs_res_synthetic_experiment,
    run_cqr_synthetic_experiment,
)


ALPHA = 0.1
TARGET_COVERAGE = 1 - ALPHA
DIM = 10
METHODS = ["TSCP_R", "Point_CHR", "Unscaled", "Empirical_copula"]
SAMPLE_LIST = [30, 50, 100, 300, 500]
NEW_TRIALS = 200
SHAPE_TRIALS = 100
CQR_TRIALS = 30
ABS_N_TRAIN = 7200
ABS_N_TEST = 800
CQR_N_TRAIN = 2400
CQR_N_TEST = 600
CQR_DIM = 3
CQR_SAMPLE_LIST = [30, 50, 100, 200]
CQR_MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_samples_leaf": 5,
}

METHOD_LABELS = {
    "TSCP_R": "TSCP (Ours)",
    "TSCP_GWC": "TSCP-GWC",
    "Point_CHR": "Point CHR",
    "Point CHR": "Point CHR",
    "Unscaled": "Unscaled Max",
    "Empirical_copula": "Emp. Copula",
    "Empirical copula": "Emp. Copula",
    "CQHR": "CQHR",
    "ShapeTemplate": "Shape Template",
}

COLORS = {
    "TSCP (Ours)": "#E15759",
    "TSCP-GWC": "#F28E2B",
    "Point CHR": "#4E79A7",
    "Unscaled Max": "#9C755F",
    "Emp. Copula": "#8C8C8C",
    "CQHR": "#7B61A8",
    "Shape Template": "#59A14F",
}

MARKERS = {
    "TSCP (Ours)": "D",
    "TSCP-GWC": "v",
    "Point CHR": "o",
    "Unscaled Max": "<",
    "Emp. Copula": "P",
    "CQHR": "s",
    "Shape Template": "^",
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
    }
)


def _linear_signal(
    rng: np.random.Generator,
    n_samples: int,
    n_features: int,
    n_targets: int,
    n_informative: int,
    coef,
):
    """Create the same linear signal contract used by the repository generators."""
    X = rng.standard_normal(size=(n_samples, n_features))
    if coef is None:
        coef_list = []
        for _ in range(n_targets):
            coef_j = np.zeros(n_features)
            idx = rng.choice(n_features, size=n_informative, replace=False)
            coef_j[idx] = rng.uniform(-10, 10, size=n_informative)
            coef_list.append(coef_j)
    else:
        coef_list = [np.asarray(value, dtype=float) for value in coef]
    signal = np.column_stack([X @ coef_j for coef_j in coef_list])
    return X, signal, coef_list


def make_partial_heteroskedastic_regression(
    n_samples=100,
    n_features=10,
    n_targets=10,
    n_informative=10,
    noise_type="Gaussian",
    noise_list=None,
    coef=None,
    random_state=42,
    heteroskedastic_fraction=0.5,
    heteroskedastic_strength=1.5,
):
    """Gaussian noise with input-dependent variance in only some coordinates.

    For the first ``heteroskedastic_fraction`` coordinates, the conditional
    variance depends on X_1. The normalization keeps each coordinate's
    unconditional second moment equal to the corresponding entry of
    ``noise_list`` squared, isolating conditional heteroskedasticity from a
    change in marginal scale.
    """
    if noise_type.lower() != "gaussian":
        raise ValueError("The partial-heteroskedastic generator is Gaussian only.")
    if noise_list is None:
        noise_list = np.arange(n_targets, 0, -1, dtype=float)
    scales = np.asarray(noise_list, dtype=float)
    if len(scales) != n_targets:
        raise ValueError("noise_list must match n_targets.")

    rng = np.random.default_rng(random_state)
    X, signal, coef_list = _linear_signal(
        rng, n_samples, n_features, n_targets, n_informative, coef
    )
    noise = rng.standard_normal(size=(n_samples, n_targets)) * scales
    n_hetero = int(round(n_targets * heteroskedastic_fraction))
    factor = np.sqrt(
        (1 + heteroskedastic_strength * X[:, [0]] ** 2)
        / (1 + heteroskedastic_strength)
    )
    noise[:, :n_hetero] *= factor
    y = signal + noise
    if coef is None:
        return X, y, coef_list
    return X, y


def make_contaminated_regression(
    n_samples=100,
    n_features=10,
    n_targets=10,
    n_informative=10,
    noise_type="Gaussian",
    noise_list=None,
    coef=None,
    random_state=42,
    contamination_fraction=0.0,
    contamination_multiplier=10.0,
):
    """Gaussian regression with exchangeable row-wise scale contamination."""
    if noise_type.lower() != "gaussian":
        raise ValueError("The contamination generator is Gaussian only.")
    if noise_list is None:
        noise_list = np.arange(n_targets, 0, -1, dtype=float)
    scales = np.asarray(noise_list, dtype=float)
    rng = np.random.default_rng(random_state)
    X, signal, coef_list = _linear_signal(
        rng, n_samples, n_features, n_targets, n_informative, coef
    )
    row_multiplier = np.where(
        rng.random(n_samples) < contamination_fraction,
        contamination_multiplier,
        1.0,
    )
    noise = (
        rng.standard_normal(size=(n_samples, n_targets))
        * scales
        * row_multiplier[:, None]
    )
    y = signal + noise
    if coef is None:
        return X, y, coef_list
    return X, y


def _result_tables(result):
    return {
        "trial": result.trial_results,
        "summary": result.summary_results,
        "coordinate_trial": result.coordinate_trial_results,
        "coordinate_summary": result.coordinate_summary_results,
    }


def _save_tables(prefix: str, tables: dict[str, pd.DataFrame]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(DATA_DIR / f"{prefix}_{name}.csv", index=False)


def _has_protocol(frame: pd.DataFrame, n_train: int, n_test: int) -> bool:
    required = {"redraw_train_test", "n_train", "n_test"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    redraw = frame["redraw_train_test"].astype(str).str.lower()
    return (
        redraw.eq("true").all()
        and frame["n_train"].astype(int).eq(n_train).all()
        and frame["n_test"].astype(int).eq(n_test).all()
    )


def _load_tables(
    prefix: str,
    *,
    n_train: int,
    n_test: int,
    trials: int,
) -> dict[str, pd.DataFrame] | None:
    names = ["trial", "summary", "coordinate_trial", "coordinate_summary"]
    paths = {name: DATA_DIR / f"{prefix}_{name}.csv" for name in names}
    if not all(path.exists() for path in paths.values()):
        return None
    try:
        tables = {name: pd.read_csv(path) for name, path in paths.items()}
    except (OSError, pd.errors.ParserError):
        return None
    if not all(_has_protocol(frame, n_train, n_test) for frame in tables.values()):
        return None
    if tables["trial"]["trial"].nunique() != trials:
        return None
    return tables


def _combine_tables(items: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.concat([item[name] for item in items], ignore_index=True)
        for name in items[0]
    }


def _with_metadata(
    tables: dict[str, pd.DataFrame], **metadata
) -> dict[str, pd.DataFrame]:
    for frame in tables.values():
        for name, value in metadata.items():
            frame[name] = value
    return tables


def _abs_experiment(**kwargs):
    defaults = {
        "dim_list": [DIM],
        "alpha_list": [ALPHA],
        "noise_type": "Gaussian",
        "trials": NEW_TRIALS,
        "methods": METHODS,
        "n_train_pool": ABS_N_TRAIN + ABS_N_TEST,
        "n_features": 10,
        "n_informative": 10,
        "redraw_train_test": True,
        "n_train": ABS_N_TRAIN,
        "n_test": ABS_N_TEST,
        "output_dir": None,
    }
    defaults.update(kwargs)
    return run_abs_res_synthetic_experiment(**defaults)


def _cqr_experiment(**kwargs):
    defaults = {
        "dim_list": [CQR_DIM],
        "alpha_list": [ALPHA],
        "noise_type": "Gaussian",
        "noises_list": [np.arange(CQR_DIM, 0, -1, dtype=float)],
        "trials": CQR_TRIALS,
        "methods": ["TSCP_R", "CQHR", "Unscaled", "Empirical_copula"],
        "score_transform": "capped",
        "base_interval_alpha": 0.5,
        "quantile_model_params": CQR_MODEL_PARAMS,
        "quantile_n_jobs": 3,
        "n_train_pool": CQR_N_TRAIN + CQR_N_TEST,
        "n_features": 10,
        "n_informative": 10,
        "redraw_train_test": True,
        "n_train": CQR_N_TRAIN,
        "n_test": CQR_N_TEST,
        "output_dir": None,
    }
    defaults.update(kwargs)
    return run_cqr_synthetic_experiment(**defaults)


def ensure_dependent_gaussian() -> dict[str, pd.DataFrame]:
    prefix = "dependent_gaussian"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running dependent Gaussian experiment...", flush=True)
    items = []
    for correlation in [0.0, 0.3, 0.6, 0.9]:
        result = _abs_experiment(
            sample_list=SAMPLE_LIST,
            noises_list=[np.arange(DIM, 0, -1, dtype=float)],
            data_generator=make_multitarget_regression_dependent_noise,
            generator_kwargs={
                "correlation": correlation,
                "correlation_structure": "equicorrelated",
            },
            experiment_name=f"{prefix}_{correlation}",
        )
        items.append(_result_tables(result))
    tables = _combine_tables(items)
    _save_tables(prefix, tables)
    return tables


def ensure_partial_heteroskedasticity() -> dict[str, pd.DataFrame]:
    prefix = "partial_heteroskedasticity"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running partial heteroskedasticity experiment...", flush=True)
    result = _abs_experiment(
        sample_list=SAMPLE_LIST,
        noises_list=[np.arange(DIM, 0, -1, dtype=float)],
        data_generator=make_partial_heteroskedastic_regression,
        generator_kwargs={
            "heteroskedastic_fraction": 0.5,
            "heteroskedastic_strength": 1.5,
        },
        experiment_name=prefix,
    )
    tables = _result_tables(result)
    _save_tables(prefix, tables)
    return tables


def ensure_contamination() -> dict[str, pd.DataFrame]:
    prefix = "contamination_stress"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None and "contamination_fraction" in cached["summary"].columns:
        return cached
    print("Running contamination stress experiment...", flush=True)
    all_tables = []
    for fraction in [0.0, 0.01, 0.05, 0.10]:
        result = _abs_experiment(
            sample_list=[100],
            noises_list=[np.arange(DIM, 0, -1, dtype=float)],
            data_generator=make_contaminated_regression,
            generator_kwargs={
                "contamination_fraction": fraction,
                "contamination_multiplier": 10.0,
            },
            experiment_name=f"{prefix}_{fraction}",
        )
        all_tables.append(
            _with_metadata(
                _result_tables(result),
                contamination_fraction=fraction,
                contamination_multiplier=10.0,
            )
        )
    tables = _combine_tables(all_tables)
    _save_tables(prefix, tables)
    return tables


def ensure_heterogeneity() -> dict[str, pd.DataFrame]:
    prefix = "heterogeneity_sweep"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running heterogeneity sweep...", flush=True)
    items = []
    for ratio in [1, 2, 5, 10, 20]:
        result = _abs_experiment(
            sample_list=[100],
            noises_list=[np.geomspace(float(ratio), 1.0, DIM)],
            experiment_name=f"{prefix}_{ratio}",
        )
        items.append(_with_metadata(_result_tables(result), noise_ratio=ratio))
    tables = _combine_tables(items)
    _save_tables(prefix, tables)
    return tables


def ensure_alpha_sensitivity() -> dict[str, pd.DataFrame]:
    prefix = "alpha_sensitivity"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running alpha sensitivity experiment...", flush=True)
    result = _abs_experiment(
        sample_list=[100],
        alpha_list=[0.05, 0.1, 0.2],
        noises_list=[np.arange(DIM, 0, -1, dtype=float)],
        data_generator=make_multitarget_regression_dependent_noise,
        generator_kwargs={
            "correlation": 0.6,
            "correlation_structure": "equicorrelated",
        },
        experiment_name=prefix,
    )
    tables = _result_tables(result)
    _save_tables(prefix, tables)
    return tables


def ensure_small_calibration() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prefix = "small_calibration_stress"
    rate_path = DATA_DIR / f"{prefix}_infinite_volume_rate.csv"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None and rate_path.exists():
        return cached, pd.read_csv(rate_path)
    print("Running small-calibration stress experiment...", flush=True)
    result = _abs_experiment(
        sample_list=[10, 20, 30, 50, 100],
        noises_list=[np.arange(DIM, 0, -1, dtype=float)],
        experiment_name=prefix,
    )
    tables = _result_tables(result)
    rate = (
        tables["trial"]
        .assign(is_infinite=lambda frame: np.isinf(frame["coverage_volume"]))
        .groupby(
            [
                "alpha",
                "n_dim",
                "n_cal",
                "method",
                "noise_type",
                "redraw_train_test",
                "n_train",
                "n_test",
            ],
            as_index=False,
        )
        .agg(infinite_volume_rate=("is_infinite", "mean"))
    )
    _save_tables(prefix, tables)
    rate.to_csv(rate_path, index=False)
    return tables, rate


def ensure_heavy_tails() -> dict[str, pd.DataFrame]:
    prefix = "heavy_tail_stress"
    cached = _load_tables(
        prefix,
        n_train=ABS_N_TRAIN,
        n_test=ABS_N_TEST,
        trials=NEW_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running homogeneous Student-t stress experiment...", flush=True)
    items = []
    for degrees_of_freedom in [1.5, 2.0, 3.0]:
        result = _abs_experiment(
            sample_list=[30, 500],
            noise_type="t",
            noises_list=[np.ones(DIM)],
            methods=["TSCP_R", "Point_CHR", "Empirical_copula"],
            generator_kwargs={"df": degrees_of_freedom},
            experiment_name=f"{prefix}_{degrees_of_freedom}",
        )
        items.append(_result_tables(result))
    tables = _combine_tables(items)
    _save_tables(prefix, tables)
    return tables


def ensure_cqr_sample_size() -> dict[str, pd.DataFrame]:
    prefix = "cqr_sample_size"
    cached = _load_tables(
        prefix,
        n_train=CQR_N_TRAIN,
        n_test=CQR_N_TEST,
        trials=CQR_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running CQR sample-size experiment...", flush=True)
    result = _cqr_experiment(
        sample_list=CQR_SAMPLE_LIST,
        experiment_name=prefix,
    )
    tables = _result_tables(result)
    _save_tables(prefix, tables)
    return tables


def ensure_cqr_base_alpha() -> dict[str, pd.DataFrame]:
    prefix = "cqr_base_alpha"
    cached = _load_tables(
        prefix,
        n_train=CQR_N_TRAIN,
        n_test=CQR_N_TEST,
        trials=CQR_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running CQR base-interval sweep...", flush=True)
    items = []
    for base_alpha in [0.9, 0.7, 0.5, 0.3]:
        result = _cqr_experiment(
            sample_list=[100],
            base_interval_alpha=base_alpha,
            experiment_name=f"{prefix}_{base_alpha}",
        )
        items.append(_result_tables(result))
    tables = _combine_tables(items)
    _save_tables(prefix, tables)
    return tables


def ensure_cqr_shift() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    prefix = "cqr_shift"
    diagnostic_path = DATA_DIR / f"{prefix}_tscp_r_vs_gwc_diagnostic.csv"
    cached = _load_tables(
        prefix,
        n_train=CQR_N_TRAIN,
        n_test=CQR_N_TEST,
        trials=CQR_TRIALS,
    )
    if cached is not None and diagnostic_path.exists():
        return cached, pd.read_csv(diagnostic_path)
    print("Running CQR shift-sensitivity experiment...", flush=True)
    items = []
    for shift_constant in [100.0, 300.0, 1000.0]:
        result = _cqr_experiment(
            sample_list=[100],
            methods=[
                "TSCP_R",
                "TSCP_GWC",
                "CQHR",
                "Unscaled",
                "Empirical_copula",
            ],
            score_transform="shifted",
            shift_constant=shift_constant,
            experiment_name=f"{prefix}_{shift_constant:g}",
        )
        items.append(_result_tables(result))
    tables = _combine_tables(items)
    comparison = (
        tables["coordinate_summary"]
        .query("method in ['TSCP_R', 'TSCP_GWC']")
        .pivot_table(
            index=["shift_constant", "coordinate"],
            columns="method",
            values="coordinate_length_avg",
        )
        .reset_index()
    )
    comparison["absolute_difference"] = (
        comparison["TSCP_R"] - comparison["TSCP_GWC"]
    ).abs()
    diagnostic = (
        comparison.groupby("shift_constant", as_index=False)
        .agg(max_abs_coordinate_length_diff=("absolute_difference", "max"))
    )
    _save_tables(prefix, tables)
    diagnostic.to_csv(diagnostic_path, index=False)
    return tables, diagnostic


def ensure_shape_template_standard() -> dict[str, pd.DataFrame]:
    prefix = "shape_template_standard"
    cached = _load_tables(
        prefix,
        n_train=CQR_N_TRAIN,
        n_test=CQR_N_TEST,
        trials=SHAPE_TRIALS,
    )
    if cached is not None:
        return cached
    print("Running low-dimensional standard baselines...", flush=True)
    result = run_abs_res_synthetic_experiment(
        dim_list=[2],
        sample_list=[30, 50, 100, 200],
        alpha_list=[ALPHA],
        noise_type="Gaussian",
        noises_list=[np.array([2.0, 1.0])],
        trials=SHAPE_TRIALS,
        methods=["TSCP_R", "Point_CHR", "Unscaled"],
        n_train_pool=CQR_N_TRAIN + CQR_N_TEST,
        n_features=10,
        n_informative=10,
        redraw_train_test=True,
        n_train=CQR_N_TRAIN,
        n_test=CQR_N_TEST,
        output_dir=None,
        experiment_name=prefix,
    )
    tables = _result_tables(result)
    _save_tables(prefix, tables)
    return tables


def ensure_shape_template_baseline() -> tuple[pd.DataFrame, pd.DataFrame]:
    trial_path = DATA_DIR / "shape_template_baseline_trial.csv"
    summary_path = DATA_DIR / "shape_template_baseline_summary.csv"
    if trial_path.exists() and summary_path.exists():
        trial_cached = pd.read_csv(trial_path)
        summary_cached = pd.read_csv(summary_path)
        if (
            _has_protocol(trial_cached, CQR_N_TRAIN, CQR_N_TEST)
            and _has_protocol(summary_cached, CQR_N_TRAIN, CQR_N_TEST)
            and trial_cached["trial"].nunique() == SHAPE_TRIALS
        ):
            return trial_cached, summary_cached

    print("Running shape-template baseline...", flush=True)

    # The released package imports its optional Gurobi ellipse dependency even
    # when only a hyperrectangle is requested. The hyperrectangle path does not
    # use it, so a placeholder module keeps that unrelated import optional.
    sys.modules.setdefault("gurobipy", types.ModuleType("gurobipy"))
    from conformal_region_designer import ConformalRegion
    from conformal_region_designer.core import ShapeTemplate
    from conformal_region_designer.density_estimation import KDE

    class ScoreConsistentHyperrectangle(ShapeTemplate):
        """Axis-aligned template with score-consistent conformal inflation."""

        def fit_shape(self, points):
            self.min = np.min(points, axis=0)
            self.max = np.max(points, axis=0)
            widths = self.max - self.min
            max_width = np.max(widths)
            if max_width <= 1e-12:
                self.aspect_ratio = np.ones_like(widths)
            else:
                # Very small clusters can be degenerate along one coordinate.
                # A modest floor keeps conformal inflation finite and stable.
                self.aspect_ratio = np.maximum(widths / max_width, 0.05)

        def score_points(self, points):
            outside = np.maximum(points - self.max, self.min - points)
            return np.max(outside * self.aspect_ratio, axis=1)

        def adjust_shape(self, score_margin):
            inflation = score_margin / self.aspect_ratio
            self.min -= inflation
            self.max += inflation

        def volume(self):
            return float(np.prod(self.max - self.min))

    dim = 2
    noise_levels = np.array([2.0, 1.0])
    _, _, coef_true = make_multitarget_regression(
        n_samples=CQR_N_TRAIN + CQR_N_TEST,
        n_features=10,
        n_informative=10,
        n_targets=dim,
        noise_type="Gaussian",
        noise_list=noise_levels,
        random_state=_stable_hash(dim),
    )
    records = []
    for sample in [30, 50, 100, 200]:
        for trial in range(SHAPE_TRIALS):
            X_trial, y_trial = make_multitarget_regression(
                n_samples=CQR_N_TRAIN + CQR_N_TEST,
                n_features=10,
                n_informative=10,
                n_targets=dim,
                noise_type="Gaussian",
                noise_list=noise_levels,
                random_state=_stable_hash(dim, sample, trial, "train_test"),
                coef=coef_true,
            )
            X_train, X_test = X_trial[:CQR_N_TRAIN], X_trial[CQR_N_TRAIN:]
            y_train, y_test = y_trial[:CQR_N_TRAIN], y_trial[CQR_N_TRAIN:]
            model = LinearRegression().fit(X_train, y_train)
            signed_test = np.asarray(y_test - model.predict(X_test), dtype=float)
            X_cal, y_cal = make_multitarget_regression(
                n_samples=sample,
                n_features=10,
                n_informative=10,
                n_targets=dim,
                noise_type="Gaussian",
                noise_list=noise_levels,
                random_state=_stable_hash(dim, sample, trial, "calibration"),
                coef=coef_true,
            )
            signed_cal = np.asarray(y_cal - model.predict(X_cal), dtype=float)
            first_size = sample // 2
            region = ConformalRegion(
                de=KDE(grid_size=20),
                cl="meanshift",
                st=ScoreConsistentHyperrectangle,
                delta=TARGET_COVERAGE,
            )
            start = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                region.fit(signed_cal[:first_size])
                region.conformalize(signed_cal[first_size:])
            runtime = time.perf_counter() - start
            records.append(
                {
                    "alpha": ALPHA,
                    "n_dim": dim,
                    "n_cal": sample,
                    "trial": trial,
                    "method": "ShapeTemplate",
                    "noise_type": "Gaussian",
                    "score_type": "absolute_residual",
                    "redraw_train_test": True,
                    "n_train": CQR_N_TRAIN,
                    "n_test": CQR_N_TEST,
                    "test_coverage": np.mean(region.calculate_scores(signed_test) <= 0),
                    "coverage_volume": region.volume() / (2**dim),
                    "runtime": runtime,
                    "n_shapes": len(region.shapes),
                    "grid_size": 20,
                    "template_calibration_size": sample - first_size,
                }
            )
    trial_df = pd.DataFrame.from_records(records)
    summary_df = (
        trial_df.groupby(
            [
                "alpha",
                "n_dim",
                "n_cal",
                "method",
                "noise_type",
                "score_type",
                "redraw_train_test",
                "n_train",
                "n_test",
            ],
            as_index=False,
        )
        .agg(
            n_trials=("trial", "nunique"),
            test_coverage_avg=("test_coverage", "mean"),
            test_coverage_1std=("test_coverage", "std"),
            coverage_vol_avg=("coverage_volume", "mean"),
            coverage_vol_1std=("coverage_volume", "std"),
            runtime_avg=("runtime", "mean"),
            n_shapes_avg=("n_shapes", "mean"),
        )
    )
    trial_df.to_csv(trial_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return trial_df, summary_df


def _facet_title(ax, title: str) -> None:
    ax.add_patch(
        MplRectangle(
            (0, 1.0),
            1,
            0.11,
            transform=ax.transAxes,
            facecolor="#E5E5E5",
            edgecolor="none",
            clip_on=False,
            zorder=-1,
        )
    )
    ax.text(0.5, 1.055, title, ha="center", va="center", transform=ax.transAxes)


def _style_axis(ax, title: str) -> None:
    ax.set_facecolor("white")
    ax.grid(True, color="#D4D4D4", linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#B8B8B8")
    _facet_title(ax, title)


def _display_name(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _lineplot(
    ax,
    data: pd.DataFrame,
    x: str,
    y: str,
    methods: list[str],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    yscale: str | None = None,
    target: float | None = None,
):
    for method in methods:
        subset = data[data["method"].eq(method)].sort_values(x)
        if subset.empty:
            continue
        label = _display_name(method)
        ax.plot(
            subset[x],
            subset[y],
            color=COLORS[label],
            marker=MARKERS[label],
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=label,
        )
    if target is not None:
        ax.axhline(target, color="#444444", linestyle="--", linewidth=0.9, zorder=0)
    if yscale:
        ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _style_axis(ax, title)


def _set_coverage_axis(ax) -> None:
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))


def _shared_legend(fig, axes, ncol=4, y=-0.12) -> None:
    handles = []
    labels = []
    for ax in np.ravel(axes):
        current_handles, current_labels = ax.get_legend_handles_labels()
        for handle, label in zip(current_handles, current_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(ncol, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.2,
        handletextpad=0.5,
    )


def _grouped_barplot(
    ax,
    data: pd.DataFrame,
    *,
    methods: list[str],
    value_column: str,
    title: str,
    ylabel: str,
) -> None:
    coordinates = sorted(data["coordinate"].unique())
    positions = np.arange(len(coordinates), dtype=float)
    width = 0.82 / len(methods)
    for method_index, method in enumerate(methods):
        subset = data[data["method"].eq(method)].set_index("coordinate")
        values = subset.loc[coordinates, value_column].to_numpy()
        label = _display_name(method)
        offset = (method_index - (len(methods) - 1) / 2) * width
        ax.bar(
            positions + offset,
            values,
            width=width,
            color=COLORS[label],
            edgecolor="white",
            linewidth=0.55,
            label=label,
        )
    ax.set_xticks(positions, coordinates)
    ax.set_xlabel("Outcome coordinate")
    ax.set_ylabel(ylabel)
    _style_axis(ax, title)
    ax.grid(axis="x", visible=False)


def _save(fig, filename: str, bottom=0.13) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(bottom=bottom, wspace=0.42, hspace=0.48)
    path = FIGURE_DIR / filename
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata={"Creator": "reviewer_update/build_experiment_update.py"},
    )
    plt.close(fig)
    return path


def figure_body_abs_residual_overview(tables) -> Path:
    summary = tables["summary"].copy()
    summary = summary[summary["correlation"].eq(0.6)].copy()
    summary["runtime_ms"] = 1000 * summary["runtime_avg"]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    _lineplot(
        axes[0], summary, "n_cal", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Calibration sample size", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], summary, "n_cal", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Calibration sample size", ylabel="Mean", yscale="log",
    )
    _lineplot(
        axes[2], summary, "n_cal", "runtime_ms", methods,
        title="Construction time", xlabel="Calibration sample size", ylabel="Milliseconds", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_body_abs_residual_overview.pdf", bottom=0.21)


def figure_body_abs_coordinate_bars(tables) -> Path:
    data = tables["coordinate_summary"].query(
        "correlation == 0.6 and n_cal == 100"
    ).copy()
    data["full_length"] = 2 * data["coordinate_length_avg"]
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    _grouped_barplot(
        ax,
        data,
        methods=methods,
        value_column="full_length",
        title="Coordinate-wise interval length",
        ylabel="Mean full length",
    )
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    return _save(fig, "fig_body_abs_coordinate_bars.pdf", bottom=0.25)


def figure_body_partial_heteroskedasticity(tables) -> Path:
    summary = tables["summary"].copy()
    summary["runtime_ms"] = 1000 * summary["runtime_avg"]
    coord = tables["coordinate_summary"].query("n_cal == 100").copy()
    coord["full_length"] = 2 * coord["coordinate_length_avg"]
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    _lineplot(
        axes[0], summary, "n_cal", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Calibration sample size", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], summary, "n_cal", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Calibration sample size", ylabel="Mean", yscale="log",
    )
    _lineplot(
        axes[2], coord, "coordinate", "full_length", methods,
        title="Coordinate-wise length", xlabel="Outcome coordinate", ylabel="Mean full length",
    )
    axes[2].axvspan(0.5, 5.5, color="#EAEAEA", zorder=-2)
    axes[2].text(
        3, 0.05, "heteroskedastic", ha="center", va="bottom",
        fontsize=6.5, transform=axes[2].get_xaxis_transform(),
    )
    axes[2].set_xticks(range(1, 11))
    _shared_legend(fig, axes)
    return _save(fig, "fig_body_partial_heteroskedasticity.pdf", bottom=0.21)


def figure_body_cqr_comparison(tables) -> Path:
    data = tables["summary"].copy()
    methods = ["Empirical_copula", "Unscaled", "CQHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45))
    _lineplot(
        axes[0], data, "n_cal", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Calibration sample size", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "n_cal", "coverage_vol_avg", methods,
        title="Outcome-space volume", xlabel="Calibration sample size", ylabel="Mean", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_body_cqr_comparison.pdf", bottom=0.22)


def figure_body_cqr_coordinate_bars(tables) -> Path:
    data = tables["coordinate_summary"].query("n_cal == 100").copy()
    methods = ["Empirical_copula", "Unscaled", "CQHR", "TSCP_R"]
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    _grouped_barplot(
        ax,
        data,
        methods=methods,
        value_column="coordinate_length_avg",
        title="Coordinate-wise outcome-space length",
        ylabel="Mean full length",
    )
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        columnspacing=1.2,
        handletextpad=0.5,
    )
    return _save(fig, "fig_body_cqr_coordinate_bars.pdf", bottom=0.25)


def figure_body_real_runtime() -> Path:
    datasets = ["stock", "rf2", "scm1d", "scm20d", "energy", "student"]
    method_map = {
        "Standardized (Shortcut)": "TSCP_R",
        "Point CHR": "Point_CHR",
        "Empirical copula": "Empirical_copula",
        "Unscaled": "Unscaled",
    }
    frames = []
    for order, dataset in enumerate(datasets):
        frame = pd.read_csv(ROOT / "real_exps" / f"{dataset}.csv")
        frame = frame[frame["Methods"].isin(method_map)].copy()
        frame["method"] = frame["Methods"].map(method_map)
        frame["dataset"] = dataset
        frame["dataset_order"] = order
        frame["runtime_ms"] = 1000 * frame["runtime_avg"]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    _lineplot(
        ax, data, "dataset_order", "runtime_ms", methods,
        title="Wall-clock construction time on real data", xlabel="Dataset", ylabel="Milliseconds", yscale="log",
    )
    ax.set_xticks(range(len(datasets)), datasets)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.24))
    return _save(fig, "fig_body_real_runtime.pdf", bottom=0.25)


def figure_app_dependence(tables) -> Path:
    data = tables["summary"].query("n_cal == 100").copy()
    data["runtime_ms"] = 1000 * data["runtime_avg"]
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    _lineplot(
        axes[0], data, "correlation", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Equicorrelation", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "correlation", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Equicorrelation", ylabel="Mean", yscale="log",
    )
    _lineplot(
        axes[2], data, "correlation", "runtime_ms", methods,
        title="Construction time", xlabel="Equicorrelation", ylabel="Milliseconds", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_dependence_sensitivity.pdf", bottom=0.21)


def figure_app_heterogeneity(tables) -> Path:
    data = tables["summary"].copy()
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    _lineplot(
        axes[0], data, "noise_ratio", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Largest/smallest noise scale", ylabel="Mean", target=TARGET_COVERAGE,
    )
    axes[0].set_xscale("log")
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "noise_ratio", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Largest/smallest noise scale", ylabel="Mean", yscale="log",
    )
    axes[1].set_xscale("log")
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_heterogeneity_sweep.pdf", bottom=0.22)


def figure_app_alpha(tables) -> Path:
    data = tables["summary"].copy()
    data["nominal_coverage"] = 1 - data["alpha"]
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    _lineplot(
        axes[0], data, "nominal_coverage", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Nominal coverage", ylabel="Mean",
    )
    limits = [0.77, 0.97]
    axes[0].plot(limits, limits, color="#444444", linestyle="--", linewidth=0.9)
    axes[0].set_xlim(limits)
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "nominal_coverage", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Nominal coverage", ylabel="Mean", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_alpha_sensitivity.pdf", bottom=0.22)


def figure_app_small_calibration(tables, inf) -> Path:
    data = tables["summary"].copy()
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    _lineplot(
        axes[0], data, "n_cal", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Calibration sample size", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    for method in methods:
        clipped = data[
            data["method"].eq(method) & data["test_coverage_avg"].lt(0.6)
        ]
        if clipped.empty:
            continue
        label = _display_name(method)
        axes[0].scatter(
            clipped["n_cal"],
            np.full(len(clipped), 0.605),
            color=COLORS[label],
            marker="v",
            edgecolor="white",
            linewidth=0.45,
            zorder=5,
        )
    finite = data.replace([np.inf, -np.inf], np.nan)
    _lineplot(
        axes[1], finite, "n_cal", "coverage_vol_avg", methods,
        title="Finite residual-space volume", xlabel="Calibration sample size", ylabel="Mean", yscale="log",
    )
    _lineplot(
        axes[2], inf, "n_cal", "infinite_volume_rate", methods,
        title="Infinite-volume frequency", xlabel="Calibration sample size", ylabel="Fraction",
    )
    axes[2].set_ylim(-0.03, 1.03)
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_small_calibration.pdf", bottom=0.21)


def figure_app_cqr_base_alpha(tables) -> Path:
    data = tables["summary"].copy()
    methods = ["Empirical_copula", "Unscaled", "CQHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    _lineplot(
        axes[0], data, "base_interval_alpha", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Base-interval miscoverage", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "base_interval_alpha", "coverage_vol_avg", methods,
        title="Outcome-space volume", xlabel="Base-interval miscoverage", ylabel="Mean", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_cqr_base_interval.pdf", bottom=0.22)


def figure_app_cqr_shift(tables, diag) -> Path:
    data = tables["summary"].copy()
    shift_values = sorted(
        data.loc[data["method"].eq("TSCP_R"), "shift_constant"].unique()
    )
    invariant_rows = []
    for method in ["Unscaled", "CQHR"]:
        row = data[data["method"].eq(method)].iloc[[0]]
        for shift_constant in shift_values:
            repeated = row.copy()
            repeated["shift_constant"] = shift_constant
            invariant_rows.append(repeated)
    data = pd.concat(
        [
            data[~data["method"].isin(["Unscaled", "CQHR"])],
            *invariant_rows,
        ],
        ignore_index=True,
    )
    methods = ["Unscaled", "CQHR", "TSCP_GWC", "TSCP_R"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    _lineplot(
        axes[0], data, "shift_constant", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Shift constant C", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "shift_constant", "coverage_vol_avg", methods,
        title="Outcome-space volume", xlabel="Shift constant C", ylabel="Mean", yscale="log",
    )
    axes[2].plot(
        diag["shift_constant"],
        diag["max_abs_coordinate_length_diff"],
        color=COLORS["TSCP (Ours)"],
        marker=MARKERS["TSCP (Ours)"],
    )
    axes[2].set_xlabel("Shift constant C")
    axes[2].set_ylabel("Maximum absolute difference")
    axes[2].set_ylim(-0.02, 0.1)
    axes[2].set_ylabel("Max. absolute difference")
    _style_axis(axes[2], "TSCP/TSCP-GWC length gap")
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_cqr_shift_sensitivity.pdf", bottom=0.21)


def figure_app_contamination(tables) -> Path:
    trial = tables["trial"].copy()
    summary = (
        trial.groupby(["contamination_fraction", "method"], as_index=False)
        .agg(
            test_coverage_avg=("test_coverage", "mean"),
            test_coverage_1std=("test_coverage", "std"),
        )
    )
    volume = (
        trial.groupby(["contamination_fraction", "method"], as_index=False)
        .agg(coverage_volume_median=("coverage_volume", "median"))
    )
    methods = ["Empirical_copula", "Unscaled", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    _lineplot(
        axes[0], summary, "contamination_fraction", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Contamination fraction", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], volume, "contamination_fraction", "coverage_volume_median", methods,
        title="Residual-space volume", xlabel="Contamination fraction", ylabel="Median", yscale="log",
    )
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_contamination_stress.pdf", bottom=0.22)


def figure_app_shape_template(standard, shape_summary) -> Path:
    data = pd.concat([standard["summary"], shape_summary], ignore_index=True, sort=False)
    data["runtime_ms"] = 1000 * data["runtime_avg"]
    methods = ["Unscaled", "Point_CHR", "ShapeTemplate", "TSCP_R"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    _lineplot(
        axes[0], data, "n_cal", "test_coverage_avg", methods,
        title="Joint coverage", xlabel="Calibration sample size", ylabel="Mean", target=TARGET_COVERAGE,
    )
    _set_coverage_axis(axes[0])
    _lineplot(
        axes[1], data, "n_cal", "coverage_vol_avg", methods,
        title="Residual-space volume", xlabel="Calibration sample size", ylabel="Mean", yscale="log",
    )
    _lineplot(
        axes[2], data, "n_cal", "runtime_ms", methods,
        title="Construction time", xlabel="Calibration sample size", ylabel="Milliseconds", yscale="log",
    )
    axes[1].set_ylabel("")
    _shared_legend(fig, axes)
    return _save(fig, "fig_app_shape_template_baseline.pdf", bottom=0.21)


def figure_app_heavy_tails(tables) -> Path:
    data = tables["summary"].copy()
    methods = ["Empirical_copula", "Point_CHR", "TSCP_R"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.7), sharex=True)
    for row, n_cal in enumerate([30, 500]):
        subset = data[data["n_cal"].eq(n_cal)]
        _lineplot(
            axes[row, 0], subset, "df", "test_coverage_avg", methods,
            title=f"Coverage, n={n_cal}", xlabel="" if row == 0 else "Degrees of freedom", ylabel="Mean", target=TARGET_COVERAGE,
        )
        _set_coverage_axis(axes[row, 0])
        _lineplot(
            axes[row, 1], subset, "df", "coverage_vol_avg", methods,
            title=f"Volume, n={n_cal}", xlabel="" if row == 0 else "Degrees of freedom", ylabel="Mean", yscale="log",
        )
    _shared_legend(fig, axes, ncol=3, y=-0.02)
    return _save(fig, "fig_app_heavy_tail_stress.pdf")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    dependent = ensure_dependent_gaussian()
    partial = ensure_partial_heteroskedasticity()
    heterogeneity = ensure_heterogeneity()
    alpha_sensitivity = ensure_alpha_sensitivity()
    small_calibration, infinite_volume_rate = ensure_small_calibration()
    contamination = ensure_contamination()
    heavy_tails = ensure_heavy_tails()
    cqr_sample_size = ensure_cqr_sample_size()
    cqr_base_alpha = ensure_cqr_base_alpha()
    cqr_shift, cqr_shift_diagnostic = ensure_cqr_shift()
    shape_standard = ensure_shape_template_standard()
    _, shape_summary = ensure_shape_template_baseline()

    paths = [
        figure_body_abs_residual_overview(dependent),
        figure_body_abs_coordinate_bars(dependent),
        figure_body_partial_heteroskedasticity(partial),
        figure_body_cqr_comparison(cqr_sample_size),
        figure_body_cqr_coordinate_bars(cqr_sample_size),
        figure_body_real_runtime(),
        figure_app_dependence(dependent),
        figure_app_heterogeneity(heterogeneity),
        figure_app_alpha(alpha_sensitivity),
        figure_app_small_calibration(small_calibration, infinite_volume_rate),
        figure_app_cqr_base_alpha(cqr_base_alpha),
        figure_app_cqr_shift(cqr_shift, cqr_shift_diagnostic),
        figure_app_contamination(contamination),
        figure_app_shape_template(shape_standard, shape_summary),
        figure_app_heavy_tails(heavy_tails),
    ]
    print("Generated figures:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()

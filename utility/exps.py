# Import general packages
import hashlib
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

# Import training packages
from sklearn.linear_model import LinearRegression, MultiTaskLasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Dataset loader
try:
    from ucimlrepo import fetch_ucirepo
except ImportError:  # Keep synthetic experiments importable without UCI tools.
    fetch_ucirepo = None
from scipy.io import arff

# Import utility packages
from utility.data_generator import (
    make_multitarget_regression,
    make_multitarget_regression_dependent_noise,
)
from utility.res_rescaled import check_coverage_rate, standardized_prediction
from utility.unscaled import unscaled_prediction, bonferroni_prediction
from utility.data_splitting import data_splitting_standardized_prediction, data_spliting_CHR_prediction, data_splitting_oracle_prediction, naive_prediction
from utility.copula import EmpiricalCopula, empirical_copula_prediction, inverse_ecdf_transform
from utility.cqhr import cqhr_adjustments
from utility.conformal_utils import add_jitter, conformal_quantile


METHOD_ALIASES = {
    "TSCP": "TSCP_R",
    "TSCP-S": "TSCP_S",
    "TSCP-GWC": "TSCP_GWC",
    "Point CHR": "Point_CHR",
    "QCH": "CQHR",
    "Quantile CHR": "CQHR",
    "Emp. copula": "Empirical_copula",
    "Unscaled Max": "Unscaled",
}


REAL_DATASETS = {
    "air",
    "crime",
    "energy",
    "rf1",
    "rf2",
    "scm1d",
    "scm20d",
    "stock",
    "student",
}


def _stable_hash(*args):
    """
    Create a reproducible hash from input arguments, useful for random seeds.

    Parameters
    ----------
    *args : Any
        Sequence of inputs to be hashed.

    Returns
    -------
    int
        A deterministic hash value based on inputs.
    """
    key = "_".join(str(a) for a in args)
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**32)


def _normalize_method_name(method: str) -> str:
    """Map display names used in figures/tables to internal method names."""
    return METHOD_ALIASES.get(method, method)


def _fetch_ucirepo_dataset(dataset_id: int):
    """Fetch a UCI dataset, with a clear error if ucimlrepo is unavailable."""
    if fetch_ucirepo is None:
        raise ImportError(
            "ucimlrepo is required for UCI real-data experiments. "
            "Install it to run run_real_experiments on UCI-backed datasets."
        )
    return fetch_ucirepo(id=dataset_id)


def _function_choice(scores, alpha, method, mu=None, std=None):
    """
    Select and run the prediction region method based on input string.

    Parameters
    ----------
    scores : np.ndarray
        Score values used to construct prediction regions.
    alpha : float
        Miscoverage level.
    method : str
        The method name specifying which region construction to use.

    Returns
    -------
    depends on method
        Either a single region (Rectangle) or list of regions.
    """

    method = _normalize_method_name(method)

    # Standardized methods
    if method == "TSCP_LWC":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=False)
    elif method == "TSCP_R":
        return standardized_prediction(scores=scores, alpha=alpha, short_cut=True)
    elif method == "TSCP_GWC":
        return standardized_prediction(scores=scores, alpha=alpha, method="GWC", short_cut=True)
    elif method == "TSCP_S":
        return data_splitting_standardized_prediction(scores=scores, alpha=alpha)
    elif method == "Population_oracle":
        if mu is None or std is None:
            raise ValueError("mu and std must be provided for Population_oracle.")
        return data_splitting_oracle_prediction(scores=scores, mu=mu, std=std, alpha=alpha)
    elif method == "Naive":
        return naive_prediction(scores=scores, alpha=alpha)
    
    # Conformalized hyper-rectangle
    elif method == "Point_CHR":
        return data_spliting_CHR_prediction(scores=scores, alpha=alpha)
    
    # Empirical copula
    elif method == "Empirical_copula":
        return empirical_copula_prediction(scores=scores, alpha=alpha)

    # No scaling methods
    elif method == "Unscaled":
        return unscaled_prediction(scores=scores, alpha=alpha)
    elif method == "Bonferroni":
        return bonferroni_prediction(scores=scores, alpha=alpha)

    raise ValueError(f"Unknown method: {method}")


# Backward-compatible aliases used by older functions in this module.
stable_hash = _stable_hash
function_choice = _function_choice


def _absolute_residual(x, y, model):
    """
    Compute coordinate-wise absolute residual scores.

    Given features x, responses y, and a fitted prediction model, this function
    computes

        S_ij = |y_pred_ij - y_ij|,

    where y_pred = model.predict(x).

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True response values.

    model : object
        A fitted regression model with a `.predict(x)` method. The prediction
        output should have shape compatible with y.

    Returns
    -------
    scores : np.ndarray of shape (n_samples, n_outputs)
        Coordinate-wise absolute residual scores.

    Raises
    ------
    ValueError
        If the predicted values and responses have incompatible shapes.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    y_pred = np.asarray(model.predict(x))

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    if y_pred.shape != y.shape:
        raise ValueError(
            f"Shape mismatch: y_pred has shape {y_pred.shape}, "
            f"but y has shape {y.shape}."
        )

    return np.abs(y_pred - y)


def _quantile_residual(
    x,
    y,
    lower_models,
    upper_models,
    const=None,
    cap_below_zero=False,
):
    """
    Compute coordinate-wise conformal quantile regression residual scores.

    For each sample i and coordinate j, this function computes the CQR score

        S_ij = max{ q_lower_j(x_i) - y_ij,
                    y_ij - q_upper_j(x_i) }.

    If `cap_below_zero=True`, the score becomes

        S_ij = max{ q_lower_j(x_i) - y_ij,
                    y_ij - q_upper_j(x_i),
                    0 }.

    The capped version assigns score zero to observations that already lie
    inside their fitted quantile interval.

    Optionally, a constant transformation can be applied:

        S_ij <- S_ij + const.

    Here `const` can be either a scalar or an array broadcastable to the score
    matrix, for example shape `(d,)`.

    Parameters
    ----------
    x : array-like of shape (n_samples, n_features)
        Feature matrix.

    y : array-like of shape (n_samples,) or (n_samples, d)
        True response values.

    lower_models : list
        List of fitted lower quantile models. The j-th model should predict
        the lower conditional quantile for coordinate j.

    upper_models : list
        List of fitted upper quantile models. The j-th model should predict
        the upper conditional quantile for coordinate j.

    const : float or array-like, optional
        Optional additive constant transformation. If provided, it is added
        to the final score matrix. It may be a scalar or an array broadcastable
        to shape `(n_samples, d)`.

    cap_below_zero : bool, default=False
        Whether to cap CQR scores below at zero.

    Returns
    -------
    scores : np.ndarray of shape (n_samples, d)
        Coordinate-wise CQR scores.

    Raises
    ------
    ValueError
        If the number of lower and upper models differ, or if y has shape
        incompatible with the number of models.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if len(lower_models) != len(upper_models):
        raise ValueError(
            "lower_models and upper_models must have the same length."
        )

    n = x.shape[0]
    dim = len(lower_models)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    if y.shape != (n, dim):
        raise ValueError(
            f"Shape mismatch: y has shape {y.shape}, "
            f"but expected shape {(n, dim)} based on x and models."
        )

    q_lower = np.zeros((n, dim))
    q_upper = np.zeros((n, dim))

    for j in range(dim):
        q_lower[:, j] = np.asarray(lower_models[j].predict(x)).reshape(-1)
        q_upper[:, j] = np.asarray(upper_models[j].predict(x)).reshape(-1)

    scores = np.maximum(
        q_lower - y,
        y - q_upper,
    )

    if cap_below_zero:
        scores = np.maximum(scores, 0.0)

    if const is not None:
        scores = scores + np.asarray(const)

    return scores


def _as_score_transform_list(score_transform: Union[str, Sequence[str]]) -> List[str]:
    """Normalize one or more CQR score transformations."""
    if isinstance(score_transform, str):
        transforms = [score_transform]
    else:
        transforms = list(score_transform)
    transforms = [transform.lower() for transform in transforms]
    valid = {"raw", "capped", "shifted"}
    unknown = set(transforms) - valid
    if unknown:
        raise ValueError(f"Unknown CQR score transform(s): {sorted(unknown)}.")
    return transforms


def _make_quantile_model(
    quantile: float,
    random_state: int,
    quantile_model_factory: Optional[Callable[[float, int], Any]] = None,
    quantile_model_params: Optional[Dict[str, Any]] = None,
):
    """Create one quantile regression model."""
    if quantile_model_factory is not None:
        return quantile_model_factory(quantile, random_state)

    params = {
        "n_estimators": 100,
        "max_depth": 3,
        "learning_rate": 0.05,
        "min_samples_leaf": 5,
        "random_state": random_state,
    }
    if quantile_model_params is not None:
        params.update(quantile_model_params)
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        **params,
    )


def _fit_coordinatewise_quantile_models(
    X_train,
    y_train,
    lower_quantile: float,
    upper_quantile: float,
    random_state: int,
    quantile_model_factory: Optional[Callable[[float, int], Any]] = None,
    quantile_model_params: Optional[Dict[str, Any]] = None,
):
    """Fit lower/upper quantile models independently for each target."""
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    if y_train.ndim == 1:
        y_train = y_train.reshape(-1, 1)

    lower_models = []
    upper_models = []
    for coordinate in range(y_train.shape[1]):
        lower_model = _make_quantile_model(
            quantile=lower_quantile,
            random_state=random_state + 2 * coordinate,
            quantile_model_factory=quantile_model_factory,
            quantile_model_params=quantile_model_params,
        )
        upper_model = _make_quantile_model(
            quantile=upper_quantile,
            random_state=random_state + 2 * coordinate + 1,
            quantile_model_factory=quantile_model_factory,
            quantile_model_params=quantile_model_params,
        )
        lower_model.fit(X_train, y_train[:, coordinate])
        upper_model.fit(X_train, y_train[:, coordinate])
        lower_models.append(lower_model)
        upper_models.append(upper_model)

    return lower_models, upper_models


def _predict_coordinatewise_quantiles(X, lower_models, upper_models):
    """Predict coordinate-wise lower and upper quantiles."""
    X = np.asarray(X)
    n = X.shape[0]
    dim = len(lower_models)
    q_lower = np.zeros((n, dim))
    q_upper = np.zeros((n, dim))

    for coordinate in range(dim):
        q_lower[:, coordinate] = np.asarray(
            lower_models[coordinate].predict(X)
        ).reshape(-1)
        q_upper[:, coordinate] = np.asarray(
            upper_models[coordinate].predict(X)
        ).reshape(-1)

    lower = np.minimum(q_lower, q_upper)
    upper = np.maximum(q_lower, q_upper)
    q_lower, q_upper = lower, upper
    return q_lower, q_upper


def _cqr_scores_and_base_lengths(X, y, lower_models, upper_models):
    """Compute raw signed CQR scores and base interval lengths."""
    y = np.asarray(y)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    q_lower, q_upper = _predict_coordinatewise_quantiles(
        X=X,
        lower_models=lower_models,
        upper_models=upper_models,
    )
    scores = np.maximum(q_lower - y, y - q_upper)
    base_lengths = q_upper - q_lower
    return scores, base_lengths


def _resolve_shift_constant(
    shift_constant: Optional[float],
    noise_list: np.ndarray,
) -> float:
    """Choose a calibration-independent shift constant for CQR scores."""
    if shift_constant is not None:
        return float(shift_constant)
    return float(10 * np.max(noise_list))


def _transform_cqr_scores(raw_scores, score_transform: str, shift_constant: float):
    """Transform raw CQR scores into nonnegative scores for TSCP-style methods."""
    if score_transform == "raw":
        return np.asarray(raw_scores, dtype=float)
    if score_transform == "capped":
        return np.maximum(raw_scores, 0.0)
    if score_transform == "shifted":
        shifted_scores = raw_scores + shift_constant
        min_shifted_score = np.min(shifted_scores)
        if min_shifted_score < -1e-12:
            raise ValueError(
                "shift_constant is not large enough to make shifted CQR "
                f"scores nonnegative. Minimum shifted score is {min_shifted_score:.4g}. "
                "Increase shift_constant."
            )
        return np.maximum(shifted_scores, 0.0)
    raise ValueError(f"Unknown CQR score transform: {score_transform}")


def _region_upper_bounds(prediction, is_lwc: bool) -> np.ndarray:
    """Extract coordinate-wise upper bounds from a method's score region."""
    if is_lwc:
        _, enclosing_region = prediction
        return enclosing_region.upper
    return prediction.upper


def _fit_score_region(
    *,
    method: str,
    scores_cal: np.ndarray,
    alpha: float,
    oracle_stats: Optional[Dict[str, np.ndarray]] = None,
):
    """Fit one conformal score region and return prediction metadata."""
    start = time.time()

    if method == "TSCP_LWC":
        raise ValueError(
            "TSCP_LWC is not supported by the CQR outcome-space experiment. "
            "Use TSCP_R or TSCP_GWC for coordinate-wise interval comparisons."
        )

    if method == "Population_oracle":
        if oracle_stats is None:
            raise ValueError("oracle_stats must be provided for Population_oracle.")
        prediction = _function_choice(
            scores=scores_cal,
            alpha=alpha,
            method=method,
            mu=oracle_stats["mu"],
            std=oracle_stats["std"],
        )
    else:
        prediction = _function_choice(
            scores=scores_cal,
            alpha=alpha,
            method=method,
        )

    return prediction, False, time.time() - start


def _fit_raw_cqr_baseline_adjustments(
    *,
    method: str,
    raw_scores_cal: np.ndarray,
    alpha: float,
):
    """
    Fit baselines that can operate directly on raw signed CQR scores.

    These methods return adjustment vectors instead of `Rectangle` objects
    because raw CQR adjustments may be negative, corresponding to shrinking the
    base quantile interval in outcome space.
    """
    start = time.time()
    raw_scores_cal = np.asarray(raw_scores_cal, dtype=float)
    _, dim = raw_scores_cal.shape

    if method == "Unscaled":
        max_scores = add_jitter(np.max(raw_scores_cal, axis=1), random_state=42)
        threshold = conformal_quantile(max_scores, alpha)
        adjustments = np.repeat(threshold, dim)
    elif method == "Empirical_copula":
        cop = EmpiricalCopula()
        cop.fit(raw_scores_cal)
        thresholds = cop.quantile_box(alpha=alpha)
        adjustments = inverse_ecdf_transform(thresholds, raw_scores_cal)
    else:
        raise ValueError(f"{method} is not configured as a raw CQR baseline.")

    return adjustments, time.time() - start


def _evaluate_cqhr_outcome_metrics(
    *,
    raw_scores_cal: np.ndarray,
    base_lengths_cal: np.ndarray,
    raw_scores_test: np.ndarray,
    base_lengths_test: np.ndarray,
    alpha: float,
    log_scale: bool,
):
    """Evaluate CQHR using raw signed CQR scores and outcome-space lengths."""
    start = time.time()
    adjustments, reference_adjustment, _ = cqhr_adjustments(
        raw_scores_cal=raw_scores_cal,
        base_lengths_cal=base_lengths_cal,
        base_lengths_test=base_lengths_test,
        alpha=alpha,
    )
    runtime = time.time() - start

    covered = np.all(raw_scores_test <= adjustments, axis=1)
    adjusted_lengths = np.maximum(base_lengths_test + 2 * adjustments, 0.0)
    trial_volume = np.mean(np.prod(adjusted_lengths, axis=1))
    if log_scale:
        trial_volume = _safe_log10(trial_volume)

    average_coordinate_lengths = np.mean(adjusted_lengths, axis=0)
    average_coordinate_adjustments = np.mean(adjustments, axis=0)

    return {
        "test_coverage": np.mean(covered),
        "coverage_volume": trial_volume,
        "coverage_max_length": np.max(average_coordinate_lengths),
        "coordinate_lengths": average_coordinate_lengths,
        "coordinate_adjustments": average_coordinate_adjustments,
        "reference_adjustment": reference_adjustment,
        "runtime": runtime,
    }


@dataclass
class ExperimentResults:
    """
    Container for experiment outputs.

    Attributes
    ----------
    trial_results : pd.DataFrame
        Long-format results with one row per
        (alpha, n_dim, n_cal, trial, method).

    summary_results : pd.DataFrame
        Aggregated results with one row per
        (alpha, n_dim, n_cal, method, noise_type).

    config : dict
        Experiment settings used to generate the results.

    coordinate_trial_results : pd.DataFrame or None
        Long-format coordinate-wise region lengths with one row per
        (alpha, n_dim, n_cal, trial, method, coordinate).

    coordinate_summary_results : pd.DataFrame or None
        Aggregated coordinate-wise lengths with one row per
        (alpha, n_dim, n_cal, method, coordinate, noise_type).

    trial_csv_path : str or None
        Path to the saved trial-level CSV file, if saved.

    summary_csv_path : str or None
        Path to the saved summary-level CSV file, if saved.

    coordinate_trial_csv_path : str or None
        Path to the saved coordinate-level trial CSV file, if saved.

    coordinate_summary_csv_path : str or None
        Path to the saved coordinate-level summary CSV file, if saved.
    """
    trial_results: pd.DataFrame
    summary_results: pd.DataFrame
    config: Dict[str, Any]
    coordinate_trial_results: Optional[pd.DataFrame] = None
    coordinate_summary_results: Optional[pd.DataFrame] = None
    trial_csv_path: Optional[str] = None
    summary_csv_path: Optional[str] = None
    coordinate_trial_csv_path: Optional[str] = None
    coordinate_summary_csv_path: Optional[str] = None


def _as_method_list(methods: Optional[Sequence[str]] = None, method: Optional[str] = None) -> List[str]:
    """Normalize either `methods=[...]` or legacy `method="..."` input."""
    if methods is None and method is None:
        return ["TSCP_R"]
    if methods is None:
        return [_normalize_method_name(method)]
    if isinstance(methods, str):
        return [_normalize_method_name(methods)]
    methods = [_normalize_method_name(method_name) for method_name in methods]
    if not methods:
        raise ValueError("methods must contain at least one method name.")
    return methods


def _get_noise_list(dim: int, index_dim: int, noises_list: Optional[Sequence]) -> np.ndarray:
    """Resolve coordinate-wise noise levels for a given response dimension."""
    if noises_list is None:
        return dim - np.arange(dim)
    return np.asarray(noises_list[index_dim])


def _safe_log10(value: float) -> float:
    """Compute log10(value), returning -inf for zero and raising on negative input."""
    if value < 0:
        raise ValueError(f"Cannot take log10 of a negative volume: {value}")
    if value == 0:
        return -np.inf
    return np.log10(value)


def _summarize_trial_results(trial_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trial-level results into summary-level results."""
    group_cols = ["alpha", "n_dim", "n_cal", "method", "noise_type"]
    extra_group_cols = [
        col
        for col in [
            "df",
            "correlation",
            "correlation_structure",
            "score_type",
            "score_transform",
            "base_interval_alpha",
            "shift_constant",
        ]
        if col in trial_df.columns
    ]
    group_cols += extra_group_cols

    return (
        trial_df
        .groupby(group_cols, as_index=False)
        .agg(
            n_trials=("trial", "nunique"),
            test_coverage_avg=("test_coverage", "mean"),
            test_coverage_1std=("test_coverage", "std"),
            coverage_vol_avg=("coverage_volume", "mean"),
            coverage_vol_1std=("coverage_volume", "std"),
            coverage_max_length_median=("coverage_max_length", "median"),
            runtime_avg=("runtime", "mean"),
        )
    )


def _summarize_coordinate_results(coordinate_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coordinate-wise trial lengths into summary-level results."""
    group_cols = ["alpha", "n_dim", "n_cal", "method", "noise_type", "coordinate"]
    extra_group_cols = [
        col
        for col in [
            "df",
            "correlation",
            "correlation_structure",
            "score_type",
            "score_transform",
            "base_interval_alpha",
            "shift_constant",
        ]
        if col in coordinate_df.columns
    ]
    group_cols += extra_group_cols

    aggregations = {
        "n_trials": ("trial", "nunique"),
        "coordinate_length_avg": ("coordinate_length", "mean"),
        "coordinate_length_1std": ("coordinate_length", "std"),
        "coordinate_length_median": ("coordinate_length", "median"),
    }
    if "coordinate_base_length" in coordinate_df.columns:
        aggregations["coordinate_base_length_avg"] = ("coordinate_base_length", "mean")
    if "coordinate_adjustment" in coordinate_df.columns:
        aggregations["coordinate_adjustment_avg"] = ("coordinate_adjustment", "mean")

    return coordinate_df.groupby(group_cols, as_index=False).agg(**aggregations)


def _save_experiment_csvs(
    trial_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]],
    experiment_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Save trial-level and summary-level CSV files if `output_dir` is provided."""
    if output_dir is None:
        return None, None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial_csv_path = output_path / f"{experiment_name}_trial_results_{timestamp}.csv"
    summary_csv_path = output_path / f"{experiment_name}_summary_results_{timestamp}.csv"

    trial_df.to_csv(trial_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    return str(trial_csv_path), str(summary_csv_path)


def _save_coordinate_csvs(
    coordinate_trial_df: pd.DataFrame,
    coordinate_summary_df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]],
    experiment_name: str,
) -> tuple[Optional[str], Optional[str]]:
    """Save coordinate-wise CSV files if `output_dir` is provided."""
    if output_dir is None:
        return None, None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    coordinate_trial_csv_path = output_path / f"{experiment_name}_coordinate_trial_results_{timestamp}.csv"
    coordinate_summary_csv_path = output_path / f"{experiment_name}_coordinate_summary_results_{timestamp}.csv"

    coordinate_trial_df.to_csv(coordinate_trial_csv_path, index=False)
    coordinate_summary_df.to_csv(coordinate_summary_csv_path, index=False)

    return str(coordinate_trial_csv_path), str(coordinate_summary_csv_path)


def _recordable_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Keep scalar generator metadata, such as df, in trial/summary tables."""
    if not metadata:
        return {}
    return {
        key: value
        for key, value in metadata.items()
        if np.isscalar(value) or isinstance(value, str)
    }


def _region_volume_and_length(prediction, log_scale: bool, is_lwc: bool) -> tuple[float, float]:
    """
    Extract volume and max coordinate length from method output.

    For TSCP_LWC, `prediction` is expected to be `(regions, enclosing_region)`.
    For all other methods, `prediction` is expected to be a single region object.
    """
    if is_lwc:
        regions, enclosing_region = prediction
        volume = sum(reg.volume() for reg in regions)
        max_length = np.max(enclosing_region.length_along_dimensions())
    else:
        region = prediction
        volume = region.volume()
        max_length = np.max(region.length_along_dimensions())

    if log_scale:
        volume = _safe_log10(volume)

    return volume, max_length


def _region_coordinate_lengths(prediction, is_lwc: bool) -> np.ndarray:
    """
    Extract coordinate-wise lengths from a prediction region.

    For TSCP_LWC, use the enclosing rectangle for a directly interpretable
    coordinate-wise width. For single-rectangle methods, use that rectangle.
    """
    if is_lwc:
        _, enclosing_region = prediction
        return enclosing_region.length_along_dimensions()
    return prediction.length_along_dimensions()


def _evaluate_one_method(
    *,
    method: str,
    scores_cal: np.ndarray,
    scores_test: np.ndarray,
    alpha: float,
    log_scale: bool,
    oracle_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, float]:
    """Run one conformal region method on fixed calibration and test scores."""
    start = time.time()

    if method == "Population_oracle":
        if oracle_stats is None:
            raise ValueError("oracle_stats must be provided for Population_oracle.")
        prediction = _function_choice(
            scores=scores_cal,
            alpha=alpha,
            method=method,
            mu=oracle_stats["mu"],
            std=oracle_stats["std"],
        )
        one_rect = True
        coverage_regions = prediction
        is_lwc = False

    elif method == "TSCP_LWC":
        prediction = _function_choice(scores=scores_cal, alpha=alpha, method=method)
        regions, _ = prediction
        one_rect = False
        coverage_regions = regions
        is_lwc = True

    else:
        prediction = _function_choice(scores=scores_cal, alpha=alpha, method=method)
        one_rect = True
        coverage_regions = prediction
        is_lwc = False

    runtime = time.time() - start

    coverage = check_coverage_rate(
        scores=scores_test,
        regions=coverage_regions,
        one_rect=one_rect,
    )
    volume, max_length = _region_volume_and_length(
        prediction=prediction,
        log_scale=log_scale,
        is_lwc=is_lwc,
    )
    coordinate_lengths = _region_coordinate_lengths(
        prediction=prediction,
        is_lwc=is_lwc,
    )

    return {
        "test_coverage": coverage,
        "coverage_volume": volume,
        "coverage_max_length": max_length,
        "coordinate_lengths": coordinate_lengths,
        "runtime": runtime,
    }


def run_abs_res_synthetic_experiment(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    noise_type: str = "Gaussian",
    noises_list: Optional[Sequence] = None,
    trials: int = 300,
    methods: Optional[Sequence[str]] = None,
    method: Optional[str] = None,
    log_scale: bool = False,
    model_factory: Optional[Callable[[], Any]] = None,
    n_train_pool: int = 8000,
    n_features: int = 10,
    n_informative: int = 10,
    test_size: float = 0.2,
    oracle_n_samples: int = 100000,
    oracle_seed_offset: int = 3,
    data_generator: Callable[..., Any] = make_multitarget_regression,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "abs_res_synthetic_experiment",
    return_dataclass: bool = True,
) -> Union[ExperimentResults, pd.DataFrame]:
    """
    Run synthetic experiments for absolute-residual conformal score methods.

    This refactored version compares multiple methods using the same fitted
    regression model, calibration scores, and test scores within each Monte
    Carlo trial. This avoids refitting the same model separately for every
    method.

    The absolute residual score is

        S_ij = |model(X_i)_j - Y_ij|.

    Parameters
    ----------
    dim_list : list of int
        Response dimensions to evaluate.
    sample_list : list of int
        Calibration sample sizes.
    alpha_list : list of float
        Miscoverage levels.
    noise_type : str, default="Gaussian"
        Noise type passed to `make_multitarget_regression`.
    noises_list : sequence, optional
        Optional coordinate-wise noise levels. If None, the default for
        dimension d is `[d, d-1, ..., 1]`.
    trials : int, default=300
        Number of Monte Carlo trials per configuration.
    methods : sequence of str, optional
        Methods to compare. Each name is passed to `_function_choice`.
    method : str, optional
        Legacy single-method argument. Used only when `methods` is None.
    log_scale : bool, default=False
        If True, save log10 volume instead of raw volume.
    model_factory : callable, optional
        Function returning a fresh sklearn-style model. Defaults to
        `LinearRegression`.
    n_train_pool : int, default=8000
        Number of samples in the generated train/test pool.
    n_features : int, default=10
        Number of features in the synthetic regression problem.
    n_informative : int, default=10
        Number of informative features used for calibration/oracle generation.
    test_size : float, default=0.2
        Test split fraction for the generated train/test pool.
    oracle_n_samples : int, default=100000
        Number of oracle samples for `Population_oracle`.
    oracle_seed_offset : int, default=3
        Offset added to the trial seed when simulating oracle samples.
    data_generator : callable, default=make_multitarget_regression
        Synthetic data generator with the same return contract as
        `make_multitarget_regression`.
    generator_kwargs : dict, optional
        Extra keyword arguments passed to `data_generator`, such as
        `{"df": 3}` for t-distributed noise or `{"correlation": 0.5}` for
        dependent Gaussian noise.
    output_dir : str or pathlib.Path, optional
        Directory where trial-level and summary-level CSV files are saved.
        If None, no CSV files are written.
    experiment_name : str, default="abs_res_synthetic_experiment"
        Prefix used for saved CSV files.
    return_dataclass : bool, default=True
        If True, return an `ExperimentResults` object. If False, return only
        the summary DataFrame for backward compatibility.

    Returns
    -------
    ExperimentResults or pd.DataFrame
        By default, returns `ExperimentResults`, which contains both raw
        trial-level results and summary results. If `return_dataclass=False`,
        returns only the summary DataFrame.
    """
    methods = _as_method_list(methods=methods, method=method)

    if trials <= 0:
        raise ValueError("trials must be positive.")
    if model_factory is None:
        model_factory = LinearRegression
    generator_kwargs = dict(generator_kwargs or {})
    metadata = _recordable_metadata(generator_kwargs)

    records = []
    coordinate_records = []
    needs_oracle = "Population_oracle" in methods

    for index_dim, dim in enumerate(dim_list):
        noise_list = _get_noise_list(dim=dim, index_dim=index_dim, noises_list=noises_list)

        X, y, coef_true = data_generator(
            n_samples=n_train_pool,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=dim,
            noise_type=noise_type,
            noise_list=noise_list,
            random_state=_stable_hash(dim),
            **generator_kwargs,
        )

        for sample in sample_list:
            for trial in range(trials):
                seed = _stable_hash(dim, sample, trial)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=seed + 42,
                )

                model = model_factory()
                model.fit(X_train, y_train)

                scores_test = _absolute_residual(X_test, y_test, model)

                X_cal, y_cal = data_generator(
                    n_samples=sample,
                    n_features=n_features,
                    n_informative=n_informative,
                    n_targets=dim,
                    noise_type=noise_type,
                    noise_list=noise_list,
                    random_state=seed,
                    coef=coef_true,
                    **generator_kwargs,
                )
                scores_cal = _absolute_residual(X_cal, y_cal, model)

                oracle_stats = None
                if needs_oracle:
                    oracle_X, oracle_y = data_generator(
                        n_samples=oracle_n_samples,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_targets=dim,
                        noise_type=noise_type,
                        noise_list=noise_list,
                        random_state=seed + oracle_seed_offset,
                        coef=coef_true,
                        **generator_kwargs,
                    )
                    oracle_scores = _absolute_residual(oracle_X, oracle_y, model)
                    oracle_stats = {
                        "mu": np.mean(oracle_scores, axis=0),
                        "std": np.std(oracle_scores, axis=0, ddof=1),
                    }

                for alpha in alpha_list:
                    for method_name in methods:
                        metrics = _evaluate_one_method(
                            method=method_name,
                            scores_cal=scores_cal,
                            scores_test=scores_test,
                            alpha=alpha,
                            log_scale=log_scale,
                            oracle_stats=oracle_stats,
                        )
                        coordinate_lengths = metrics.pop("coordinate_lengths")

                        record = {
                            "alpha": alpha,
                            "n_dim": dim,
                            "n_cal": sample,
                            "trial": trial,
                            "method": method_name,
                            "noise_type": noise_type,
                            "score_type": "absolute_residual",
                            **metrics,
                            **metadata,
                        }
                        records.append(record)
                        for coordinate, coordinate_length in enumerate(coordinate_lengths, start=1):
                            coordinate_records.append({
                                "alpha": alpha,
                                "n_dim": dim,
                                "n_cal": sample,
                                "trial": trial,
                                "method": method_name,
                                "noise_type": noise_type,
                                "score_type": "absolute_residual",
                                "coordinate": coordinate,
                                "coordinate_length": coordinate_length,
                                **metadata,
                            })

    trial_df = pd.DataFrame.from_records(records)
    summary_df = _summarize_trial_results(trial_df)
    coordinate_trial_df = pd.DataFrame.from_records(coordinate_records)
    coordinate_summary_df = _summarize_coordinate_results(coordinate_trial_df)

    trial_csv_path, summary_csv_path = _save_experiment_csvs(
        trial_df=trial_df,
        summary_df=summary_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    coordinate_trial_csv_path, coordinate_summary_csv_path = _save_coordinate_csvs(
        coordinate_trial_df=coordinate_trial_df,
        coordinate_summary_df=coordinate_summary_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    config = {
        "dim_list": dim_list,
        "sample_list": sample_list,
        "alpha_list": alpha_list,
        "methods": methods,
        "noise_type": noise_type,
        "trials": trials,
        "log_scale": log_scale,
        "n_train_pool": n_train_pool,
        "n_features": n_features,
        "n_informative": n_informative,
        "test_size": test_size,
        "oracle_n_samples": oracle_n_samples,
        "oracle_seed_offset": oracle_seed_offset,
        "data_generator": getattr(data_generator, "__name__", str(data_generator)),
        "generator_kwargs": generator_kwargs,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "experiment_name": experiment_name,
        "trial_csv_path": trial_csv_path,
        "summary_csv_path": summary_csv_path,
        "coordinate_trial_csv_path": coordinate_trial_csv_path,
        "coordinate_summary_csv_path": coordinate_summary_csv_path,
    }

    result = ExperimentResults(
        trial_results=trial_df,
        summary_results=summary_df,
        config=config,
        coordinate_trial_results=coordinate_trial_df,
        coordinate_summary_results=coordinate_summary_df,
        trial_csv_path=trial_csv_path,
        summary_csv_path=summary_csv_path,
        coordinate_trial_csv_path=coordinate_trial_csv_path,
        coordinate_summary_csv_path=coordinate_summary_csv_path,
    )

    if return_dataclass:
        return result
    return summary_df


def run_synthetic_experiment(*args, **kwargs) -> pd.DataFrame:
    """Backward-compatible name for absolute-residual synthetic experiments."""
    kwargs.setdefault("return_dataclass", False)
    return run_abs_res_synthetic_experiment(*args, **kwargs)


def run_abs_res_dependent_gaussian_experiment(
    *args,
    correlation: float = 0.5,
    correlation_structure: str = "equicorrelated",
    generator_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[ExperimentResults, pd.DataFrame]:
    """
    Run synthetic absolute-residual experiments with dependent Gaussian noise.

    This is a convenience wrapper around `run_abs_res_synthetic_experiment`.
    It keeps the same experiment loop and method evaluation code, changing only
    the data generator to draw target noise jointly across dimensions.
    """
    dependent_kwargs = dict(generator_kwargs or {})
    dependent_kwargs.update({
        "correlation": correlation,
        "correlation_structure": correlation_structure,
    })
    kwargs.setdefault("noise_type", "Gaussian")
    kwargs.setdefault("experiment_name", "abs_res_dependent_gaussian_experiment")
    return run_abs_res_synthetic_experiment(
        *args,
        data_generator=make_multitarget_regression_dependent_noise,
        generator_kwargs=dependent_kwargs,
        **kwargs,
    )


def run_cqr_synthetic_experiment(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    noise_type: str = "Gaussian",
    noises_list: Optional[Sequence] = None,
    trials: int = 300,
    methods: Optional[Sequence[str]] = None,
    method: Optional[str] = None,
    score_transform: Union[str, Sequence[str]] = "capped",
    raw_score_methods: Optional[Sequence[str]] = None,
    shift_constant: Optional[float] = None,
    base_interval_alpha: Optional[float] = None,
    quantile_model_factory: Optional[Callable[[float, int], Any]] = None,
    quantile_model_params: Optional[Dict[str, Any]] = None,
    log_scale: bool = False,
    n_train_pool: int = 8000,
    n_features: int = 10,
    n_informative: int = 10,
    test_size: float = 0.2,
    oracle_n_samples: int = 100000,
    oracle_seed_offset: int = 3,
    data_generator: Callable[..., Any] = make_multitarget_regression,
    generator_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    experiment_name: str = "cqr_synthetic_experiment",
    return_dataclass: bool = True,
) -> Union[ExperimentResults, pd.DataFrame]:
    """
    Run synthetic experiments using coordinate-wise CQR scores.

    Raw CQR scores are signed:

        S_ij = max{ q_lower_j(X_i) - Y_ij,
                    Y_ij - q_upper_j(X_i) }.

    Since TSCP-style methods require nonnegative scores, this runner supports:
    - `score_transform="raw"`: use signed CQR scores directly for methods that
      can work with negative adjustments;
    - `score_transform="capped"`: use max(S_ij, 0);
    - `score_transform="shifted"`: use S_ij + C, then subtract C from the
      fitted coordinate-wise adjustment before computing outcome-space metrics.

    By default, `Unscaled` and `Empirical_copula` are evaluated on raw signed
    CQR scores even when TSCP-style methods use capped or shifted scores,
    because these baselines can be interpreted directly as thresholding raw CQR
    adjustments. Override `raw_score_methods=[]` to force all non-CQHR methods
    through `score_transform`.

    The `CQHR` method is handled natively from raw signed CQR scores following
    Algorithm 2 of Sampson and Chan (2024), so it is evaluated once with
    `score_transform="native"` regardless of the requested TSCP score transforms.

    For CQR, volume and coordinate lengths are measured in the original outcome
    space and averaged over the test covariates.
    """
    methods = _as_method_list(methods=methods, method=method)
    score_transforms = _as_score_transform_list(score_transform)
    score_region_methods = [method_name for method_name in methods if method_name != "CQHR"]
    use_cqhr = "CQHR" in methods
    if raw_score_methods is None:
        raw_score_methods = ["Unscaled", "Empirical_copula"]
    raw_score_methods = {
        _normalize_method_name(method_name)
        for method_name in raw_score_methods
    }

    if "TSCP_LWC" in methods:
        raise ValueError(
            "TSCP_LWC is not supported by run_cqr_synthetic_experiment. "
            "Use TSCP_R or TSCP_GWC for CQR coordinate-wise comparisons."
        )
    if trials <= 0:
        raise ValueError("trials must be positive.")

    generator_kwargs = dict(generator_kwargs or {})
    metadata = _recordable_metadata(generator_kwargs)
    records = []
    coordinate_records = []
    needs_oracle = "Population_oracle" in score_region_methods

    for index_dim, dim in enumerate(dim_list):
        noise_list = _get_noise_list(dim=dim, index_dim=index_dim, noises_list=noises_list)

        X, y, coef_true = data_generator(
            n_samples=n_train_pool,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=dim,
            noise_type=noise_type,
            noise_list=noise_list,
            random_state=_stable_hash(dim),
            **generator_kwargs,
        )

        for sample in sample_list:
            for trial in range(trials):
                seed = _stable_hash(dim, sample, trial)

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=seed + 42,
                )

                X_cal, y_cal = data_generator(
                    n_samples=sample,
                    n_features=n_features,
                    n_informative=n_informative,
                    n_targets=dim,
                    noise_type=noise_type,
                    noise_list=noise_list,
                    random_state=seed,
                    coef=coef_true,
                    **generator_kwargs,
                )

                oracle_X = None
                oracle_y = None
                if needs_oracle:
                    oracle_X, oracle_y = data_generator(
                        n_samples=oracle_n_samples,
                        n_features=n_features,
                        n_informative=n_informative,
                        n_targets=dim,
                        noise_type=noise_type,
                        noise_list=noise_list,
                        random_state=seed + oracle_seed_offset,
                        coef=coef_true,
                        **generator_kwargs,
                    )

                for alpha in alpha_list:
                    interval_alpha = base_interval_alpha if base_interval_alpha is not None else alpha
                    lower_quantile = interval_alpha / 2
                    upper_quantile = 1 - interval_alpha / 2
                    quantile_seed = _stable_hash(dim, sample, trial, alpha, "cqr")

                    lower_models, upper_models = _fit_coordinatewise_quantile_models(
                        X_train=X_train,
                        y_train=y_train,
                        lower_quantile=lower_quantile,
                        upper_quantile=upper_quantile,
                        random_state=quantile_seed,
                        quantile_model_factory=quantile_model_factory,
                        quantile_model_params=quantile_model_params,
                    )

                    raw_scores_cal, base_lengths_cal = _cqr_scores_and_base_lengths(
                        X=X_cal,
                        y=y_cal,
                        lower_models=lower_models,
                        upper_models=upper_models,
                    )
                    raw_scores_test, base_lengths_test = _cqr_scores_and_base_lengths(
                        X=X_test,
                        y=y_test,
                        lower_models=lower_models,
                        upper_models=upper_models,
                    )

                    raw_scores_oracle = None
                    if needs_oracle:
                        raw_scores_oracle, _ = _cqr_scores_and_base_lengths(
                            X=oracle_X,
                            y=oracle_y,
                            lower_models=lower_models,
                            upper_models=upper_models,
                        )

                    if use_cqhr:
                        metrics = _evaluate_cqhr_outcome_metrics(
                            raw_scores_cal=raw_scores_cal,
                            base_lengths_cal=base_lengths_cal,
                            raw_scores_test=raw_scores_test,
                            base_lengths_test=base_lengths_test,
                            alpha=alpha,
                            log_scale=log_scale,
                        )
                        average_coordinate_lengths = metrics.pop("coordinate_lengths")
                        average_coordinate_adjustments = metrics.pop("coordinate_adjustments")

                        record = {
                            "alpha": alpha,
                            "n_dim": dim,
                            "n_cal": sample,
                            "trial": trial,
                            "method": "CQHR",
                            "noise_type": noise_type,
                            "score_type": "cqr",
                            "score_transform": "native",
                            "base_interval_alpha": interval_alpha,
                            "shift_constant": 0.0,
                            **metrics,
                            **metadata,
                        }
                        records.append(record)

                        base_coordinate_lengths = np.mean(base_lengths_test, axis=0)
                        for coordinate, coordinate_length in enumerate(
                            average_coordinate_lengths,
                            start=1,
                        ):
                            coordinate_records.append({
                                "alpha": alpha,
                                "n_dim": dim,
                                "n_cal": sample,
                                "trial": trial,
                                "method": "CQHR",
                                "noise_type": noise_type,
                                "score_type": "cqr",
                                "score_transform": "native",
                                "base_interval_alpha": interval_alpha,
                                "shift_constant": 0.0,
                                "coordinate": coordinate,
                                "coordinate_base_length": base_coordinate_lengths[coordinate - 1],
                                "coordinate_adjustment": average_coordinate_adjustments[coordinate - 1],
                                "coordinate_length": coordinate_length,
                                **metadata,
                            })

                    for method_name in score_region_methods:
                        method_transforms = (
                            ["raw"]
                            if method_name in raw_score_methods
                            else score_transforms
                        )
                        for transform in method_transforms:
                            current_shift_constant = (
                                _resolve_shift_constant(shift_constant, noise_list)
                                if transform == "shifted"
                                else 0.0
                            )
                            scores_cal = _transform_cqr_scores(
                                raw_scores=raw_scores_cal,
                                score_transform=transform,
                                shift_constant=current_shift_constant,
                            )

                            oracle_stats = None
                            if needs_oracle and method_name == "Population_oracle":
                                oracle_scores = _transform_cqr_scores(
                                    raw_scores=raw_scores_oracle,
                                    score_transform=transform,
                                    shift_constant=current_shift_constant,
                                )
                                oracle_stats = {
                                    "mu": np.mean(oracle_scores, axis=0),
                                    "std": np.std(oracle_scores, axis=0, ddof=1),
                                }

                            if transform == "raw":
                                adjustments, runtime = _fit_raw_cqr_baseline_adjustments(
                                    method=method_name,
                                    raw_scores_cal=scores_cal,
                                    alpha=alpha,
                                )
                            else:
                                prediction, is_lwc, runtime = _fit_score_region(
                                    method=method_name,
                                    scores_cal=scores_cal,
                                    alpha=alpha,
                                    oracle_stats=oracle_stats,
                                )
                                score_upper_bounds = _region_upper_bounds(
                                    prediction=prediction,
                                    is_lwc=is_lwc,
                                )
                                adjustments = (
                                    score_upper_bounds - current_shift_constant
                                    if transform == "shifted"
                                    else score_upper_bounds
                                )

                            covered = np.all(raw_scores_test <= adjustments, axis=1)
                            adjusted_lengths = np.maximum(
                                base_lengths_test + 2 * adjustments,
                                0.0,
                            )
                            trial_volume = np.mean(np.prod(adjusted_lengths, axis=1))
                            if log_scale:
                                trial_volume = _safe_log10(trial_volume)

                            average_coordinate_lengths = np.mean(adjusted_lengths, axis=0)
                            max_average_length = np.max(average_coordinate_lengths)

                            record = {
                                "alpha": alpha,
                                "n_dim": dim,
                                "n_cal": sample,
                                "trial": trial,
                                "method": method_name,
                                "noise_type": noise_type,
                                "score_type": "cqr",
                                "score_transform": transform,
                                "base_interval_alpha": interval_alpha,
                                "shift_constant": current_shift_constant,
                                "test_coverage": np.mean(covered),
                                "coverage_volume": trial_volume,
                                "coverage_max_length": max_average_length,
                                "runtime": runtime,
                                **metadata,
                            }
                            records.append(record)

                            base_coordinate_lengths = np.mean(base_lengths_test, axis=0)
                            for coordinate, coordinate_length in enumerate(
                                average_coordinate_lengths,
                                start=1,
                            ):
                                coordinate_records.append({
                                    "alpha": alpha,
                                    "n_dim": dim,
                                    "n_cal": sample,
                                    "trial": trial,
                                    "method": method_name,
                                    "noise_type": noise_type,
                                    "score_type": "cqr",
                                    "score_transform": transform,
                                    "base_interval_alpha": interval_alpha,
                                    "shift_constant": current_shift_constant,
                                    "coordinate": coordinate,
                                    "coordinate_base_length": base_coordinate_lengths[coordinate - 1],
                                    "coordinate_adjustment": adjustments[coordinate - 1],
                                    "coordinate_length": coordinate_length,
                                    **metadata,
                                })

    trial_df = pd.DataFrame.from_records(records)
    summary_df = _summarize_trial_results(trial_df)
    coordinate_trial_df = pd.DataFrame.from_records(coordinate_records)
    coordinate_summary_df = _summarize_coordinate_results(coordinate_trial_df)

    trial_csv_path, summary_csv_path = _save_experiment_csvs(
        trial_df=trial_df,
        summary_df=summary_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )
    coordinate_trial_csv_path, coordinate_summary_csv_path = _save_coordinate_csvs(
        coordinate_trial_df=coordinate_trial_df,
        coordinate_summary_df=coordinate_summary_df,
        output_dir=output_dir,
        experiment_name=experiment_name,
    )

    config = {
        "dim_list": dim_list,
        "sample_list": sample_list,
        "alpha_list": alpha_list,
        "methods": methods,
        "noise_type": noise_type,
        "score_transform": score_transforms,
        "raw_score_methods": sorted(raw_score_methods),
        "shift_constant": shift_constant,
        "base_interval_alpha": base_interval_alpha,
        "trials": trials,
        "log_scale": log_scale,
        "n_train_pool": n_train_pool,
        "n_features": n_features,
        "n_informative": n_informative,
        "test_size": test_size,
        "oracle_n_samples": oracle_n_samples,
        "oracle_seed_offset": oracle_seed_offset,
        "data_generator": getattr(data_generator, "__name__", str(data_generator)),
        "generator_kwargs": generator_kwargs,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "experiment_name": experiment_name,
        "trial_csv_path": trial_csv_path,
        "summary_csv_path": summary_csv_path,
        "coordinate_trial_csv_path": coordinate_trial_csv_path,
        "coordinate_summary_csv_path": coordinate_summary_csv_path,
    }

    result = ExperimentResults(
        trial_results=trial_df,
        summary_results=summary_df,
        config=config,
        coordinate_trial_results=coordinate_trial_df,
        coordinate_summary_results=coordinate_summary_df,
        trial_csv_path=trial_csv_path,
        summary_csv_path=summary_csv_path,
        coordinate_trial_csv_path=coordinate_trial_csv_path,
        coordinate_summary_csv_path=coordinate_summary_csv_path,
    )

    if return_dataclass:
        return result
    return summary_df


def heavy_t(
    dim_list: List[int],
    sample_list: List[int],
    alpha_list: List[float],
    df_list: Optional[Sequence[int]] = None,
    trials: int = 300,
    method: str = "TSCP_R",
    log_scale: bool = False,
) -> pd.DataFrame:
    """Run absolute-residual synthetic experiments with t-distributed noise."""
    if df_list is None:
        df_list = [2, 3, 10, 30, 50, 100]

    summaries = []
    for df in df_list:
        summary = run_abs_res_synthetic_experiment(
            dim_list=dim_list,
            sample_list=sample_list,
            alpha_list=alpha_list,
            noise_type="t",
            trials=trials,
            method=method,
            log_scale=log_scale,
            oracle_n_samples=10000,
            oracle_seed_offset=0,
            generator_kwargs={"df": df},
            return_dataclass=False,
        )
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame(columns=[
            "alpha", "n_dim", "n_cals", "df", "n_trials", "noise_type",
            "test_coverage_avg", "test_coverage_1std",
            "coverage_vol_avg", "coverage_vol_1std",
            "coverage_max_length_median",
            "runtime_avg",
        ])

    output = pd.concat(summaries, ignore_index=True).rename(columns={"n_cal": "n_cals"})
    columns = [
        "alpha", "n_dim", "n_cals", "df", "n_trials", "noise_type",
        "test_coverage_avg", "test_coverage_1std",
        "coverage_vol_avg", "coverage_vol_1std",
        "coverage_max_length_median",
        "runtime_avg",
    ]
    return output[columns]


def _random_forest_model(
    *,
    n_estimators=200,
    max_depth=16,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
):
    """Shared RandomForestRegressor factory for real-data experiments."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1,
    )


def _load_arff_dataset(path: str, n_features: int):
    """Load an ARFF regression dataset split by feature count."""
    df = arff.loadarff(path)
    df = pd.DataFrame(df[0]).dropna()
    return df.iloc[:, :n_features], df.iloc[:, n_features:]


def _load_real_experiment_data(data: str):
    """Return (X, y, model) for one supported real-data experiment."""
    if data == "stock":
        stock_portfolio_performance = _fetch_ucirepo_dataset(390)
        X = stock_portfolio_performance.data.features
        y = stock_portfolio_performance.data.targets
        X = X.drop(columns=y.columns)
        y = y.map(lambda x: float(x.strip("%")) / 100 if isinstance(x, str) and "%" in x else x)
        return X, y, MultiTaskLasso(alpha=0.0001)

    if data == "rf1":
        X, y = _load_arff_dataset("real_exps/data/rf1.arff", n_features=64)
        return X, y, _random_forest_model()

    if data == "rf2":
        X, y = _load_arff_dataset("real_exps/data/rf2.arff", n_features=576)
        return X, y, _random_forest_model()

    if data == "scm1d":
        X, y = _load_arff_dataset("real_exps/data/scm1d.arff", n_features=280)
        return X, y, _random_forest_model()

    if data == "scm20d":
        X, y = _load_arff_dataset("real_exps/data/scm20d.arff", n_features=61)
        return X, y, _random_forest_model()

    if data == "student":
        student_performance = _fetch_ucirepo_dataset(320)
        X = student_performance.data.features
        y = student_performance.data.targets
        categorical_cols = X.select_dtypes(include="object").columns.tolist()
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ],
            remainder="passthrough",
        )
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", _random_forest_model(
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=1.0,
            )),
        ])
        return X, y, model

    if data == "air":
        air_quality = _fetch_ucirepo_dataset(360)
        target_cols = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]
        df = air_quality.data.features.drop(columns=["Date", "Time", "NMHC(GT)"])
        feature_cols = df.columns.difference(target_cols)
        df[feature_cols] = df[feature_cols].replace(-200, np.nan)
        df[feature_cols] = SimpleImputer(strategy="mean").fit_transform(df[feature_cols])
        df = df[(df[target_cols] != -200).all(axis=1)]
        return df.drop(columns=target_cols), df[target_cols], _random_forest_model()

    if data == "crime":
        communities_and_crime = _fetch_ucirepo_dataset(211)
        X = communities_and_crime.data.features.drop(columns="State")
        X = X.loc[:, X.isna().mean() < 0.3]
        y = communities_and_crime.data.targets.dropna()
        X = X.loc[y.index]
        return X, y, _random_forest_model()

    if data == "energy":
        energy_efficiency = _fetch_ucirepo_dataset(242)
        X = energy_efficiency.data.features
        y = energy_efficiency.data.targets
        return X, y, _random_forest_model(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=77,
        )

    raise ValueError(f"Unknown real dataset: {data}. Expected one of {sorted(REAL_DATASETS)}.")



def run_real_experiments(data, num_splits, alpha=0.1, cal_size=0.2, test_size=0.2):
    """Run all baseline methods on one real dataset over repeated splits."""
    if data not in REAL_DATASETS:
        raise ValueError(f"Unknown real dataset: {data}. Expected one of {sorted(REAL_DATASETS)}.")
    if num_splits <= 0:
        raise ValueError("num_splits must be positive.")

    methods = [
        "TSCP",
        "TSCP-S",
        "TSCP-GWC",
        "Point CHR",
        "Emp. copula",
        "Unscaled Max",
        "Bonferroni",
    ]

    X, y, model = _load_real_experiment_data(data)
    results = np.zeros((len(methods), 4, num_splits))

    for i in range(num_splits):
        X_train, X_cal_test, y_train, y_cal_test = train_test_split(
            X,
            y,
            test_size=test_size + cal_size,
            random_state=stable_hash(i),
        )
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_cal_test,
            y_cal_test,
            test_size=test_size / (cal_size + test_size),
            random_state=stable_hash(i),
        )

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        prediction_cal = model.predict(X_cal)
        scores_cal = np.asarray(np.abs(prediction_cal - y_cal), dtype=np.float64)
        prediction_test = model.predict(X_test)
        scores_test = np.asarray(np.abs(prediction_test - y_test), dtype=np.float64)

        # Run the methods
        for index, method in enumerate(methods):

            start = time.time()
            prediction = function_choice(scores_cal, alpha, method)
            results[index][0][i] = time.time()-start
            results[index][1][i] = check_coverage_rate(scores_test, prediction)
            results[index][2][i] = prediction.volume()
            results[index][3][i] = _safe_log10(prediction.volume())
        
    output = []
    for index, method in enumerate(methods):
        row = [
            method,
            scores_cal.shape,
            scores_test.shape,
            np.mean(results[index][1]),
            np.std(results[index][1], ddof=1),
            np.mean(results[index][2]),
            np.std(results[index][2], ddof=1),
            np.mean(results[index][3]),
            np.std(results[index][3], ddof=1),
            np.mean(results[index][0]),
        ]
        output.append(row)
        
    columns = [
        "Methods",
        "cal_size",
        "test_size",
        "test_coverage_avg",
        "test_coverage_1std",
        "coverage_vol",
        "coverage_vol_1std",
        "coverage_vol_log",
        "coverage_vol_log_1std",
        "runtime_avg",
    ]

    return pd.DataFrame(output, columns=columns)

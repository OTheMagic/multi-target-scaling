"""Reusable experiment computations; imports do not load data or fit models."""

from .generation import (
    make_multitarget_regression, make_multitarget_regression_dependent_noise,
    make_partial_heteroskedastic_regression, make_contaminated_regression,
    draw_matched_regression, draw_fitted_interval_stress_regression,
    draw_regression_splits,
)
from .absolute import (
    fit_point_model, predict_absolute_residuals, apply_absolute_method,
    evaluate_absolute_regions,
)
from .quantile import (
    fit_quantile_models, predict_quantile_intervals, quantile_scores,
    fit_residual_quantile_intervals, predict_residual_quantile_intervals,
    quantile_diagnostics, apply_quantile_method, evaluate_quantile_predictions,
)
from .real import (
    load_arff_dataset, prepare_real_data, split_real_data, make_real_estimator,
)

__all__ = [
    "make_multitarget_regression", "make_multitarget_regression_dependent_noise",
    "make_partial_heteroskedastic_regression", "make_contaminated_regression",
    "draw_matched_regression", "draw_fitted_interval_stress_regression",
    "draw_regression_splits", "fit_point_model", "predict_absolute_residuals",
    "apply_absolute_method", "evaluate_absolute_regions", "fit_quantile_models",
    "predict_quantile_intervals", "quantile_scores", "fit_residual_quantile_intervals",
    "predict_residual_quantile_intervals", "quantile_diagnostics",
    "apply_quantile_method", "evaluate_quantile_predictions", "load_arff_dataset",
    "prepare_real_data", "split_real_data", "make_real_estimator",
]

"""Point fitting, absolute residuals, reviewed methods, and outcome evaluation.

Sources: synthetic_absolute.py:draw_data/compute_trial, utility/exps.py's
point fitting, and envelope_method/experiments.py:metric. These sources are
provenance only; none is imported. All randomness is explicit and local.
"""

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from utility import methods
from ._evaluation import summarize_predictions


def _targets(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
        raise ValueError(f"{name} must have finite nonempty shape (n,d)")
    return values


def fit_point_model(X_train, y_train, estimator=None, *, model_params=None):
    """Fit a fresh estimator on explicit training data only; return the model.

    X_train is (n,p), including DataFrames accepted by preprocessing pipelines;
    y_train is (n,d), or (n,) for one target. Default: ordinary least squares.
    A supplied sklearn estimator is cloned before fitting. Its random_state
    belongs in that estimator or model_params, as do all nondefault settings.
    """
    y_train = _targets(y_train, "y_train")
    model = LinearRegression() if estimator is None else clone(estimator)
    if model_params:
        model.set_params(**model_params)
    model.fit(X_train, y_train)
    return model


def predict_absolute_residuals(model, X, y):
    """Return (predictions, abs(y-predictions)), both (n,d), without fitting."""
    y = _targets(y, "y")
    prediction = _targets(model.predict(X), "prediction")
    if prediction.shape != y.shape:
        raise ValueError("predictions and y must have the same shape")
    return prediction, np.abs(y - prediction)


def apply_absolute_method(scores_cal, *, method, alpha, n_test=1,
                          random_state=42, split_random_state=42,
                          coordinate_keys=None, scalar_keys=None, test_keys=None,
                          reference_dim=0, search="backward", mu=None, std=None):
    """Apply one reviewed method to independent test instances.

    scores_cal is finite nonnegative (n,d). Returns a dictionary with method,
    adjustments/upper_closed (m,d), empty (m,), and objects (m native regions).
    All regions reuse calibration keys; every test row has independent keys.
    coordinate_keys is (n,d), scalar_keys is (n,). EC/Bonferroni test_keys is
    (m,d); all other methods use (m,). Omitted keys come from random_state
    (integer or Generator). Exact key collisions raise in the reviewed methods.

    Methods: unscaled, ec, point_chr, signed_gwc (alias gwc), envelope,
    bonferroni, naive, oracle, split_standardized. Split methods use the separate
    integer split_random_state in [0, 2**32), reusing the same split for every
    test row; None, booleans, and mutable RNG objects are rejected for that
    parameter. Oracle requires caller-supplied independent mu/std. Alpha is
    always explicit. All regions use lower=0; Point CHR retains its fixed
    nonnegative-score domain. No algorithms are implemented here.
    """
    scores = _targets(scores_cal, "scores_cal")
    if (scores < 0).any():
        raise ValueError("absolute calibration scores must be nonnegative")
    if not isinstance(n_test, (int, np.integer)) or n_test < 1:
        raise ValueError("n_test must be a positive integer")
    method = method.lower()
    if method == "gwc":
        method = "signed_gwc"
    functions = {
        "unscaled": methods.unscaled_prediction,
        "ec": methods.empirical_copula_prediction,
        "point_chr": methods.data_spliting_CHR_prediction,
        "signed_gwc": methods.gwc_prediction,
        "bonferroni": methods.bonferroni_prediction,
        "naive": methods.naive_prediction,
        "oracle": methods.data_splitting_oracle_prediction,
        "split_standardized": methods.data_splitting_standardized_prediction,
    }
    if method not in functions and method != "envelope":
        raise ValueError(f"unknown absolute method: {method}")
    if method in ("point_chr", "split_standardized") and (
        isinstance(split_random_state, (bool, np.bool_))
        or not isinstance(split_random_state, (int, np.integer))
        or not 0 <= split_random_state < 2**32
    ):
        raise ValueError("split_random_state must be an integer in [0, 2**32)")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if method == "oracle" and (mu is None or std is None):
        raise ValueError("oracle requires independently supplied mu and std")
    n, d = scores.shape
    rng = np.random.default_rng(random_state)
    marginal = method in ("ec", "bonferroni")
    kwargs = {"alpha": alpha}
    if marginal or method == "point_chr":
        kwargs["coordinate_keys"] = (rng.random((n, d)) if coordinate_keys is None
                                     else np.asarray(coordinate_keys, dtype=float))
    if not marginal:
        kwargs["scalar_keys"] = (rng.random(n) if scalar_keys is None
                                 else np.asarray(scalar_keys, dtype=float))
    test_shape = (n_test, d) if marginal else (n_test,)
    test_keys = rng.random(test_shape) if test_keys is None else np.asarray(test_keys, float)
    if test_keys.shape != test_shape:
        raise ValueError(f"test_keys must have shape {test_shape}")
    if method in ("point_chr", "split_standardized"):
        kwargs["random_state"] = split_random_state
    else:
        kwargs["random_state"] = rng
    if method == "point_chr":
        kwargs["reference_dim"] = reference_dim
    if method == "oracle":
        kwargs.update(mu=mu, std=std)
    if method not in ("point_chr", "envelope"):
        kwargs["lower"] = 0.0
    objects = []
    if method == "envelope":
        calibration = methods.EnvelopeCalibration(scores, **kwargs)
        for key in test_keys:
            objects.append(calibration.predict(lower=0.0, search=search, test_key=key))
    else:
        function = functions[method]
        for key in test_keys:
            key_kw = {"test_keys": key} if marginal else {"test_key": key}
            objects.append(function(scores, **kwargs, **key_kw))
    return dict(method=method, adjustments=np.stack([obj.upper for obj in objects]),
                upper_closed=np.stack([obj.upper_closed for obj in objects]),
                empty=np.array([obj.empty for obj in objects], dtype=bool),
                objects=objects)


def evaluate_absolute_regions(prediction, scores_test, *, point_predictions=None):
    """Evaluate (m,d) absolute test scores against per-test native regions.

    prediction is apply_absolute_method's dictionary. Outcome side lengths are
    twice its score upper bounds. Returns per-row membership/lengths/volume and
    log10 volume, plus joint and coordinate averages. Optional point_predictions
    (m,d) adds outcome_lower/outcome_upper and boundary-closure arrays. Empty
    rows remain explicitly flagged; their numeric corners do not define a set.
    """
    scores = _targets(scores_test, "scores_test")
    if (scores < 0).any():
        raise ValueError("absolute test scores must be nonnegative")
    adjustments = np.asarray(prediction["adjustments"], dtype=float)
    closed = np.asarray(prediction["upper_closed"], dtype=bool)
    empty = np.asarray(prediction["empty"], dtype=bool)
    objects = prediction["objects"]
    if adjustments.shape != scores.shape or closed.shape != scores.shape:
        raise ValueError("prediction arrays must match scores_test")
    if empty.shape != (len(scores),) or len(objects) != len(scores):
        raise ValueError("one native region and empty flag are required per test row")
    lower = np.stack([obj.lower for obj in objects])
    coordinate = ((scores >= lower) &
                  ((scores < adjustments) | ((scores == adjustments) & closed)))
    joint = np.array([obj.contains(score) for obj, score in zip(objects, scores)], bool)
    lengths = np.maximum(2.0 * adjustments, 0.0)
    result = summarize_predictions(coordinate, lengths, empty,
                                   adjustments=adjustments, joint_covered=joint)
    if point_predictions is not None:
        centers = _targets(point_predictions, "point_predictions")
        if centers.shape != scores.shape:
            raise ValueError("point_predictions must match scores_test")
        result.update(outcome_lower=centers - adjustments,
                      outcome_upper=centers + adjustments,
                      outcome_lower_closed=closed.copy(), outcome_upper_closed=closed.copy())
    return result

"""Preparation and estimators for the retained real-data designs.

Extracted from ``utility/exps.py:1709-1861`` and the data preparation used by
``real_exps.ipynb``.  Inputs are explicit paths or caller-supplied data; this
module never fetches datasets.  Air imputation and crime feature selection
are intentionally fitted on training rows only, as approved during review.
Use :mod:`utility.experiments.absolute` for fitting, residuals, and evaluation.
"""

from numbers import Integral

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import MultiTaskLasso
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


__all__ = [
    "load_arff_dataset",
    "prepare_real_data",
    "split_real_data",
    "make_real_estimator",
]

_DATASETS = frozenset(
    {"stock", "rf1", "rf2", "scm1d", "scm20d", "student", "energy", "air", "crime"}
)
_AIR_TARGETS = ("CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)")


def _check_dataset(dataset):
    if dataset not in _DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {sorted(_DATASETS)}")


def _check_frame(frame, name):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.columns.has_duplicates or frame.index.has_duplicates:
        raise ValueError(f"{name} must have unique column names and row indices")
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column")


def load_arff_dataset(path, n_features):
    """Read an explicit ARFF path and return feature and target DataFrames.

    The source convention is preserved: drop rows missing any field, use the
    first ``n_features`` columns as ``X`` (n,p), and the remaining columns as
    ``y`` (n,d).  Retained feature counts are rf1=64, rf2=576, scm1d=280,
    and scm20d=61.  No dataset name is inferred from the path and no data are
    downloaded, cached, or written.  Original column order and retained row
    indices are preserved.  Source: ``utility/exps.py:_load_arff_dataset``.
    """
    if isinstance(n_features, bool) or not isinstance(n_features, Integral):
        raise TypeError("n_features must be an integer")
    frame = pd.DataFrame(arff.loadarff(path)[0]).dropna()
    if not 0 < n_features < frame.shape[1]:
        raise ValueError("n_features must leave at least one feature and one target")
    if frame.empty:
        raise ValueError("No complete rows remain in the ARFF dataset")
    return frame.iloc[:, :n_features].copy(), frame.iloc[:, n_features:].copy()


def prepare_real_data(dataset, features, targets=None):
    """Perform deterministic dataset cleanup without fitting preprocessing.

    ``features`` is a pandas DataFrame (n,p); ``targets`` is a DataFrame
    (n,d), or a Series for one target.  Return independent ``(X, y)``
    DataFrames with aligned row indices and preserved feature order.

    * stock: remove target-named feature columns and convert target strings
      containing percent signs to fractions.
    * air: supply the original feature frame containing the four targets
      CO(GT), NOx(GT), NO2(GT), C6H6(GT); omit ``targets``.  Remove Date,
      Time, NMHC(GT), replace feature -200 with NaN, and remove rows whose
      targets contain -200.  Target order is the order above.
    * crime: remove State; drop missing-target rows and align feature rows.
    * rf1, rf2, scm1d, scm20d: drop rows missing any feature or target,
      preserving the same complete-case rule as :func:`load_arff_dataset`.
    * student and energy: retain supplied frames.

    Air missing features remain missing until a training-fitted imputer is
    applied.  Crime column selection is deferred to a training-fitted
    selector.  These are approved deviations from the full-cohort fitting
    in ``utility/exps.py:_load_real_experiment_data``.  Crime's remaining
    missing features are not imputed.  Caller-owned frames are never mutated.
    """
    _check_dataset(dataset)
    _check_frame(features, "features")
    X = features.copy()
    if dataset == "air":
        if targets is not None:
            raise ValueError("For air, supply targets within features and omit targets")
        frame = X.drop(columns=["Date", "Time", "NMHC(GT)"])
        y = frame.loc[:, list(_AIR_TARGETS)].copy()
        X = frame.drop(columns=list(_AIR_TARGETS)).replace(-200, np.nan)
        keep = (y != -200).all(axis=1)
        X, y = X.loc[keep].copy(), y.loc[keep].copy()
    else:
        if isinstance(targets, pd.Series):
            targets = targets.to_frame()
        _check_frame(targets, "targets")
        y = targets.copy()
        if dataset == "crime":
            X = X.drop(columns="State")
            y = y.dropna()
            if not y.index.isin(X.index).all():
                raise ValueError("Every retained target row must exist in features")
            X = X.loc[y.index].copy()
        else:
            if not X.index.equals(y.index):
                raise ValueError("features and targets must have matching ordered row indices")
            if dataset == "stock":
                X = X.drop(columns=X.columns.intersection(y.columns, sort=False))
                for column in y.columns:
                    y[column] = y[column].map(
                        lambda value: float(value.strip("%")) / 100
                        if isinstance(value, str) and "%" in value else value
                    )
            elif dataset in {"rf1", "rf2", "scm1d", "scm20d"}:
                keep = X.notna().all(axis=1) & y.notna().all(axis=1)
                X, y = X.loc[keep].copy(), y.loc[keep].copy()
    if X.shape[0] == 0 or X.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("Preparation must leave rows, features, and targets")
    return X, y


def split_real_data(X, y, *, cal_size=0.05, test_size=0.2, random_state):
    """Return one train/calibration/test split and its positional row indices.

    ``X`` (n,p) and ``y`` (n,d) may be DataFrames or array-like; DataFrame
    indices must align when both are frames.  ``cal_size`` and ``test_size``
    are fractions of the entire cohort, strictly positive and summing to
    less than one.  Defaults reproduce the retained notebook's .75/.05/.20
    design.  ``random_state`` is an explicit integer seed or sklearn-compatible
    RandomState; None is rejected to avoid hidden global randomness.

    Return a dictionary with ``X_train``, ``y_train``, ``X_cal``, ``y_cal``,
    ``X_test``, ``y_test``, and integer arrays ``train_indices``, ``cal_indices``,
    ``test_indices``.  Indices refer to input row positions, not frame labels.
    DataFrames remain DataFrames; other inputs become NumPy arrays.

    The source's two shuffled sklearn splits are preserved: first hold out
    cal_size + test_size, then allocate test_size / (cal_size + test_size)
    of that holdout to testing, passing the same random_state each time.
    An integer therefore restarts the same seed for each split; a supplied
    mutable RandomState advances.  No preprocessing or estimator is fitted.
    Source: ``utility/exps.py:run_real_experiments``.  Replaying historical
    trial seeds is the caller's responsibility; no trial loop is provided.
    """
    X = X if isinstance(X, pd.DataFrame) else np.asarray(X)
    y = y if isinstance(y, pd.DataFrame) else np.asarray(y)
    if X.ndim != 2 or y.ndim != 2 or len(X) != len(y):
        raise ValueError("X and y must be 2D with the same number of rows")
    if X.shape[1] == 0 or y.shape[1] == 0:
        raise ValueError("X and y must each contain at least one column")
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.DataFrame):
        if not X.index.equals(y.index):
            raise ValueError("X and y must have matching ordered row indices")
    if not (np.isfinite(cal_size) and np.isfinite(test_size)
            and cal_size > 0 and test_size > 0 and cal_size + test_size < 1):
        raise ValueError("cal_size and test_size must be positive fractions summing to < 1")
    if random_state is None:
        raise ValueError("Supply an explicit random_state")
    train, holdout = train_test_split(
        np.arange(len(X)), test_size=cal_size + test_size, random_state=random_state
    )
    cal, test = train_test_split(
        holdout, test_size=test_size / (cal_size + test_size), random_state=random_state
    )
    result = {}
    for label, indices in (("train", train), ("cal", cal), ("test", test)):
        result[f"{label}_indices"] = indices
        for name, values in (("X", X), ("y", y)):
            result[f"{name}_{label}"] = (
                values.iloc[indices].copy() if isinstance(values, pd.DataFrame)
                else values[indices].copy()
            )
    return result


class _TrainingMissingnessSelector(TransformerMixin, BaseEstimator):
    """Keep crime features with training missingness strictly below threshold."""

    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def fit(self, X, y=None):
        _check_frame(X, "X")
        if not np.isfinite(self.threshold) or not 0 < self.threshold <= 1:
            raise ValueError("missingness_threshold must be in (0, 1]")
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.columns_ = X.columns[X.isna().mean() < self.threshold].tolist()
        if not self.columns_:
            raise ValueError("No features meet the training missingness threshold")
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        _check_frame(X, "X")
        if not np.array_equal(np.asarray(X.columns, dtype=object), self.feature_names_in_):
            raise ValueError("Feature columns and their order must match training")
        return X.loc[:, self.columns_].copy()

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "columns_")
        if input_features is not None and not np.array_equal(
            np.asarray(input_features, dtype=object), self.feature_names_in_
        ):
            raise ValueError("input_features must match training feature names")
        return np.asarray(self.columns_, dtype=object)


def _object_columns(X):
    _check_frame(X, "X")
    return X.select_dtypes(include="object").columns.tolist()


def make_real_estimator(
    dataset, *, categorical_columns=None, model_params=None, missingness_threshold=0.3
):
    """Construct an unfitted estimator with the retained dataset's settings.

    ``model_params`` maps explicit parameter overrides to the underlying
    regressor, including ``random_state`` and ``n_jobs``.  Defaults are:
    stock MultiTaskLasso(alpha=.0001); otherwise RandomForestRegressor with
    200 trees, depth 16, split minimum 5, leaf minimum 2, max_features='sqrt',
    bootstrap=True, random_state=42, n_jobs=-1.  Student and energy use depth
    None, split minimum 2, leaf minimum 1, max_features=1.0; energy seed is 77.

    Student returns a pipeline with OneHotEncoder(handle_unknown='ignore')
    and passthrough numeric columns.  ``categorical_columns`` may give explicit
    column names; otherwise object-dtype columns are selected when fitting
    on the training DataFrame, as in the source.  Air returns a mean-imputer
    pipeline.  Crime returns a cloneable selector pipeline keeping columns
    whose *training* missing fraction is strictly below ``missingness_threshold``.
    Train-only air/crime fitting is the approved correction to the source.
    Crime's remaining NaNs are passed to the forest without imputation; this
    requires a scikit-learn forest implementation supporting missing values.

    Prepare frames with :func:`prepare_real_data`, split before fitting, and
    fit only the returned estimator on training X (n_train,p), y (n_train,d).
    Its predict output follows sklearn's target shape.  Do not fit or refit
    preprocessing on calibration/test data.  Feature order is retained by
    air/crime transforms; student's category-first ordering matches its source.
    No training, loading, downloads, or writes occur here.  Source:
    ``utility/exps.py:_random_forest_model`` and ``_load_real_experiment_data``.
    """
    _check_dataset(dataset)
    if categorical_columns is not None and dataset != "student":
        raise ValueError("categorical_columns applies only to the student dataset")
    overrides = {} if model_params is None else dict(model_params)
    if dataset == "stock":
        return MultiTaskLasso(**{"alpha": 0.0001, **overrides})
    settings = dict(
        n_estimators=200, max_depth=16, min_samples_split=5, min_samples_leaf=2,
        max_features="sqrt", bootstrap=True, random_state=42, n_jobs=-1,
    )
    if dataset in {"student", "energy"}:
        settings.update(max_depth=None, min_samples_split=2, min_samples_leaf=1,
                        max_features=1.0)
    if dataset == "energy":
        settings["random_state"] = 77
    settings.update(overrides)
    regressor = RandomForestRegressor(**settings)
    if dataset == "student":
        columns = _object_columns if categorical_columns is None else list(categorical_columns)
        preprocessor = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore"), columns)], remainder="passthrough"
        )
    elif dataset == "air":
        preprocessor = SimpleImputer(strategy="mean")
    elif dataset == "crime":
        preprocessor = _TrainingMissingnessSelector(threshold=missingness_threshold)
    else:
        return regressor
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])

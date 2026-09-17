"""Small computational checks: python -B -m unittest utility.experiments.test_experiments.

No research data, experiments, plots, models, or intermediate arrays are saved.
Legacy sources are not imported or required by these reusable checks.
"""

import ast
import io
import subprocess
import sys
import unittest
from importlib.util import resolve_name
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from utility import methods
from . import (
    apply_absolute_method, apply_quantile_method, draw_regression_splits,
    draw_matched_regression, draw_fitted_interval_stress_regression,
    evaluate_absolute_regions, evaluate_quantile_predictions, fit_point_model,
    fit_quantile_models, fit_residual_quantile_intervals, load_arff_dataset,
    make_contaminated_regression, make_multitarget_regression,
    make_multitarget_regression_dependent_noise, make_partial_heteroskedastic_regression,
    make_real_estimator, predict_absolute_residuals, predict_quantile_intervals,
    predict_residual_quantile_intervals, prepare_real_data, quantile_diagnostics,
    quantile_scores, split_real_data,
)
from ._evaluation import summarize_predictions


class ExperimentChecks(unittest.TestCase):
    def test_import_is_passive_and_runtime_boundary(self):
        code = '''
import sys, os
import numpy, pandas, scipy, sklearn
from sklearn.linear_model import LinearRegression, QuantileRegressor, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
def forbidden(*args, **kwargs):
    raise AssertionError("import performed computation or data IO")
for cls in (LinearRegression, QuantileRegressor, MultiTaskLasso,
            RandomForestRegressor, GradientBoostingRegressor,
            TransformedTargetRegressor, ColumnTransformer, StandardScaler, SimpleImputer):
    cls.fit = forbidden
for name in ("load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt"):
    setattr(numpy, name, forbidden)
for name in ("read_csv", "read_pickle", "read_excel", "read_parquet"):
    setattr(pandas, name, forbidden)
numpy.random.default_rng = forbidden
def audit(event, args):
    if event in ("socket.connect", "subprocess.Popen"):
        forbidden()
    if event == "open":
        path, mode, flags = args
        if (isinstance(mode,str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            forbidden()
        if str(path).lower().endswith((".csv", ".tsv", ".npy", ".npz", ".arff", ".xlsx", ".pkl")):
            forbidden()
sys.addaudithook(audit)
import utility.experiments
allowed = {"utility", "utility.rectangle", "utility.signed_geometry",
           "utility.tie_breaking", "utility.conformal_utils"}
bad = [n for n in sys.modules if n.startswith("utility")
       and n not in allowed and not n.startswith(("utility.methods", "utility.experiments"))]
assert not bad, bad
'''
        result = subprocess.run([sys.executable, "-B", "-c", code],
                                cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_static_transitive_local_dependencies(self):
        root = Path(__file__).resolve().parents[2]
        exceptions = {"utility.rectangle", "utility.signed_geometry",
                      "utility.tie_breaking", "utility.conformal_utils"}
        queue, seen = ["utility.experiments"], set()
        while queue:
            module = queue.pop()
            if module in seen:
                continue
            seen.add(module)
            self.assertTrue(module == "utility" or module in exceptions or
                            module.startswith(("utility.experiments", "utility.methods")), module)
            path = root.joinpath(*module.split("."))
            package = path.is_dir()
            path = path / "__init__.py" if package else path.with_suffix(".py")
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imports = []
                if isinstance(node, ast.Import):
                    imports = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    target = node.module or ""
                    if node.level:
                        target = resolve_name("."*node.level + target,
                                              module if package else module.rsplit(".", 1)[0])
                    imports = [target]
                    if target == "utility":
                        imports.extend("utility." + a.name for a in node.names)
                for target in imports:
                    if target == "utility" or target.startswith("utility."):
                        if module.startswith("utility.experiments"):
                            self.assertNotIn(target, exceptions, "exception is indirect only")
                        queue.append(target)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ("exec", "eval", "__import__"))
        self.assertTrue(exceptions.issubset(seen))

    def test_generators_shapes_randomness_and_preserved_noise_rules(self):
        generators = (make_multitarget_regression, make_multitarget_regression_dependent_noise,
                      make_partial_heteroskedastic_regression, make_contaminated_regression)
        for generate in generators:
            for n in (7, 13):
                kwargs = dict(n_samples=n, n_features=2, n_targets=3, n_informative=2,
                              random_state=9, noise_list=[3, 2, 1])
                x, y, coef = generate(**kwargs)
                self.assertEqual((x.shape, y.shape), ((n, 2), (n, 3)))
                for a, b in zip((x, y, coef), generate(**kwargs)):
                    assert_array_equal(a, b)
        for law in ("cauchy", "t"):
            a = make_multitarget_regression(7, 2, 2, 2, law, [1, 1], df=4)[1]
            b = make_multitarget_regression(7, 2, 2, 2, law, [3, 8], df=4)[1]
            assert_array_equal(a, b)
        x, y = make_multitarget_regression(7, 2, 2, 2, "gamma", coef=np.zeros((2, 2)))
        assert_array_equal(y[:, 0], np.zeros(7))

    def test_synthetic_splits_fixed_coefficients_and_nested_prefixes(self):
        kwargs = dict(coef=np.array([[1., 2.], [2., 1.]]), n_train=12, n_test=8,
                      train_random_state=3, calibration_random_state=4, test_random_state=5,
                      calibration_pool_size=15, n_validation=4, validation_random_state=6,
                      generator_kwargs=dict(n_features=2, n_targets=2, n_informative=2))
        a = draw_regression_splits(make_multitarget_regression, n_cal=5, **kwargs)
        b = draw_regression_splits(make_multitarget_regression, n_cal=9, **kwargs)
        assert_array_equal(a["X_cal"], b["X_cal"][:5])
        assert_array_equal(a["y_cal"], b["y_cal"][:5])
        assert_array_equal(a["y_train"], b["y_train"])
        self.assertEqual(a["X_validation"].shape, (4, 2))
        self.assertFalse(np.array_equal(a["X_train"][:5], a["X_cal"]))

    def test_matched_and_width_generators(self):
        coef = np.arange(1, 7).reshape(2, 3)/6
        for n in (6, 11):
            x, y = draw_matched_regression(n, coef, noise_design="normal_uniform", random_state=3)
            x2, y2 = draw_matched_regression(n, coef, noise_design="normal_uniform", random_state=3)
            assert_array_equal(x, x2)
            assert_array_equal(y, y2)
            self.assertTrue(np.all(np.abs((y-x@coef)[:, 1]) <= np.sqrt(3)))
            for design in ("gaussian_matched", "coordinate_width_misspecified",
                           "covariate_tail_mismatch", "nonlinear_mean_misspecified"):
                data = draw_fitted_interval_stress_regression(n, coef, noise_scales=[3, 2, 1],
                                                              design=design, random_state=8)
                self.assertEqual([v.shape for v in data], [(n, 2), (n, 3), (n, 3)])
                self.assertTrue((data[2] > 0).all())

    def test_point_fit_is_cloned_and_residuals_agree(self):
        rng = np.random.default_rng(8)
        for n in (9, 17):
            x = rng.normal(size=(n, 2))
            y = x @ np.array([[1., 2.], [3., 1.]]) + rng.normal(size=(n, 2))
            original = LinearRegression()
            model = fit_point_model(x, y, original)
            self.assertFalse(hasattr(original, "coef_"))
            reference = LinearRegression().fit(x, y)
            pred, scores = predict_absolute_residuals(model, x[:3], y[:3])
            assert_array_equal(pred, reference.predict(x[:3]))
            assert_array_equal(scores, np.abs(y[:3]-pred))

    def test_all_absolute_methods_and_reproducible_keys(self):
        scores = np.arange(1, 61, dtype=float).reshape(20, 3)
        for method in ("unscaled", "ec", "point_chr", "signed_gwc", "envelope",
                       "bonferroni", "naive", "oracle", "split_standardized"):
            kwargs = dict(mu=0., std=1.) if method == "oracle" else {}
            a = apply_absolute_method(scores, method=method, alpha=.2, n_test=2, **kwargs)
            b = apply_absolute_method(scores, method=method, alpha=.2, n_test=2, **kwargs)
            assert_array_equal(a["adjustments"], b["adjustments"])
            assert_array_equal(a["upper_closed"], b["upper_closed"])
            result = evaluate_absolute_regions(a, scores[:2], point_predictions=np.zeros((2, 3)))
            assert_allclose(result["outcome_volumes"], np.prod(2*a["adjustments"], axis=1))
            assert_allclose(result["log10_volumes"], np.log10(result["outcome_volumes"]))

    def test_absolute_ties_use_per_test_keys(self):
        scores = np.array([[0.], [0.], [2.]])
        keys = np.array([.1, .4, .7])
        prediction = apply_absolute_method(scores, method="unscaled", alpha=.5, n_test=2,
                                           scalar_keys=keys, test_keys=[.2, .8])
        result = evaluate_absolute_regions(prediction, np.zeros((2, 1)))
        assert_array_equal(result["covered"], [True, False])
        assert_array_equal(result["empty"], [False, True])
        for index, key in enumerate((.2, .8)):
            native = methods.unscaled_prediction(scores, alpha=.5, scalar_keys=keys, test_key=key)
            assert_array_equal(prediction["adjustments"][index], native.upper)
        assert_array_equal(result["outcome_volumes"], [0., 0.])
        assert_array_equal(result["log10_volumes"], [-np.inf, -np.inf])

    def test_metric_zero_infinite_and_log_conventions(self):
        result = summarize_predictions(np.ones((3, 2), bool),
                                       [[0., np.inf], [2., 5.], [np.inf, np.inf]],
                                       [False, False, True])
        assert_array_equal(result["outcome_volumes"], [0., 10., 0.])
        assert_array_equal(result["log10_volumes"], [-np.inf, 1., -np.inf])
        assert_array_equal(result["coordinate_lengths"], [2/3, np.inf])
        self.assertEqual(result["test_coverage"], 2/3)
        self.assertEqual(result["mean_log10_volume"], -np.inf)

    def test_volume_arithmetic_and_split_seed_regressions(self):
        for lengths, expected in (([1e-200, 1e-200, 1e200], 1e-200),
                                  ([1e-200, 1e-200, np.inf], np.inf),
                                  ([1e200, 1e200, 1e-200, 1e-200], 1.)):
            result = summarize_predictions(np.ones((1, len(lengths)), bool), [lengths], [False])
            assert_allclose(result["outcome_volumes"], [expected], rtol=1e-12, atol=0)
        result = summarize_predictions([[True], [True]], [[1e308], [1e308]], [False, False])
        self.assertEqual(result["outcome_volume"], 1e308)
        for method in ("point_chr", "split_standardized"):
            with self.assertRaises(ValueError):
                apply_absolute_method(np.arange(1, 25.).reshape(12, 2), method=method,
                                      alpha=.3, split_random_state=None)

    def test_quantile_fits_shapes_scaling_and_repeatability(self):
        rng = np.random.default_rng(12)
        for n in (12, 24):
            x, y = rng.normal(size=(n, 2)), rng.normal(size=(n, 2)) + [2., 40.]
            kwargs = dict(lower_quantile=.1, upper_quantile=.9, random_state=6,
                          model_params={"n_estimators": 3, "min_samples_leaf": 2},
                          standardize_targets=True, linear_init=True)
            fit = fit_quantile_models(x, y, **kwargs)
            pred = predict_quantile_intervals(fit, x[:4])
            same = predict_quantile_intervals(fit_quantile_models(x, y, **kwargs), x[:4])
            assert_array_equal(pred["lower"], same["lower"])
            self.assertEqual(pred["upper"].shape, (4, 2))
            assert_allclose(fit["lower_models"][0].transformer_.mean_, [y[:, 0].mean()])
        fitted = fit_quantile_models(x, y, lower_quantile=.2, upper_quantile=.8,
                                     model="linear_quantile")
        self.assertEqual(predict_quantile_intervals(fitted, x[:2])["lower"].shape, (2, 2))

    def test_quantile_signed_scores_crossings_and_diagnostics(self):
        class Constant:
            def __init__(self, value):
                self.value = value
            def predict(self, X):
                return np.full(len(X), self.value)
        pred = predict_quantile_intervals({"lower_models": [Constant(2)],
                                           "upper_models": [Constant(-2)]}, np.ones((2, 1)))
        self.assertEqual(pred["crossing_rate"], 1.)
        scores, lengths = quantile_scores([[0.], [3.]], pred["lower"], pred["upper"])
        assert_array_equal(scores, [[-2.], [1.]])
        assert_array_equal(lengths, [[4.], [4.]])
        diagnostics = quantile_diagnostics([[0.], [3.]], pred["raw_lower"], pred["raw_upper"],
                                           lower_quantile=.1, upper_quantile=.9)
        self.assertEqual(diagnostics["crossing_rate"], 1.)
        assert_array_equal(diagnostics["coordinate_base_coverage"], [.5])

    def test_residual_width_learner_matches_training_formula(self):
        rng = np.random.default_rng(10)
        x, y = rng.normal(size=(18, 2)), rng.normal(size=(18, 2))
        widths = np.exp(rng.normal(size=(18, 2)))
        fit = fit_residual_quantile_intervals(x, y, widths, base_alpha=.2,
                                              coordinate_factors=[.5, 2.])
        expected = np.quantile(np.abs(y-fit["center_model"].predict(x))/widths, .8, axis=0)*[.5, 2.]
        assert_array_equal(fit["width_multiplier"], expected)
        pred = predict_residual_quantile_intervals(fit, x[:3], widths[:3])
        assert_allclose(pred["upper"]-pred["lower"], 2*widths[:3]*expected)

    def test_quantile_native_methods_and_outcome_inversion(self):
        scores = np.array([[-2., -1.], [-1., 0.], [0., 1.], [1., 2.], [2., 3.]])
        cal_lengths, test_lengths = np.full_like(scores, 6.), np.array([[2., 4.], [4., 2.]])
        keys, tests = [.1, .2, .3, .4, .5], [.15, .8]
        for method in ("base", "cqhr", "gwc", "envelope"):
            pred = apply_quantile_method(scores, cal_lengths, test_lengths, method=method,
                                         alpha=.4, scalar_keys=keys, test_keys=tests)
            result = evaluate_quantile_predictions(pred, np.zeros((2, 2)),
                                                   base_lower=-test_lengths/2, base_upper=test_lengths/2)
            self.assertEqual(result["outcome_lengths"].shape, (2, 2))
            assert_allclose(result["outcome_upper"]-result["outcome_lower"],
                            test_lengths+2*pred["adjustments"])
            if method == "cqhr":
                native = methods.cqhr_adjustments(scores, cal_lengths, test_lengths, alpha=.4,
                                                  scalar_keys=keys, test_keys=tests)
                assert_array_equal(pred["adjustments"], native.adjustments)
                assert_array_equal(result["covered"], native.contains(np.zeros((2, 2))) & ~pred["empty"])
            elif method in ("gwc", "envelope"):
                function = methods.gwc_prediction if method == "gwc" else methods.envelope_prediction
                for i in range(2):
                    native = function(scores, alpha=.4, lower=-test_lengths[i]/2,
                                      scalar_keys=keys, test_key=tests[i])
                    assert_array_equal(pred["adjustments"][i], native.upper)

    def test_cqhr_shrinkage_empty_and_open_singleton(self):
        scores = np.array([[-2.], [-1.], [0.]])
        pred = apply_quantile_method(scores, np.full((3, 1), 4.), np.array([[1.], [2.], [2.]]),
                                     method="cqhr", alpha=.5, scalar_keys=[.1, .4, .7],
                                     test_keys=[.2, .2, .8])
        result = evaluate_quantile_predictions(pred, [[-.5], [-1.], [-1.]])
        assert_array_equal(result["empty"], [True, False, True])
        assert_array_equal(result["covered"], [False, True, False])
        assert_array_equal(result["outcome_volumes"], np.zeros(3))
        assert_array_equal(result["log10_volumes"], np.full(3, -np.inf))

    def test_signed_unscaled_native_rectangle_and_attainable_domain(self):
        scores = np.array([[-2., -3.], [-1., -2.], [0., -1.]])
        lengths = np.array([[1., 4.], [2., 2.], [2., 2.], [4., 6.]])
        kwargs = dict(method="unscaled", alpha=.5, scalar_keys=[.1, .4, .7],
                      test_keys=[.2, .2, .8, .2])
        pred = apply_quantile_method(scores, np.full_like(scores, 6.), lengths, **kwargs)
        assert_array_equal(pred["adjustments"], np.full((4, 2), -1.))
        self.assertIsInstance(pred["objects"], list)
        for i, key in enumerate(kwargs["test_keys"]):
            native = methods.unscaled_prediction(
                scores, alpha=.5, lower=-lengths[i]/2,
                scalar_keys=kwargs["scalar_keys"], test_key=key,
            )
            self.assertEqual(type(pred["objects"][i]).__name__, "Rectangle")
            assert_array_equal(pred["objects"][i].lower, -lengths[i]/2)
            assert_array_equal(pred["adjustments"][i], native.upper)
            assert_array_equal(pred["upper_closed"][i], native.upper_closed)
            self.assertEqual(pred["empty"][i], native.empty)
        result = evaluate_quantile_predictions(pred, -lengths/2,
                                               base_lower=-lengths/2, base_upper=lengths/2)
        assert_array_equal(result["empty"], [True, False, True, False])
        assert_array_equal(result["covered"], [False, True, False, True])
        assert_array_equal(result["outcome_volumes"], [0., 0., 0., 8.])
        positive = scores + 4.
        identity = apply_quantile_method(positive, np.full_like(scores, 6.), lengths, **kwargs)
        for i, key in enumerate(kwargs["test_keys"]):
            native = methods.unscaled_prediction(positive, alpha=.5,
                                                  scalar_keys=kwargs["scalar_keys"], test_key=key)
            assert_array_equal(identity["adjustments"][i], native.upper)
            assert_array_equal(identity["upper_closed"][i], native.upper_closed)
        infinite = apply_quantile_method(scores, np.full_like(scores, 6.), lengths,
                                         method="unscaled", alpha=.01)
        self.assertTrue(np.isposinf(infinite["adjustments"]).all())
        self.assertFalse(infinite["empty"].any())

    def test_signed_marginal_regions_shrinkage_and_randomized_faces(self):
        scores = np.array([[-2.], [-1.], [0.]])
        lengths = np.array([[1.], [2.], [2.], [4.]])
        for name in ("ec", "bonferroni"):
            pred = apply_quantile_method(
                scores, np.full_like(scores, 4.), lengths, method=name, alpha=.5,
                coordinate_keys=[[.1], [.4], [.7]], test_keys=[[.2], [.2], [.8], [.2]],
            )
            assert_array_equal(pred["adjustments"], np.full((4, 1), -1.))
            result = evaluate_quantile_predictions(
                pred, -lengths/2, base_lower=-lengths/2, base_upper=lengths/2,
            )
            assert_array_equal(result["empty"], [True, False, True, False])
            assert_array_equal(result["covered"], [False, True, False, True])
            assert_array_equal(result["outcome_volumes"], [0., 0., 0., 2.])
            assert_array_equal(result["outcome_lower_closed"], pred["upper_closed"])
            assert_array_equal(result["outcome_upper_closed"], pred["upper_closed"])

        # Each coordinate keeps its own candidate key at a tied raw boundary.
        scores = np.full((3, 2), -1.)
        coordinate_keys = np.array([[.1, .2], [.4, .5], [.7, .8]])
        for name in ("ec", "bonferroni"):
            pred = apply_quantile_method(
                scores, np.full_like(scores, 4.), np.full((2, 2), 4.),
                method=name, alpha=.5, coordinate_keys=coordinate_keys,
                test_keys=[[.05, .95], [.95, .05]],
            )
            assert_array_equal(pred["upper_closed"], [[True, False], [False, True]])
            result = evaluate_quantile_predictions(pred, np.full((2, 2), -1.))
            assert_array_equal(result["coordinate_covered"], pred["upper_closed"])
            assert_array_equal(result["covered"], [False, False])

    def test_signed_rectangles_match_native_methods_and_reuse_keys(self):
        functions = {
            "unscaled": methods.unscaled_prediction,
            "ec": methods.empirical_copula_prediction,
            "bonferroni": methods.bonferroni_prediction,
            "naive": methods.naive_prediction,
            "oracle": methods.data_splitting_oracle_prediction,
            "split_standardized": methods.data_splitting_standardized_prediction,
        }
        coordinate_keys = np.linspace(.1, .8, 24).reshape(12, 2)
        scalar_keys = np.linspace(.1, .8, 12)
        lengths = np.array([[40., 60.], [50., 70.]])
        negative = np.column_stack((np.arange(-12., 0.), np.arange(-18., 0., 1.5)))
        for scores in (negative, negative + [7., 10.]):
            for name, function in functions.items():
                marginal = name in ("ec", "bonferroni")
                tests = np.array([[.05, .95], [.95, .05]]) if marginal else [.05, .95]
                options = dict(mu=[-4., -6.], std=[2., 3.]) if name == "oracle" else {}
                key_options = (dict(coordinate_keys=coordinate_keys) if marginal else
                               dict(scalar_keys=scalar_keys))
                pred = apply_quantile_method(
                    scores, np.full_like(scores, 40.), lengths, method=name, alpha=.5,
                    random_state=19, split_random_state=42, test_keys=tests,
                    **key_options, **options,
                )
                assert_array_equal(pred["coordinate_keys"] if marginal else pred["scalar_keys"],
                                   coordinate_keys if marginal else scalar_keys)
                for i, key in enumerate(tests):
                    test_option = {"test_keys": key} if marginal else {"test_key": key}
                    native = function(scores, alpha=.5, lower=-lengths[i]/2,
                                      random_state=42, **key_options, **test_option, **options)
                    self.assertEqual(type(pred["objects"][i]).__name__, "Rectangle")
                    assert_array_equal(pred["objects"][i].lower, -lengths[i]/2)
                    assert_array_equal(pred["adjustments"][i], native.upper)
                    assert_array_equal(pred["upper_closed"][i], native.upper_closed)
                    self.assertEqual(pred["empty"][i], native.empty)
                result = evaluate_quantile_predictions(
                    pred, pred["adjustments"], base_lower=-lengths/2, base_upper=lengths/2,
                )
                assert_array_equal(result["covered"], [
                    obj.contains(row) for obj, row in zip(pred["objects"], pred["adjustments"])
                ])
                assert_allclose(result["outcome_upper"]-result["outcome_lower"],
                                lengths+2*pred["adjustments"])
                if scores is negative:
                    self.assertTrue((pred["adjustments"] < 0).all(), name)

    def test_quantile_method_input_contracts_and_point_chr_exclusion(self):
        scores = np.arange(-12., 0.).reshape(6, 2)
        lengths_cal, lengths_test = np.full_like(scores, 30.), np.full((2, 2), 30.)
        with self.assertRaisesRegex(ValueError, "(?i)point.*chr|nonnegative|absolute"):
            apply_quantile_method(scores, lengths_cal, lengths_test, method="point_chr", alpha=.5)
        for name, options in (("split_standardized", {"split_random_state": None}),
                              ("oracle", {}), ("ec", {"test_keys": [.1, .2]}),
                              ("bonferroni", {"coordinate_keys": np.zeros((6, 1))})):
            with self.assertRaises(ValueError):
                apply_quantile_method(scores, lengths_cal, lengths_test,
                                      method=name, alpha=.5, **options)
        for name in ("unscaled", "bonferroni", "naive", "oracle", "split_standardized"):
            options = dict(mu=0., std=1.) if name == "oracle" else {}
            pred = apply_quantile_method(scores, lengths_cal, lengths_test,
                                         method=name, alpha=.01, **options)
            self.assertTrue(np.isposinf(pred["adjustments"]).all(), name)
            self.assertFalse(pred["empty"].any())
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            apply_absolute_method(scores, method="unscaled", alpha=.5)

    def test_real_loading_preparation_and_split_protocol(self):
        arff = io.StringIO("@RELATION tiny\n@ATTRIBUTE x NUMERIC\n@ATTRIBUTE y NUMERIC\n@DATA\n1,2\n?,3\n4,5\n")
        x, y = load_arff_dataset(arff, 1)
        assert_array_equal(x.to_numpy(), [[1.], [4.]])
        assert_array_equal(y.to_numpy(), [[2.], [5.]])
        for n in (20, 40):
            x, y = pd.DataFrame({"x": np.arange(n)}), pd.DataFrame({"y": np.arange(n)})
            result = split_real_data(x, y, random_state=7)
            train, remaining = train_test_split(np.arange(n), test_size=.25, random_state=7)
            cal, test = train_test_split(remaining, test_size=.8, random_state=7)
            for name, ids in (("train", train), ("cal", cal), ("test", test)):
                assert_array_equal(result[name+"_indices"], ids)
                assert_array_equal(result["X_"+name].index, ids)
        x = pd.DataFrame({"x": [1., 2.], "target": [100., 200.]})
        y = pd.DataFrame({"target": ["10%", "20%"]})
        prepared_x, prepared_y = prepare_real_data("stock", x, y)
        self.assertEqual(list(prepared_x.columns), ["x"])
        assert_allclose(prepared_y.to_numpy(), [[.1], [.2]])
        self.assertEqual(x.shape, (2, 2))

    def test_real_preprocessors_are_training_only(self):
        x = pd.DataFrame({"a": [1., 3., 5., np.nan], "b": [2., 4., 6., 8.]})
        y = pd.DataFrame({"y1": [1., 2., 3., 4.], "y2": [4., 3., 2., 1.]})
        original = make_real_estimator("air", model_params={"n_estimators": 2, "n_jobs": 1})
        fit = fit_point_model(x, y, original)
        self.assertFalse(hasattr(original["preprocessor"], "statistics_"))
        assert_array_equal(fit["preprocessor"].statistics_, [3., 5.])
        predict_absolute_residuals(fit, pd.DataFrame({"a": [np.nan], "b": [100000.]}), [[0., 0.]])
        assert_array_equal(fit["preprocessor"].statistics_, [3., 5.])
        crime_x = pd.DataFrame({"keep": [1., 2., 3., 4.], "drop": [np.nan, np.nan, 1., 2.]})
        crime = fit_point_model(crime_x, y, make_real_estimator("crime", model_params={"n_estimators": 2, "n_jobs": 1}))
        self.assertEqual(crime["preprocessor"].columns_, ["keep"])
        assert_array_equal(crime["preprocessor"].transform(pd.DataFrame({"keep": [np.nan], "drop": [9.]})).columns, ["keep"])
        student_x = pd.DataFrame({"group": ["a", "b", "a", "b"], "number": [1., 2., 3., 4.]})
        student = fit_point_model(student_x, y, make_real_estimator("student", model_params={"n_estimators": 2, "n_jobs": 1}))
        p, _ = predict_absolute_residuals(student, pd.DataFrame({"group": ["new"], "number": [8.]}), [[0., 0.]])
        self.assertEqual(p.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()

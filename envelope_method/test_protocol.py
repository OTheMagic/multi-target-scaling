"""Regression tests for fresh-per-trial sampling and tied mean-cell selection."""
import hashlib
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from utility import exps
from utility.data_generator import make_multitarget_regression
from utility.res_rescaled import mean_index_solver, standardized_prediction
from utility.envelope import envelope_prediction


def main():
    for fn in [exps.run_abs_res_synthetic_experiment, exps.run_cqr_synthetic_experiment]:
        assert 'redraw_train_test' not in inspect.signature(fn).parameters
        assert 'n_train_pool' not in inspect.signature(fn).parameters
        draws, fits = [], []
        def gen(**kwargs):
            result = make_multitarget_regression(**kwargs)
            if kwargs.get('coef') is not None:
                draws.append((kwargs['random_state'], hashlib.sha256(result[0].tobytes()).hexdigest()))
            return result
        class Model(LinearRegression):
            def fit(self, X, y):
                fits.append(hashlib.sha256(np.asarray(X).tobytes()).hexdigest())
                return super().fit(X, y)
        class Quantile(DummyRegressor):
            def fit(self, X, y):
                fits.append(hashlib.sha256(np.asarray(X).tobytes()).hexdigest())
                return super().fit(X, y)
        kw = dict(dim_list=[2], sample_list=[30], alpha_list=[.1], trials=3,
                  n_train=60, n_test=20, n_features=2, n_informative=2,
                  methods=['Unscaled', 'Envelope'], data_generator=gen)
        if fn == exps.run_abs_res_synthetic_experiment:
            kw['model_factory'] = Model
        else:
            kw.update(base_interval_alpha=.1,
                      quantile_model_factory=lambda q, seed: Quantile(strategy='quantile', quantile=q))
        result = fn(**kw)
        assert len(draws) == 6 and len(set(draws)) == 6
        assert len(set(fits)) == 3
        assert len(fits) == (3 if fn == exps.run_abs_res_synthetic_experiment else 12)
        assert result.trial_results.redraw_train_test.all()
        assert result.trial_results.n_train.eq(60).all()
        print(fn.__name__, 'fresh draws and fits verified')
    tied = np.array([[0., 0.], [0., 0.], [0., 1.], [4., 3.]])
    np.testing.assert_array_equal(mean_index_solver(tied), [3, 3])
    rng = np.random.default_rng(2026090901)
    for _ in range(500):
        scores = np.where(rng.random((30, 3)) < .85, 0, rng.exponential(size=(30, 3)))
        if np.any(scores.std(0) == 0):
            continue
        env, old = envelope_prediction(scores, .1), standardized_prediction(scores, .1)
        assert np.all(env.upper <= old.upper + 1e-8 * (1 + abs(old.upper)))
    print('500 tied-score trials audited; tests passed')
    for d in [1, 2, 3]:
        for seed in range(100):
            rng = np.random.default_rng(seed)
            capped = np.zeros((80, d))
            capped[-6:] = rng.exponential(size=(6, d))
            for search in ['backward', 'rank', 'exhaustive']:
                region = envelope_prediction(capped, .1, search=search)
                assert not region.empty
                assert region.contains(np.zeros(d)), (d, seed, search)
    print('900 exact-zero boundary checks passed without discarding accepted atoms')


if __name__ == '__main__':
    main()

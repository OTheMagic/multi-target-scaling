"""Bounded training pilot; never launches the formal experiment grid."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits
from utility.data_generator import make_multitarget_regression
from envelope_method.experiments import dump


def main():
    dest = ROOT / 'data/envelope_method/cqr10/timing_pilot'
    dest.mkdir(parents=True, exist_ok=True)
    kw = dict(n_features=5, n_informative=5, n_targets=10,
              noise_type='gaussian', noise_list=list(range(10, 0, -1)))
    _, _, coef = make_multitarget_regression(n_samples=1, random_state=91001, **kw)
    xt, yt = make_multitarget_regression(n_samples=12000, coef=coef, random_state=91002, **kw)
    xv, yv = make_multitarget_regression(n_samples=3000, coef=coef, random_state=91003, **kw)
    arrays = dict(X_train=xt, y_train=yt, X_validation=xv, y_validation=yv,
                  dgp_coef=np.asarray(coef), seeds=np.array([91001, 91002, 91003]))
    specs = {
        'legacy_gbr': dict(n_train=2400, n_estimators=100, max_depth=5,
                           min_samples_leaf=5, learning_rate=.05),
        'strong_gbr': dict(n_train=12000, n_estimators=800, max_depth=3,
                           min_samples_leaf=20, learning_rate=.05,
                           n_iter_no_change=60, validation_fraction=.2, tol=1e-5),
        'linear_init_gbr': dict(n_train=12000, n_estimators=800, max_depth=3,
                           min_samples_leaf=20, learning_rate=.05,
                           n_iter_no_change=60, validation_fraction=.2, tol=1e-5),
    }
    rows = []
    with threadpool_limits(limits=1):
        for name, spec in specs.items():
            for j, q in [(0, .05), (9, .95)]:
                params = dict(spec)
                n = params.pop('n_train')
                loc, scale = yt[:n, j].mean(), yt[:n, j].std()
                model = GradientBoostingRegressor(
                    loss='quantile', alpha=q, random_state=91004+j,
                    init=LinearRegression() if name == 'linear_init_gbr' else None,
                    **params)
                start = time.perf_counter()
                model.fit(xt[:n], (yt[:n, j]-loc)/scale)
                seconds = time.perf_counter()-start
                pred = loc+scale*model.predict(xv)
                oracle = xv @ coef[j] + kw['noise_list'][j]*norm.ppf(q)
                noise = kw['noise_list'][j]
                row = dict(model=name, coordinate=j, quantile=q, n_train=n,
                           fit_seconds=seconds, trees=int(model.n_estimators_),
                           validation_tail_rate=float(np.mean(yv[:, j] <= pred)),
                           normalized_oracle_rmse=float(np.sqrt(np.mean((pred-oracle)**2))/noise),
                           normalized_pinball=float(mean_pinball_loss(yv[:, j], pred, alpha=q)/noise),
                           oracle_pinball=float(mean_pinball_loss(yv[:, j], oracle, alpha=q)/noise))
                arrays[f'{name}_j{j}_prediction'] = pred
                rows.append(row)
                dump(dest/'models.json', dict(purpose='timing and training-quality pilot, not formal trials',
                     specs=specs, rows=rows, dgp=kw, seeds=[91001, 91002, 91003, 91004]))
                print(json.dumps(row), flush=True)
    np.savez_compressed(dest/'model_pilot.npz', **arrays)


if __name__ == '__main__':
    main()

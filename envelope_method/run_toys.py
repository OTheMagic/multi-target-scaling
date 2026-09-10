"""Replay saved toy generators and save observations, bounds, and trial metrics."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tmp'), str(ROOT / 'envelope_method')]
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from envelope_cqhr_toy import draw_residual_sample, draw_scales, equicorrelated_normals, Z90_NORMAL
from envelope_cqhr_misspecified_width import draw_sample
from envelope_boundary_probe import SCENARIOS
from envelope_boundary_stress import draw_scores
from utility.envelope import EnvelopeCalibration, envelope_prediction, score_at
from experiments import evaluate_cqr, evaluate_absolute, dump, metric

OUT = ROOT / 'data/envelope_method/results/toys'


def fitted_trial(rng, draw, n_cal, n_test, base_alpha=None, n_train=1000):
    """Fit centers and nominal interval widths using fresh training data only."""
    datasets = []
    for count in [n_train, n_cal, n_test]:
        raw = draw(count)
        residual, width = raw[:2] if isinstance(raw, tuple) else (raw, None)
        magnitude = residual if width is None else residual + width / 2
        d = residual.shape[1]
        X = rng.normal(size=(count, 3))
        coef = np.arange(1, 3*d+1, dtype=float).reshape(3, d) / (3*d)
        noise = raw[2] if isinstance(raw, tuple) and len(raw) == 3 else magnitude * rng.choice([-1., 1.], size=magnitude.shape)
        datasets.append((X, X @ coef + noise, width))
    xt, yt, wt = datasets[0]
    model = LinearRegression().fit(xt, yt)
    train_residual = abs(yt - model.predict(xt))
    multiplier = None if wt is None else np.quantile(train_residual / (wt / 2), 1-base_alpha, axis=0)
    arrays = dict(model_coef=model.coef_, model_intercept=model.intercept_)
    scores, lengths = [], []
    for name, (X, y, width) in zip(['train', 'cal', 'test'], datasets):
        arrays['X_'+name], arrays['y_'+name] = X, y
        if width is not None:
            arrays['width_covariate_'+name] = width
        if name != 'train':
            score = abs(y - model.predict(X))
            if width is not None:
                length = width * multiplier
                score -= length / 2
                lengths.append(length)
            scores.append(score)
    if multiplier is not None:
        arrays['width_multiplier'] = multiplier
    return tuple(scores + lengths), arrays


def gaussian_outcomes(rng, n, scenario):
    scales = draw_scales(rng, n, scenario['d'], scenario)
    noise = scales * equicorrelated_normals(rng, n, scenario['d'], scenario.get('rho', 0.))
    return abs(noise) - Z90_NORMAL*scales, 2*Z90_NORMAL*scales, noise


def save_study(name, config, draws, cqr=True):
    dest = OUT / name
    dest.mkdir(parents=True, exist_ok=True)
    config = dict(config, n_train=1000, protocol='fresh_training_calibration_testing_every_trial',
                  model='OLS center; training-residual quantile width multiplier for CQR')
    dump(dest / 'config.json', config)
    records = []
    for trial, (values, observations) in enumerate(draws):
        saved = dest / f'trial_{trial:04d}.json'
        if saved.exists():
            prior = json.loads(saved.read_text())
            if isinstance(prior, dict) and prior.get('version') == 3:
                records.extend(prior['records'])
                continue
        if cqr:
            cal, test, lc, lt = values
            rr, arrays = evaluate_cqr(cal, test, lc, lt, .1, [dict(name='capped', shift=0.)])
            arrays.update(base_lengths_cal=lc, base_lengths_test=lt)
        else:
            cal, test = values
            rr, arrays = evaluate_absolute(cal, test, .1)
        for row in rr:
            row.update(trial=trial, study=name, redraw_train_test=True, n_train=1000)
        arrays.update(observations)
        np.savez_compressed(dest / f'trial_{trial:04d}.npz', scores_cal=cal, scores_test=test, **arrays)
        dump(saved, dict(version=3, records=rr))
        records.extend(rr)
    df = pd.DataFrame(records)
    df.to_csv(dest / 'trials.csv', index=False)
    df.groupby('method')[['test_coverage','outcome_volume','runtime']].agg(['mean','std','count']).to_csv(dest / 'summary.csv')
    dump(dest / 'status.json', dict(status='complete', trials=df.trial.nunique()))
    print('COMPLETE toy', name, df.trial.nunique(), flush=True)


def cqr_toys():
    scenarios = json.loads((ROOT / 'envelope_method/toy_designs.json').read_text())
    scenarios.append(dict(name='2d_high_correlation', d=2, scale_kind='constant',
                          base_scales=[1.,1.], rho=.95, seed=2026090807))
    for scenario in scenarios:
        config = dict(scenario, n_cal=80, n_test=120, trials=50, base_alpha=.1, alpha=.1)
        def draws():
            rng = np.random.default_rng(scenario['seed'])
            for _ in range(50):
                yield fitted_trial(rng, lambda n: gaussian_outcomes(rng, n, scenario),
                                   80, 120, base_alpha=.1)
        save_study(scenario['name'], config, draws())
    def misspecified():
        rng = np.random.default_rng(2026091601)
        for _ in range(80):
            yield fitted_trial(rng, lambda n: draw_sample(rng, n, 2, tail_scales=[.8,1.4]),
                               80, 160, base_alpha=.1)
    save_study('misspecified_width', dict(seed=2026091601, trials=80, n_cal=80, n_test=160,
               base_alpha=.1, alpha=.1, source='tmp/envelope_cqhr_misspecified_width.py'), misspecified())


def positive_and_signed():
    cases = [('gamma_1_03', 'gamma', [1., .3]), ('gamma_1_1', 'gamma', [1., 1.]),
             ('exp_1_03', 'exp', [1.,.3]), ('lognormal_1_03','lognormal',[1.,.3])]
    for idx, (name, family, scales) in enumerate(cases):
        # The former positive-only run did not retain its generator or seed.
        # These are new, fully recorded replications of the reported families.
        config = dict(seed=2026090810+idx, n_cal=50, n_test=3000, trials=200,
                      family=family, scales=scales, gamma_shape=2.,
                      provenance='new replication; earlier transient generator was not saved')
        def draws():
            rng = np.random.default_rng(config['seed'])
            def draw(n):
                if family == 'gamma':
                    return rng.gamma(2., 1., (n, 2)) * scales
                if family == 'exp':
                    return rng.exponential(size=(n, 2)) * scales
                return rng.lognormal(0., .6, (n, 2)) * scales
            for _ in range(200):
                yield fitted_trial(rng, draw, 50, 3000)
        save_study(name, config, draws(), cqr=False)
    config = dict(seed=2026090820, n_cal=60, n_test=2500, trials=500, base_alpha=.02,
                  alpha=.1, inside_probability=.98, negative_beta=[2.,3.], positive_scale=.2,
                  provenance='new replication; original seed-6 generator was not saved')
    def draws():
        rng = np.random.default_rng(config['seed'])
        def draw(n):
            mask = rng.random((n, 2)) < .98
            return np.where(mask, -rng.beta(2.,3.,(n,2)), rng.exponential(.2,(n,2))), np.full((n,2),2.)
        for _ in range(500):
            yield fitted_trial(rng, draw, 60, 2500, base_alpha=.02)
    save_study('conservative_signed_2d', config, draws())


def boundary_record(cal, lower, kind, trial):
    model = EnvelopeCalibration(cal, .1)
    reg, sequences = model.predict(lower, search='exhaustive', return_cells=True)
    fast = model.predict(lower)
    np.testing.assert_array_equal(fast.upper, reg.upper)
    records = []
    for j, seq in enumerate(sequences):
        flags = np.array([c['survives'] for c in seq])
        b = model.mean[j] - model.scale[j] / np.sqrt(model.n - 1)
        for cell in seq:
            lo = cell['lo']
            if lo <= b or lo == cell['hi']:
                continue
            # Independently evaluate the proved rank-count characterization.
            from utility.envelope import interval_sup
            off = np.full(model.n, -np.inf)
            for ell in range(model.d):
                if ell != j:
                    off = np.maximum(off, interval_sup(cal[:,ell],model.mean[ell],model.scale[ell],model.n,
                                                      lower[ell],reg.gwc_upper[ell]))
            cutoff = score_at(lo, model.mean[j], model.scale[j], model.n, lo)
            pred = np.sum((cal[:,j] < lo) & (off < cutoff)) < model.rank
            # Exact ties at a calibration knot may move by floating-point roundoff.
            if pred != cell['survives']:
                assert np.isclose(cell['bound'], lo, rtol=1e-12, atol=1e-12)
        records.append(dict(kind=kind, trial=trial, coordinate=j+1, cells=len(seq),
                            prefix=not bool(np.any(np.diff(flags.astype(int))>0)),
                            rightmost_matches=True, backward_evaluations=fast.evaluations[j],
                            certified_range_boundary=bool(reg.upper[j] > b)))
    return records, sequences


def boundary_toys():
    dest = OUT / 'boundary'
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for scenario in SCENARIOS:
        rng = np.random.default_rng(scenario['seed'])
        all_cal, all_lower = [], []
        for trial in range(300):
            cal, _ = draw_residual_sample(rng,80,scenario['d'],scenario)
            _, lt = draw_residual_sample(rng,1,scenario['d'],scenario)
            rr, _ = boundary_record(cal,-lt[0]/2,scenario['name'],trial)
            records.extend(rr)
            all_cal.append(cal)
            all_lower.append(-lt[0]/2)
        np.savez_compressed(dest / (scenario['name']+'.npz'),calibration=all_cal,lower=all_lower)
        print('COMPLETE boundary',scenario['name'],flush=True)
    rng = np.random.default_rng(2026091501)
    for kind in ['signed_mixture','rare_huge_tail','positive_lognormal','two_cluster_signed']:
        all_cal, all_lower = [], []
        for trial in range(2000):
            cal = draw_scores(rng,30,2,kind)
            lower = np.full(2,min(-1.,float(cal.min())-.25))
            rr, _ = boundary_record(cal,lower,kind,trial)
            records.extend(rr)
            all_cal.append(cal)
            all_lower.append(lower)
        np.savez_compressed(dest/(kind+'.npz'),calibration=all_cal,lower=all_lower)
        print('COMPLETE boundary',kind,flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(dest/'trials.csv',index=False)
    frame.groupby('kind').agg(coordinates=('trial','size'),prefix_fraction=('prefix','mean'),
               rightmost_agreement=('rightmost_matches','mean'),
               mean_backward_evaluations=('backward_evaluations','mean'),
               certified_range_fraction=('certified_range_boundary','mean')).to_csv(dest/'summary.csv')
    dump(dest/'status.json',dict(status='complete',splits=9200,coordinate_sequences=len(records)))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    task = sys.argv[1] if len(sys.argv)>1 else 'cqr'
    {'cqr':cqr_toys,'positive':positive_and_signed,'boundary':boundary_toys}[task]()

"""Checkpointed experiment replay from saved configurations, preserving repetitions.

Run with --kind absolute or --kind cqr. Each unique simulation protocol has its
own directory, and completed trials are reused only with the same configuration.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import platform
from importlib.metadata import version as package_version
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from threadpoolctl import threadpool_limits
from utility.data_generator import make_multitarget_regression, make_multitarget_regression_dependent_noise
from utility.envelope import EnvelopeCalibration, envelope_prediction
from utility.exps import (_stable_hash, _fit_coordinatewise_quantile_models,
                          _cqr_scores_and_base_lengths, _function_choice,
                          _fit_raw_cqr_baseline_adjustments)
from utility.res_rescaled import standardized_prediction
from utility.cqhr import cqhr_adjustments
from envelope_method.archive_storage import (STORAGE_MODES, get_storage_mode,
    records_sha256, require_storage, save_trial_archive, sha256_file, validate_archive)

OUT = ROOT / 'envelope_method/results'
PARAMS = dict(n_estimators=100, max_depth=5, learning_rate=.05, min_samples_leaf=5)
GENERATOR_KEYS = ['correlation', 'correlation_structure', 'df', 'contamination_fraction',
                  'contamination_multiplier', 'heteroskedastic_fraction', 'heteroskedastic_strength']


def dump(path, data):
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


def specs():
    """Frozen scenario design; execution always redraws and refits every trial."""
    items = json.loads((ROOT / 'envelope_method/settings.json').read_text())
    if not all(i['config'].get('redraw') is True for i in items):
        raise ValueError('Only fresh-per-trial configurations are supported.')
    return items


def execution_provenance():
    return dict(python=platform.python_version(),
        packages={name: package_version(name) for name in ('numpy', 'scipy', 'pandas', 'scikit-learn')},
        code_sha256={name: sha256_file(ROOT / name) for name in (
            'envelope_method/experiments.py', 'utility/data_generator.py',
            'utility/envelope.py', 'utility/res_rescaled.py', 'utility/exps.py', 'utility/cqhr.py')})


def generator(config):
    kw = config['generator_kwargs']
    if 'heteroskedastic_fraction' in kw:
        from reviewer_update.build_experiment_update import make_partial_heteroskedastic_regression
        return make_partial_heteroskedastic_regression
    if 'contamination_fraction' in kw:
        from reviewer_update.build_experiment_update import make_contaminated_regression
        return make_contaminated_regression
    if 'correlation' in kw:
        return make_multitarget_regression_dependent_noise
    return make_multitarget_regression


def metric(method, adjustment, test, base, seconds, empty=False):
    adjustment = np.broadcast_to(np.asarray(adjustment), test.shape)
    inside = (test <= adjustment) & ~np.asarray(empty)[..., None]
    lengths = np.maximum(base + 2 * adjustment, 0)
    if np.ndim(empty) == 0 and empty:
        lengths[:] = 0
    elif np.ndim(empty):
        lengths[np.asarray(empty)] = 0
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        volume = np.prod(lengths, axis=1)
        volume[np.any(lengths == 0, axis=1)] = 0
        log_volume = np.sum(np.log(lengths), axis=1)
    return dict(method=method, test_coverage=float(inside.all(axis=1).mean()),
                covered_count=int(inside.all(axis=1).sum()), n_test=len(test),
                outcome_volume=float(volume.mean()), mean_log_volume=float(log_volume.mean()),
                coordinate_lengths=lengths.mean(axis=0).tolist(),
                coordinate_coverage=inside.mean(axis=0).tolist(),
                mean_adjustments=adjustment.mean(axis=0).tolist(), runtime=seconds,
                empty_rate=float(np.mean(np.any(lengths < 0, axis=1) | np.asarray(empty))))


def evaluate_absolute(cal, test, alpha):
    records, arrays = [], {}
    for method in ['Envelope', 'TSCP_R', 'TSCP_GWC', 'Unscaled', 'Point_CHR', 'Empirical_copula']:
        start = time.perf_counter()
        reg = envelope_prediction(cal, alpha) if method == 'Envelope' else _function_choice(cal, alpha, method)
        seconds = time.perf_counter() - start
        records.append(metric(method, reg.upper, test, 0., seconds, getattr(reg, 'empty', False)))
        arrays[method] = reg.upper
        if method == 'Envelope':
            records[-1]['candidate_evaluations'] = sum(reg.evaluations)
            records[-1]['fallback'] = reg.fallback
    assert np.all(arrays['Envelope'] <= arrays['TSCP_R'] + 1e-8 * (1 + abs(arrays['TSCP_R'])))
    return records, arrays


def evaluate_cqr(cal, test, lengths_cal, lengths_test, alpha, transforms, *, envelope_search='backward'):
    records, arrays = [], {}
    start = time.perf_counter()
    trained = EnvelopeCalibration(cal, alpha)
    uppers, globals_, empties = [], [], []
    distinct_lengths, inverse = np.unique(lengths_test, axis=0, return_inverse=True)
    for length in distinct_lengths:
        reg = trained.predict(-length / 2, search=envelope_search)
        uppers.append(reg.upper)
        globals_.append(reg.gwc_upper)
        empties.append(reg.empty)
    arrays['Envelope_signed'] = np.array(uppers)[inverse]
    arrays['Signed_GWC'] = np.array(globals_)[inverse]
    empties = np.array(empties)[inverse]
    records.append(metric('Envelope_signed', arrays['Envelope_signed'], test, lengths_test,
                          time.perf_counter() - start, np.array(empties)))
    records.append(metric('Signed_GWC', arrays['Signed_GWC'], test, lengths_test, np.nan,
                          np.any(arrays['Signed_GWC'] < -lengths_test / 2, axis=1)))
    start = time.perf_counter()
    adj, _, _ = cqhr_adjustments(cal, lengths_cal, lengths_test, alpha)
    arrays['CQHR'] = adj
    records.append(metric('CQHR', adj, test, lengths_test, time.perf_counter() - start,
                          np.any(lengths_test + 2 * adj < 0, axis=1)))
    records.append(metric('Base', np.zeros_like(cal[0]), test, lengths_test, 0.))
    baseline_records, baseline_arrays = evaluate_raw_cqr_baselines(cal, test, lengths_test, alpha)
    records.extend(baseline_records)
    arrays.update(baseline_arrays)
    for transform in transforms:
        name, shift = transform['name'], transform['shift']
        scores = np.maximum(cal, 0) if name == 'capped' else cal + shift
        for method in ['Envelope', 'TSCP_R', 'TSCP_GWC']:
            start = time.perf_counter()
            reg = envelope_prediction(scores, alpha, search=envelope_search) if method == 'Envelope' else standardized_prediction(scores, alpha, method='GWC' if method == 'TSCP_GWC' else 'LWC')
            adj = reg.upper - (shift if name == 'shifted' else 0)
            key = f'{method}_{name}_{shift:g}'
            arrays[key] = adj
            records.append(metric(key, adj, test, lengths_test, time.perf_counter() - start,
                                  np.any(lengths_test + 2 * adj < 0, axis=1)))
            records[-1]['fallback'] = getattr(reg, 'fallback', '')
        en, old = arrays[f'Envelope_{name}_{shift:g}'], arrays[f'TSCP_R_{name}_{shift:g}']
        if np.std(scores, axis=0).min() > 0 and scores.min() >= 0:
            assert np.all(en <= old + 1e-8 * (1 + abs(old)))
    return records, arrays


def evaluate_raw_cqr_baselines(cal, test, lengths_test, alpha):
    records, arrays = [], {}
    for method in ['Unscaled', 'Empirical_copula']:
        adjustment, seconds = _fit_raw_cqr_baseline_adjustments(
            method=method, raw_scores_cal=cal, alpha=alpha)
        arrays[method] = adjustment
        records.append(metric(method, adjustment, test, lengths_test, seconds,
                              np.any(lengths_test + 2 * adjustment < 0, axis=1)))
    return records, arrays


def run_spec(item, storage_mode=None, output_root=None):
    """Use one writer per configuration, including concurrent cluster invocations."""
    dest = Path(output_root or OUT) / item['config']['kind'] / item['id']
    dest.mkdir(parents=True, exist_ok=True)
    lock = dest / '.run_spec.lock'
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f'Configuration locked: {lock}. Confirm its worker stopped before removing a stale lock.') from exc
    try:
        return _run_spec(item, storage_mode, output_root)
    finally:
        lock.unlink(missing_ok=True)


def _run_spec(item, storage_mode=None, output_root=None):
    cfg, ident = item['config'], item['id']
    mode = storage_mode or item.get('storage_mode') or ('scores' if cfg['kind'] == 'absolute' else 'compact')
    if mode not in STORAGE_MODES:
        raise ValueError(f'Unknown storage mode: {mode!r}')
    dest = Path(output_root or OUT) / cfg['kind'] / ident
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'config.json').exists():
        existing = json.loads((dest / 'config.json').read_text())
        if existing['config'] != cfg or existing.get('transforms', []) != item.get('transforms', []):
            raise ValueError(f'Configuration mismatch at {dest}; choose a separate output root.')
    dump(dest / 'config.json', item)
    provenance = execution_provenance()
    gen = generator(cfg)
    kwargs = dict(n_features=cfg['n_features'], n_informative=cfg['n_features'],
                  n_targets=cfg['d'], noise_type=cfg['noise_type'], noise_list=cfg['noise_levels'],
                  **cfg['generator_kwargs'])
    total = cfg['n_train'] + cfg['n_test']
    _, _, coef = gen(n_samples=total, random_state=_stable_hash(cfg['d']), **kwargs)
    records = []
    for trial in range(item['trials']):
        saved = dest / f'trial_{trial:03d}.json'
        archive = dest / f'trial_{trial:03d}.npz'
        if saved.exists():
            prior = json.loads(saved.read_text())
            if prior.get('version') == 2:
                if prior.get('provenance') is not None and prior['provenance'] != provenance:
                    raise ValueError(f'Code/environment changed since {saved}. Use a separate output root to avoid mixing implementations.')
                require_storage(prior, mode)
                if get_storage_mode(prior) != 'compact':
                    validate_archive(archive, prior)
                records.extend(prior['records'])
                continue
        d, n = cfg['d'], cfg['n_cal']
        seed = _stable_hash(d, n, trial)
        fit_start = time.perf_counter()
        with threadpool_limits(limits=1), np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            xx, yy = gen(n_samples=total, random_state=_stable_hash(d, n, trial, 'train_test'), coef=coef, **kwargs)
            xt, xv, yt, yv = xx[:cfg['n_train']], xx[cfg['n_train']:], yy[:cfg['n_train']], yy[cfg['n_train']:]
            cal_seed = _stable_hash(d, n, trial, 'calibration')
            xc, yc = gen(n_samples=n, random_state=cal_seed, coef=coef, **kwargs)
            if cfg['kind'] == 'absolute':
                model = LinearRegression().fit(xt, yt)
                cal, test = np.abs(yc - model.predict(xc)), np.abs(yv - model.predict(xv))
                fit_time = time.perf_counter() - fit_start
                rr, arrays = evaluate_absolute(cal, test, cfg['alpha'])
                arrays.update(model_coef=model.coef_, model_intercept=model.intercept_)
            else:
                qseed = _stable_hash(d, n, trial, cfg['alpha'], 'cqr')
                lower, upper = _fit_coordinatewise_quantile_models(
                    xt, yt, cfg['base_alpha'] / 2, 1 - cfg['base_alpha'] / 2,
                    random_state=qseed, quantile_model_params=cfg.get('quantile_params', PARAMS), n_jobs=1)
                cal, lc = _cqr_scores_and_base_lengths(xc, yc, lower, upper)
                test, lt = _cqr_scores_and_base_lengths(xv, yv, lower, upper)
                fit_time = time.perf_counter() - fit_start
                rr, arrays = evaluate_cqr(cal, test, lc, lt, cfg['alpha'], item['transforms'])
                arrays.update(base_lengths_cal=lc, base_lengths_test=lt)
        arrays.update(X_train=xt, y_train=yt, X_cal=xc, y_cal=yc, X_test=xv, y_test=yv,
                      dgp_coef=np.asarray(coef))
        for row in rr:
            row.update(trial=trial, config_id=ident, alpha=cfg['alpha'], n_cal=n, n_dim=d,
                       fit_seconds=fit_time, calibration_seed=cal_seed, split_seed=seed,
                       training_seed=_stable_hash(d, n, trial, 'train_test'),
                       redraw_train_test=True, n_train=cfg['n_train'])
        storage = save_trial_archive(archive, dict(scores_cal=cal, scores_test=test, **arrays), mode)
        storage['records_sha256'] = records_sha256(rr)
        dump(saved, dict(version=2, records=rr, archive_sha256=storage['archive_sha256'],
                         storage=storage, provenance=provenance))
        records.extend(rr)
        if (trial + 1) % 25 == 0:
            dump(dest / 'status.json', dict(status='running', completed=trial + 1, expected=item['trials']))
    df = pd.DataFrame(records)
    df.to_csv(dest / 'trials.csv', index=False)
    scalar = ['test_coverage', 'outcome_volume', 'mean_log_volume', 'runtime', 'empty_rate']
    summary = df.groupby('method')[scalar].agg(['mean', 'std', 'median', 'count'])
    quantiles = df.groupby('method')[scalar].quantile([.05, .25, .75, .95], interpolation='higher').unstack()
    quantiles.columns = pd.MultiIndex.from_tuples(
        [(field, f'q{int(q * 100):02d}') for field, q in quantiles.columns])
    summary = summary.join(quantiles)
    summary.to_csv(dest / 'summary.csv')
    df.groupby('method').outcome_volume.agg(
        infinite_trials=lambda values: int(np.isinf(values).sum()),
        invalid_trials=lambda values: int(np.isnan(values).sum()),
        zero_trials=lambda values: int((values == 0).sum())).to_csv(dest / 'volume_states.csv')
    dump(dest / 'status.json', dict(status='complete', completed=item['trials'], expected=item['trials']))
    return dict(id=ident, config=cfg, trials=item['trials'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=['absolute', 'cqr', 'inventory'], default='inventory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--storage', choices=STORAGE_MODES,
                        help='Default: scores for absolute; compact for CQR. Full retains raw observations.')
    parser.add_argument('--out', type=Path, default=OUT, help='Result root (kind/config directories are added).')
    args = parser.parse_args()
    inventory = specs()
    args.out.mkdir(exist_ok=True, parents=True)
    dump(args.out / 'simulation_inventory.json', inventory)
    for kind in ['absolute', 'cqr']:
        items = [i for i in inventory if i['config']['kind'] == kind]
        print(kind, len(items), 'configurations;', sum(i['trials'] for i in items), 'trials', flush=True)
    if args.kind == 'inventory':
        return
    items = [i for i in inventory if i['config']['kind'] == args.kind]
    if args.limit:
        items = items[:args.limit]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_spec, item, args.storage, args.out): item for item in items}
        errors = []
        for i, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                result = future.result()
                print('COMPLETE', i, '/', len(items), result, flush=True)
            except Exception as exc:
                dest = args.out / args.kind / item['id']
                dest.mkdir(parents=True, exist_ok=True)
                dump(dest / 'error.json', dict(error=repr(exc), config=item))
                errors.append(dict(id=item['id'], error=repr(exc)))
                print('ERROR', item['id'], repr(exc), flush=True)
        dump(args.out / f'{args.kind}_run_status.json', dict(status='failed' if errors else 'complete',
             configurations=len(items), errors=errors))
    if errors:
        raise RuntimeError(errors)


if __name__ == '__main__':
    main()

"""Fresh-data ten-outcome CQR study, with per-trial auditable checkpoints.

Full LWC is deliberately unavailable. TSCP_R is the inexpensive old shortcut.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for variable in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(variable, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import norm
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_pinball_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from utility.data_generator import make_multitarget_regression
from envelope_method.experiments import dump, evaluate_cqr

HERE = Path(__file__).resolve().parent
DATA = ROOT / 'data/envelope_method/cqr10'
DEFAULT_OUT = DATA / 'results'
PROTOCOL = 'cqr10-fresh-v1'
STORAGE_LEVELS = ('compact', 'scores', 'full')
CHECKPOINT_VERSION = 2
MODEL_PARAMS = dict(n_estimators=800, max_depth=3, min_samples_leaf=20,
                    learning_rate=.05, n_iter_no_change=60,
                    validation_fraction=.2, tol=1e-5)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def implementation_hash():
    files = [Path(__file__), ROOT/'envelope_method/experiments.py', *sorted((ROOT/'utility').glob('*.py'))]
    return digest({str(p.relative_to(ROOT)): sha256(p) for p in files})


def versions():
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, joblib=joblib.__version__)


def make_config(n_features=5, n_cal=100, base_alpha=.1, *, shifts=()):
    # Keep the historical save_models field so existing scientific IDs/seeds do
    # not change. Artifact retention is now controlled by the separate storage
    # argument; full always saves models, compact/scores never do.
    return dict(protocol=PROTOCOL, d=10, n_features=n_features, n_train=12000,
                n_validation=3000, n_cal=n_cal, n_test=600,
                alpha=.1, base_alpha=base_alpha, noise_levels=list(range(10, 0, -1)),
                model='linear_init_gbr', model_params=dict(MODEL_PARAMS),
                envelope_search='rank', seed=20260909, save_models=True,
                transforms=[dict(name='capped', shift=0.)]+
                           [dict(name='shifted', shift=float(s)) for s in shifts],
                quality_max_normalized_rmse=.25, quality_max_tail_error=.04)


def validate_config(cfg):
    if cfg.get('protocol') != PROTOCOL or cfg.get('d') != 10:
        raise ValueError('This protocol is exclusively fresh-per-trial, ten-outcome CQR.')
    if cfg['model'] not in ('linear_init_gbr', 'gbr'):
        raise ValueError('Supported models: linear_init_gbr or gbr.')
    if cfg['envelope_search'] not in ('rank', 'backward'):
        raise ValueError('Only certified rank or backward envelope search is supported.')
    if not 0 < cfg['alpha'] < 1 or not 0 < cfg['base_alpha'] < 1:
        raise ValueError('Miscoverage levels must be in (0,1).')
    if len(cfg['noise_levels']) != 10 or min(cfg['noise_levels']) <= 0:
        raise ValueError('Ten positive Gaussian noise scales are required.')
    for key in ['n_features', 'n_train', 'n_validation', 'n_cal', 'n_test']:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if any(t['name'] not in ('capped', 'shifted') or t['shift'] < 0 for t in cfg['transforms']):
        raise ValueError('Only capped and nonnegative shifted-score comparisons are supported.')


def experiment_grid(include_large_calibration=False):
    """Extend every primary CQR configuration, preserving its repetitions/shifts."""
    originals = json.loads((ROOT/'envelope_method/settings.json').read_text())
    items = []
    for old in originals:
        c = old['config']
        if c['kind'] != 'cqr':
            continue
        if c['d'] != 3 or c['generator_kwargs'] or c['noise_type'].lower() != 'gaussian':
            raise ValueError('Review new source CQR designs before extending this grid.')
        shifts = [t['shift'] for t in old['transforms'] if t['name'] == 'shifted']
        cfg = make_config(c['n_features'], c['n_cal'], c['base_alpha'], shifts=shifts)
        items.append(dict(config=cfg, trials=old['trials'], source_id=old['id'],
                          sources=old['sources'], optional=False))
    if include_large_calibration:
        for p, trials in [(5, 100), (10, 30)]:
            for n in [500, 1000]:
                items.append(dict(config=make_config(p, n), trials=trials,
                                  source_id=None, sources=['optional high-calibration extension'], optional=True))
    return sorted(items, key=lambda x: (x['config']['n_features'], x['config']['base_alpha'], x['config']['n_cal']))


def trial_seeds(cfg, trial):
    ident = digest(cfg)
    seed = lambda *parts: int(digest(parts)[:8], 16)
    return dict(dgp=seed(cfg['seed'], cfg['n_features'], 'dgp'),
                **{s: seed(cfg['seed'], ident, trial, s) for s in ['train', 'validation', 'cal', 'test', 'fit']})


def draw_data(cfg, trial):
    seeds = trial_seeds(cfg, trial)
    kwargs = dict(n_features=cfg['n_features'], n_informative=cfg['n_features'],
                  n_targets=10, noise_list=cfg['noise_levels'], noise_type='gaussian')
    _, _, coef = make_multitarget_regression(n_samples=1, random_state=seeds['dgp'], **kwargs)
    arrays = dict(dgp_coef=np.asarray(coef))
    for name in ['train', 'validation', 'cal', 'test']:
        x, y = make_multitarget_regression(n_samples=cfg[f'n_{name}'], coef=coef,
                                          random_state=seeds[name], **kwargs)
        arrays[f'X_{name}'], arrays[f'y_{name}'] = x, y
    return arrays, seeds


def fit_models(cfg, arrays, fit_seed):
    pairs, diagnostics = [], []
    xt, yt = arrays['X_train'], arrays['y_train']
    xv, yv = arrays['X_validation'], arrays['y_validation']
    for j in range(10):
        pair = []
        for k, q in enumerate([cfg['base_alpha']/2, 1-cfg['base_alpha']/2]):
            base = GradientBoostingRegressor(
                loss='quantile', alpha=q, random_state=(fit_seed+2*j+k) % (2**32),
                init=LinearRegression() if cfg['model'] == 'linear_init_gbr' else None,
                **cfg['model_params'])
            model = TransformedTargetRegressor(regressor=base, transformer=StandardScaler())
            start = time.perf_counter()
            model.fit(xt, yt[:, j])
            seconds = time.perf_counter()-start
            pred = model.predict(xv)
            # Oracle quantities are diagnostics only, never estimator inputs.
            noise = cfg['noise_levels'][j]
            oracle = xv @ arrays['dgp_coef'][j] + noise*norm.ppf(q)
            error = float(np.sqrt(np.mean((pred-oracle)**2))/noise)
            tail = float(np.mean(yv[:, j] <= pred))
            diagnostics.append(dict(coordinate=j, quantile=q, fit_seconds=seconds,
                trees=int(model.regressor_.n_estimators_),
                hit_tree_cap=bool(model.regressor_.n_estimators_ == cfg['model_params']['n_estimators']),
                normalized_oracle_rmse=error, validation_tail_rate=tail,
                normalized_pinball=float(mean_pinball_loss(yv[:, j], pred, alpha=q)/noise),
                normalized_oracle_pinball=float(mean_pinball_loss(yv[:, j], oracle, alpha=q)/noise),
                quality_flag=bool(error > cfg['quality_max_normalized_rmse'] or
                                  abs(tail-q) > cfg['quality_max_tail_error'])))
            pair.append(model)
        pairs.append(pair)
    return pairs, diagnostics


def predict_scores(models, arrays):
    crossings = {}
    for split in ['validation', 'cal', 'test']:
        x, y = arrays[f'X_{split}'], arrays[f'y_{split}']
        raw_lo = np.column_stack([pair[0].predict(x) for pair in models])
        raw_hi = np.column_stack([pair[1].predict(x) for pair in models])
        lo, hi = np.minimum(raw_lo, raw_hi), np.maximum(raw_lo, raw_hi)
        arrays[f'quantile_lower_raw_{split}'] = raw_lo
        arrays[f'quantile_upper_raw_{split}'] = raw_hi
        arrays[f'scores_{split}'] = np.maximum(lo-y, y-hi)
        arrays[f'base_lengths_{split}'] = hi-lo
        crossings[split] = float(np.mean(raw_lo > raw_hi))
    return crossings


def validate_storage(storage):
    if storage not in STORAGE_LEVELS:
        raise ValueError(f'storage must be one of {STORAGE_LEVELS}.')
    return storage


def checkpoint_files(trial, storage, *, legacy_models=True):
    names = set()
    if storage != 'compact':
        names.add(f'trial_{trial:04d}.npz')
    if storage == 'full' and legacy_models:
        names.add(f'trial_{trial:04d}.joblib')
    return names


def retained_arrays(arrays, storage):
    validate_storage(storage)
    if storage == 'compact':
        return {}
    if storage == 'full':
        return arrays
    return {k: v for k, v in arrays.items()
            if k.startswith(('scores_', 'base_lengths_', 'bound__'))}


def array_fingerprints(arrays):
    """Compact provenance witnesses; hashes cannot reconstruct discarded data."""
    return {k: dict(shape=list(np.shape(v)), dtype=str(np.asarray(v).dtype),
                    sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
            for k, v in arrays.items()}


def verified_checkpoint(directory, cfg, trial, *, minimum_storage=None):
    directory = Path(directory)
    if minimum_storage is not None:
        validate_storage(minimum_storage)
    saved = directory/f'trial_{trial:04d}.json'
    if not saved.exists():
        return None
    prior = json.loads(saved.read_text())
    if prior.get('config_hash') != digest(cfg) or prior.get('trial') != trial or prior.get('status') != 'complete':
        raise ValueError(f'Invalid checkpoint: {saved}')
    legacy = 'checkpoint_version' not in prior
    if legacy:
        # Historical pilot archives stay verifiable without rewriting them.
        storage = 'full' if cfg.get('save_models', True) else 'scores'
        required = checkpoint_files(trial, storage)
    else:
        if prior['checkpoint_version'] != CHECKPOINT_VERSION:
            raise ValueError(f'Unsupported checkpoint version: {saved}')
        storage = validate_storage(prior.get('storage'))
        content = {k: v for k, v in prior.items() if k != 'checkpoint_sha256'}
        if digest(content) != prior.get('checkpoint_sha256'):
            raise ValueError(f'Checkpoint metadata checksum mismatch: {saved}')
        required = checkpoint_files(trial, storage)
        if not prior.get('records') or len(prior.get('quality', [])) != 2*cfg['d']:
            raise ValueError(f'Incomplete metrics or diagnostics: {saved}')
    if not isinstance(prior.get('files'), dict) or set(prior['files']) != required:
        raise ValueError(f'Checkpoint required files are missing or inconsistent with {storage}: {saved}')
    for name, expected in prior['files'].items():
        artifact = directory/name
        if not artifact.is_file():
            raise ValueError(f'Checkpoint required file missing: {artifact}')
        if artifact.stat().st_size != expected['bytes'] or sha256(artifact) != expected['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch: {directory/name}')
    if minimum_storage is not None and STORAGE_LEVELS.index(storage) < STORAGE_LEVELS.index(minimum_storage):
        raise ValueError(f'Checkpoint stores {storage}, requested {minimum_storage}. '
                         'Storage upgrades require a separate output directory and deterministic refitting; '
                         'existing trials are never silently replaced.')
    return prior


def run_trial(cfg, trial, out=DEFAULT_OUT, *, storage='compact'):
    validate_config(cfg)
    validate_storage(storage)
    if trial < 0:
        raise ValueError('Trial index must be nonnegative.')
    directory = Path(out)/digest(cfg)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory/'config.json'
    expected = dict(config=cfg, implementation_hash=implementation_hash(), versions=versions())
    if manifest.exists():
        if json.loads(manifest.read_text()) != expected:
            raise ValueError('Code/environment/config changed. Use a new output directory; do not mix cohorts.')
    else:
        dump(manifest, expected)
    prior = verified_checkpoint(directory, cfg, trial, minimum_storage=storage)
    if prior is not None:
        return prior
    unexpected = checkpoint_files(trial, 'full') - checkpoint_files(trial, storage)
    if any((directory/name).exists() for name in unexpected):
        raise ValueError('Unfinished richer artifacts exist. Resume at their storage tier or '
                         'use a separate output directory; compact mode never leaves hidden raw archives.')
    lock = directory/f'trial_{trial:04d}.lock'
    try:
        with lock.open('x') as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as exc:
        raise RuntimeError(f'Trial locked: {lock}. Remove only after confirming its worker is stopped.') from exc
    start = time.perf_counter()
    try:
        with threadpool_limits(limits=1):
            arrays, seeds = draw_data(cfg, trial)
            fit_start = time.perf_counter()
            models, quality = fit_models(cfg, arrays, seeds['fit'])
            fit_seconds = time.perf_counter()-fit_start
            predict_start = time.perf_counter()
            crossings = predict_scores(models, arrays)
            predict_seconds = time.perf_counter()-predict_start
            evaluate_start = time.perf_counter()
            records, bounds = evaluate_cqr(
                arrays['scores_cal'], arrays['scores_test'], arrays['base_lengths_cal'],
                arrays['base_lengths_test'], cfg['alpha'], cfg['transforms'],
                envelope_search=cfg['envelope_search'])
            evaluate_seconds = time.perf_counter()-evaluate_start
        arrays.update({f'bound__{name}': bound for name, bound in bounds.items()})
        flagged = any(row['quality_flag'] for row in quality)
        for row in records:
            row.update(trial=trial, config_id=directory.name, n_dim=10, n_features=cfg['n_features'],
                       n_cal=cfg['n_cal'], base_alpha=cfg['base_alpha'], alpha=cfg['alpha'],
                       n_train=cfg['n_train'], n_validation=cfg['n_validation'],
                       redraw_train_test=True, quality_flag=flagged,
                       infinite_volume=bool(np.isposinf(row['outcome_volume'])),
                       invalid_volume=bool(np.isnan(row['outcome_volume'])),
                       zero_volume=bool(row['outcome_volume'] == 0))
        save_start = time.perf_counter()
        fingerprints = array_fingerprints(arrays)
        files = {}
        payload = retained_arrays(arrays, storage)
        if payload:
            archive = directory/f'trial_{trial:04d}.npz'
            tmp = archive.with_suffix('.npz.tmp')
            with tmp.open('wb') as stream:
                np.savez_compressed(stream, **payload)
            os.replace(tmp, archive)
            files[archive.name] = dict(sha256=sha256(archive), bytes=archive.stat().st_size)
        if storage == 'full':
            model_file = directory/f'trial_{trial:04d}.joblib'
            tmp = model_file.with_suffix('.joblib.tmp')
            joblib.dump(dict(models=models, config=cfg, seeds=seeds), tmp, compress=3)
            os.replace(tmp, model_file)
            files[model_file.name] = dict(sha256=sha256(model_file), bytes=model_file.stat().st_size)
        result = dict(status='complete', protocol=PROTOCOL, trial=trial, config_hash=digest(cfg),
                      checkpoint_version=CHECKPOINT_VERSION, storage=storage,
                      implementation_hash=expected['implementation_hash'], versions=expected['versions'],
                      array_fingerprints=fingerprints,
                      seeds=seeds, files=files, fit_seconds=fit_seconds,
                      predict_seconds=predict_seconds, evaluate_seconds=evaluate_seconds,
                      save_seconds=time.perf_counter()-save_start, total_seconds=time.perf_counter()-start,
                      quality_flag=flagged, quality=quality, crossings=crossings, records=records)
        result['checkpoint_sha256'] = digest(result)
        dump(directory/f'trial_{trial:04d}.json', result)
        return result
    finally:
        lock.unlink(missing_ok=True)


def summarize_config(cfg, out=DEFAULT_OUT):
    directory = Path(out)/digest(cfg)[:16]
    rows, quality = [], []
    for p in sorted(directory.glob('trial_*.json')):
        result = json.loads(p.read_text())
        verified_checkpoint(directory, cfg, result['trial'])
        rows.extend(result['records'])
        quality.extend([dict(trial=result['trial'], **q) for q in result['quality']])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(directory/'trials.csv', index=False)
    pd.DataFrame(quality).to_csv(directory/'model_quality.csv', index=False)
    fields = ['test_coverage', 'outcome_volume', 'mean_log_volume', 'empty_rate']
    summary = df.groupby('method')[fields].agg(['mean', 'std', 'median', 'count'])
    quantiles = df.groupby('method')[fields].quantile([.05, .25, .75, .95], interpolation='higher').unstack()
    quantiles.columns = pd.MultiIndex.from_tuples(
        [(name, f'q{int(q*100):02d}') for name, q in quantiles.columns])
    summary = summary.join(quantiles)
    summary.to_csv(directory/'summary.csv')
    states = df.groupby('method').outcome_volume.agg(
        infinite_trials=lambda x: int(np.isposinf(x).sum()),
        invalid_trials=lambda x: int(np.isnan(x).sum()),
        zero_trials=lambda x: int((x == 0).sum()))
    states.to_csv(directory/'volume_states.csv')
    # Independent units are fitted trials, not individual test points.
    wide = df.pivot(index='trial', columns='method', values='outcome_volume')
    comparisons = []
    for other in ['CQHR', 'TSCP_R_capped_0', 'Envelope_capped_0', 'Signed_GWC']:
        if other not in wide or 'Envelope_signed' not in wide:
            continue
        a, b = wide['Envelope_signed'], wide[other]
        finite = np.isfinite(a) & np.isfinite(b) & (b > 0)
        reduction = 1-a[finite]/b[finite]
        comparisons.append(dict(comparator=other, all_trials=len(a), finite_pairs=int(finite.sum()),
            excluded_pairs=int((~finite).sum()), mean_paired_volume_reduction=float(reduction.mean()),
            paired_standard_error=float(reduction.std(ddof=1)/np.sqrt(len(reduction))) if len(reduction)>1 else None,
            ratio_of_finite_mean_volumes=float(a[finite].mean()/b[finite].mean()) if finite.any() else None))
    dump(directory/'paired_comparisons.json', comparisons)
    dump(directory/'status.json', dict(status='summarized', completed=int(df['trial'].nunique()),
         quality_flagged_trials=int(df.groupby('trial')['quality_flag'].any().sum())))
    return summary


def run_configuration(item, out=DEFAULT_OUT, *, storage='compact'):
    cfg = item['config']
    for trial in range(item['trials']):
        result = run_trial(cfg, trial, out, storage=storage)
        if (trial+1) % 10 == 0 or trial+1 == item['trials']:
            print(f'{digest(cfg)[:16]} {trial+1}/{item["trials"]}; '
                  f'last={result["total_seconds"]:.1f}s quality_flag={result["quality_flag"]}', flush=True)
    summarize_config(cfg, out)
    return digest(cfg)[:16]


def run_study(items, out=DEFAULT_OUT, workers=1, *, storage='compact'):
    """Parallelize configurations, never methods on independently refitted data."""
    if workers < 1 or not items:
        raise ValueError('Select configurations and at least one worker.')
    validate_storage(storage)
    ids = [digest(item['config']) for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate configurations must be deduplicated before execution.')
    Path(out).mkdir(parents=True, exist_ok=True)
    dump(Path(out)/'study_manifest.json', dict(items=items, workers=workers,
         requested_storage=storage, checkpoint_version=CHECKPOINT_VERSION,
         implementation_hash=implementation_hash(), versions=versions()))
    if workers == 1:
        return [run_configuration(item, out, storage=storage) for item in items]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(run_configuration, item, str(out), storage=storage): item for item in items}
        for future in as_completed(tasks):
            results.append(future.result())
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--large-calibration', action='store_true')
    parser.add_argument('--storage', choices=STORAGE_LEVELS, default='compact')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    items = experiment_grid(args.large_calibration)
    print(f'{len(items)} configurations, {sum(i["trials"] for i in items)} fresh fitted trials')
    print(f'Storage: {args.storage}; output: {args.out.resolve()}')
    if args.run:
        run_study(items, out=args.out, workers=args.workers, storage=args.storage)
    else:
        print('Dry run. Use the notebook or explicitly pass --run to start.')


if __name__ == '__main__':
    main()

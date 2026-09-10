"""Replay every cached real-data split with paired envelope/old/GWC diagnostics."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from utility.envelope import envelope_prediction
from utility.res_rescaled import standardized_prediction

OUT = ROOT / 'data/envelope_method/results/real'
CACHE = ROOT / 'data/reviewer_update/real_diagnostics/cache'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {}
    for folder in sorted(CACHE.iterdir()):
        if not folder.is_dir():
            continue
        files = sorted(folder.glob('split_*.npz'))
        if not files:
            continue
        dest = OUT / folder.name
        dest.mkdir(exist_ok=True)
        records, coords = [], []
        for i, path in enumerate(files):
            saved = dest / f'{path.stem}.json'
            if saved.exists():
                data = json.loads(saved.read_text())
                records.extend(data['trials'])
                coords.extend(data['coordinates'])
                continue
            with np.load(path, allow_pickle=False) as data:
                cal, test = data['scores_cal'], data['scores_test']
                trial, seed = int(data['trial']), int(data['split_seed'])
            rr, cc, bounds = [], [], {}
            for alpha in [.1, .3, .5, .7, .9]:
                with threadpool_limits(limits=1), np.errstate(divide='ignore', invalid='ignore'):
                    diag = {}
                    old = standardized_prediction(cal, alpha, diagnostics=diag)
                    times = []
                    for _ in range(7 if alpha == .1 else 1):
                        start = time.perf_counter()
                        env = envelope_prediction(cal, alpha)
                        times.append(time.perf_counter() - start)
                upper = env.upper
                assert env.empty or env.fallback or np.all(upper <= old.upper + 1e-8 * (1 + abs(old.upper)))
                for method, u, empty in [('Envelope', upper, env.empty), ('TSCP_R', old.upper, False),
                                          ('Signed_GWC', env.gwc_upper, np.any(env.gwc_upper < 0))]:
                    inside = (test <= u) & ~np.bool_(empty)
                    lengths = np.zeros(cal.shape[1]) if empty else 2 * u
                    volume = 0.0 if np.any(lengths == 0) else float(np.prod(lengths))
                    common = dict(dataset=folder.name, trial=trial, split_seed=seed, alpha=alpha,
                                  method=method, n_cal=len(cal), n_test=len(test), n_dim=cal.shape[1])
                    rr.append({**common, 'test_coverage': float(inside.all(axis=1).mean()),
                               'covered_count': int(inside.all(axis=1).sum()), 'outcome_volume': volume,
                               'runtime_median': float(np.median(times)) if method == 'Envelope' else None,
                               'empty': bool(empty), 'old_fallback': bool(diag['fallback']),
                               'envelope_fallback': env.fallback,
                               'candidate_evaluations': int(sum(env.evaluations))})
                    for j in range(cal.shape[1]):
                        cc.append({**common, 'coordinate': j + 1, 'upper': float(u[j]),
                                   'outcome_length': float(lengths[j]),
                                   'coordinate_coverage': float(inside[:, j].mean())})
                    bounds[f'{method}_{alpha}'] = u
                bounds[f'Envelope_times_{alpha}'] = np.array(times)
            np.savez_compressed(dest / path.name, **bounds)
            payload = dict(source=str(path.relative_to(ROOT)),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           trials=rr, coordinates=cc)
            saved.write_text(json.dumps(payload), encoding='utf-8')
            records.extend(rr)
            coords.extend(cc)
            if (i + 1) % 25 == 0:
                print(f'{folder.name}: {i+1}/{len(files)}', flush=True)
        frame = pd.DataFrame(records)
        frame.to_csv(dest / 'trials.csv', index=False)
        pd.DataFrame(coords).to_csv(dest / 'coordinates.csv', index=False)
        frame.groupby(['dataset', 'alpha', 'method']).agg(
            trials=('trial', 'size'), coverage=('test_coverage', 'mean'), coverage_sd=('test_coverage', 'std'),
            volume=('outcome_volume', 'mean'), runtime=('runtime_median', 'mean'),
            empty_rate=('empty', 'mean')).to_csv(dest / 'summary.csv')
        status[folder.name] = dict(status='complete', splits=len(files), alpha=[.1,.3,.5,.7,.9])
        (OUT / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(f'COMPLETE {folder.name}: {len(files)} splits', flush=True)


if __name__ == '__main__':
    main()

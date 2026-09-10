"""Six single-trial pilots plus calibration-size timings, outside formal results."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from utility.data_generator import make_multitarget_regression
from envelope_method.experiments import dump, evaluate_cqr
from envelope_method.cqr10.runner import HERE, DATA, digest, make_config, run_trial, versions


def main():
    out = DATA/'timing_pilot/full_trials'
    rows = []
    for p in [5, 10]:
        for base in [.1, .5, .9]:
            cfg = make_config(p, 100, base, shifts=[100, 300, 1000] if p == 5 and base == .5 else [])
            # Timing pilots have seeds disjoint from the formal study.
            cfg['seed'] = 2026090901
            result = run_trial(cfg, 0, out, storage='full')
            row = dict(n_features=p, base_alpha=base, config=cfg,
                       config_id=digest(cfg)[:16], **{k: result[k] for k in [
                           'fit_seconds', 'predict_seconds', 'evaluate_seconds', 'save_seconds',
                           'total_seconds', 'quality_flag', 'quality', 'crossings']},
                       files_bytes=sum(f['bytes'] for f in result['files'].values()),
                       method_times={r['method']: r['runtime'] for r in result['records']})
            row['calibration_timings'] = []
            directory = out/digest(cfg)[:16]
            with np.load(directory/'trial_0000.npz') as archive:
                arrays = {k: archive[k] for k in ['scores_test', 'base_lengths_test', 'dgp_coef']}
            if base == .1:
                import joblib
                models = joblib.load(directory/'trial_0000.joblib')['models']
                kwargs = dict(n_features=p, n_informative=p, n_targets=10,
                              noise_type='gaussian', noise_list=cfg['noise_levels'])
                # Reusing one fitted pilot isolates method costs. These are timing
                # measurements only, never empirical trials or coverage evidence.
                for n in [30, 50, 100, 200, 500, 1000]:
                    xc, yc = make_multitarget_regression(n_samples=n, coef=arrays['dgp_coef'],
                                                        random_state=910100+n+p, **kwargs)
                    with threadpool_limits(limits=1):
                        lo = np.column_stack([pair[0].predict(xc) for pair in models])
                        hi = np.column_stack([pair[1].predict(xc) for pair in models])
                        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                        scores = np.maximum(lo-yc, yc-hi)
                        start = time.perf_counter()
                        records, _ = evaluate_cqr(scores, arrays['scores_test'], hi-lo,
                            arrays['base_lengths_test'], .1, cfg['transforms'], envelope_search='rank')
                        timing = dict(n_cal=n, evaluate_seconds=time.perf_counter()-start,
                                      method_times={r['method']: r['runtime'] for r in records})
                    row['calibration_timings'].append(timing)
                    print(json.dumps(dict(n_features=p, base_alpha=base, **timing)), flush=True)
            rows.append(row)
            dump(DATA/'timing_pilot/study.json', dict(purpose='Runtime/quality pilots only; no formal study launched',
                 versions=versions(), logical_cpus=os.cpu_count(), rows=rows))
            print(json.dumps({k: v for k, v in row.items() if k not in ['quality', 'calibration_timings', 'config']}), flush=True)


if __name__ == '__main__':
    main()

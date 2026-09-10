"""Fresh fitted reruns of the small exploratory notebook configurations."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from experiments import ROOT, OUT, PARAMS, dump, run_spec


def main():
    items = []
    def add(kind, d, n, trials, source, n_train=800, n_test=200, **kwargs):
        cfg = dict(kind=kind, d=d, n_cal=n, alpha=.1, noise_type='Gaussian',
                   noise_levels=list(range(d, 0, -1)), n_features=10,
                   n_train=n_train, n_test=n_test, redraw=True, generator_kwargs={})
        cfg.update(kwargs)
        if kind == 'cqr':
            cfg.setdefault('base_alpha', .8)
        ident = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
        items.append(dict(id=ident, config=cfg, trials=trials, sources=[source],
                          transforms=[dict(name='capped', shift=0.)] if kind == 'cqr' else []))
    for d in [2, 4]:
        for n in [20, 50, 100, 300]:
            add('cqr', d, n, 10, 'exps.ipynb: exploratory quantile sweep')
    add('cqr', 2, 30, 10, 'exps.ipynb: quantile probes', noise_levels=[1, 2])
    add('cqr', 2, 30, 10, 'exps.ipynb: quantile probes')
    add('absolute', 4, 30, 200, 'smoke_test_coordinate_lengths.ipynb',
        n_train=6400, n_test=1600, n_features=6, alpha=.2)
    add('absolute', 10, 50, 200, 'smoke_test_dependent_noise.ipynb',
        n_train=6400, n_test=1600, n_features=5,
        generator_kwargs=dict(correlation=.5, correlation_structure='equicorrelated'))
    add('cqr', 2, 20, 200, 'smoke_test_cqhr.ipynb', n_features=5, base_alpha=.5,
        quantile_params=dict(PARAMS, learning_rate=.08))
    add('cqr', 3, 30, 100, 'smoke_test_cqr.ipynb', n_train=1600, n_test=400,
        n_features=6, noise_levels=[1,5,10], base_alpha=.9,
        quantile_params=dict(PARAMS, n_estimators=50, max_depth=2, learning_rate=.08))
    items[-1]['transforms'].append(dict(name='shifted', shift=20.))
    # Replicate each pilot design at the requested common base miscoverage.
    for item in list(items):
        if item['config']['kind'] == 'cqr':
            cfg = dict(item['config'], base_alpha=.1)
            ident = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]
            items.append(dict(item, id=ident, config=cfg, sources=item['sources']+['requested_base_alpha_0.1']))
    dump(ROOT / 'envelope_method/notebook_settings.json', items)
    print('Notebook configurations', len(items), 'trials', sum(i['trials'] for i in items), flush=True)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_spec, item) for item in items]
        for index, future in enumerate(as_completed(futures), 1):
            print('COMPLETE notebook', index, future.result()['id'], flush=True)
    dump(OUT / 'notebook_run_status.json', dict(status='complete', configurations=len(items)))


if __name__ == '__main__':
    main()

"""Fresh fitted replacements for baseline-only configurations omitted by the old shortcut inventory."""
import hashlib
import json
from experiments import ROOT, dump, run_spec
from run_auxiliary import run as run_auxiliary


def main():
    items = []
    for config, trials, sources, note in [
        (dict(kind='absolute', d=50, n_cal=30, alpha=.1, noise_type='Laplace',
              noise_levels=list(range(50,0,-1)), n_features=10, n_train=6400, n_test=1600,
              redraw=True, generator_kwargs={}), 30,
         ['syn_exps\\laplace\\tscp_r_laplace_30sample.csv'],
         'The d=50 setting exists in Naive/Population_oracle/TSCP_GWC/TSCP_S archived tables, but not in the old shortcut table.'),
        (dict(kind='absolute', d=2, n_cal=12, alpha=.1, noise_type='Gaussian', noise_levels=[2,1],
              n_features=2, n_train=80, n_test=20, redraw=True,
              generator_kwargs=dict(correlation=.4, correlation_structure='equicorrelated')), 3,
         ['reviewer_exps/absolute_residual/_smoke/smoke_abs_res_trial.csv'],
         'Fresh smoke replication: archived table preserves d=2, n_cal=12, alpha=.1, correlation=.4, and one trial, but not training size or features. Explicit replacement uses 80 train / 20 test, two features, three fresh trials; it is not an exact replay.'),
    ]:
        ident = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
        items.append(dict(id=ident, config=config, trials=trials, sources=sources, transforms=[], provenance=note))
    dump(ROOT/'envelope_method/repair_settings.json', items)
    for item in items:
        print('Primary repair', run_spec(item,storage_mode='scores'), flush=True)
        if item['config']['d'] == 50:
            # No multidimensional local-cell search; this only adds the four
            # ordinary auxiliary baselines to the same fresh paired trials.
            run_auxiliary(item, include_full_lwc=False)
            print('Auxiliary repair complete', item['id'], flush=True)


if __name__ == '__main__':
    main()

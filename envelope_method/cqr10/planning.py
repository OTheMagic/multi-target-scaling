"""Runtime projections from saved, explicitly non-study pilots."""
import json
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from envelope_method.experiments import dump
from envelope_method.cqr10.runner import HERE, DATA, digest, experiment_grid, validate_storage


def storage_anchors(pilot, timing_file):
    """Project retention from saved pilots without rewriting or refitting them."""
    result = {}
    for row in pilot['rows']:
        directory = Path(timing_file).parent/'full_trials'/row['config_id']
        with ZipFile(directory/'trial_0000.npz') as archive:
            members = archive.infolist()
        # Budget metadata fingerprints plus duplicated CSV exports; this is an
        # engineering projection, not a newly measured compact execution.
        compact = 2.5*((directory/'trial_0000.json').stat().st_size + 256*len(members) + 2000)
        score_members = [m for m in members if m.filename.startswith(('scores_', 'base_lengths_', 'bound__'))]
        calibration = sum(m.compress_size for m in score_members
                          if m.filename in ('scores_cal.npy', 'base_lengths_cal.npy'))
        other = sum(m.compress_size for m in score_members) - calibration + 4096
        result[row['config_id']] = dict(compact=compact, calibration=calibration,
                                       other=other, full=row['files_bytes'])
    return result


def estimate_runtime(items, timing_file=DATA/'timing_pilot/study.json', *, storage='compact'):
    validate_storage(storage)
    pilot = json.loads(Path(timing_file).read_text())
    disk = storage_anchors(pilot, timing_file)
    rows = []
    for item in items:
        cfg = item['config']
        anchors = sorted([r for r in pilot['rows'] if r['n_features'] == cfg['n_features']],
                         key=lambda r: r['base_alpha'])
        if len(anchors) != 3:
            raise ValueError('Complete the three base-alpha timing pilots for each input dimension first.')
        for key in ['model', 'model_params', 'n_train', 'n_validation', 'n_test', 'd', 'save_models', 'envelope_search']:
            if cfg[key] != anchors[0]['config'][key]:
                raise ValueError(f'{key} differs from the timed configuration; benchmark it before estimating.')
        alphas = [r['base_alpha'] for r in anchors]
        interp = lambda values: float(np.interp(cfg['base_alpha'], alphas, values))
        if not min(alphas) <= cfg['base_alpha'] <= max(alphas):
            raise ValueError('Base alpha outside the measured timing range.')
        timings = anchors[0]['calibration_timings']
        ns = [t['n_cal'] for t in timings]
        if cfg['n_cal'] not in ns:
            raise ValueError('Calibration size must have a measured timing anchor.')
        factor = next(t['evaluate_seconds'] for t in timings if t['n_cal'] == cfg['n_cal']) / next(
            t['evaluate_seconds'] for t in timings if t['n_cal'] == 100)
        shifted_times = [sum(v for k, v in r['method_times'].items() if '_shifted_' in k) for r in anchors]
        evaluate = interp([r['evaluate_seconds']-s for r, s in zip(anchors, shifted_times)])*factor
        # The p=5 pilot measures three shifted-score variants on one shared fit.
        shift_anchor = next(r for r in pilot['rows'] if r['n_features'] == 5 and r['base_alpha'] == .5)
        shift_unit = sum(v for k, v in shift_anchor['method_times'].items() if '_shifted_' in k)/3
        shifts = sum(t['name'] == 'shifted' for t in cfg['transforms'])
        extra_shift = shifts*shift_unit*factor
        fitting = interp([r['fit_seconds'] for r in anchors])
        overhead = interp([r['total_seconds']-r['fit_seconds']-r['evaluate_seconds'] for r in anchors])
        total = fitting+overhead+evaluate+extra_shift
        hours = total*item['trials']/3600
        # Bound arrays vary with the number of shifted methods; use the largest
        # observed metadata/score payload as a conservative common anchor.
        compact_bytes = max(disk[r['config_id']]['compact'] for r in anchors)
        scores_bytes = max(disk[r['config_id']]['other'] +
                           disk[r['config_id']]['calibration']*cfg['n_cal']/r['config']['n_cal']
                           for r in anchors)
        full_bytes = interp([disk[r['config_id']]['full'] for r in anchors])
        storage_gb = dict(compact=compact_bytes*item['trials']/1e9,
                          scores=(compact_bytes+scores_bytes)*item['trials']/1e9,
                          full=(compact_bytes+full_bytes)*item['trials']/1e9)
        rows.append(dict(config_id=digest(cfg)[:16], n_features=cfg['n_features'],
            n_outcomes=10, base_alpha=cfg['base_alpha'], n_cal=cfg['n_cal'], trials=item['trials'],
            shifted_variants=shifts, fit_seconds=fitting, method_seconds=evaluate+extra_shift,
            seconds_per_trial=total, serial_hours=hours, planning_low_hours=.75*hours,
            planning_high_hours=1.6*hours, optional=item.get('optional', False),
            storage=storage, estimated_gb=storage_gb[storage],
            compact_estimated_gb=storage_gb['compact'], scores_estimated_gb=storage_gb['scores'],
            full_estimated_gb=storage_gb['full'],
            shift_incremental_seconds=extra_shift*item['trials']))
    return pd.DataFrame(rows)


def experiment_totals(table):
    groups = []
    for p in sorted(table['n_features'].unique()):
        t = table[(table.n_features == p) & ~table.optional]
        selections = [
            ('Base alpha 0.1: calibration-size sweep', t[t.base_alpha == .1]),
            ('Base alpha 0.5: calibration-size sweep', t[t.base_alpha == .5]),
            ('Base-alpha sweep: additional 0.3, 0.7, 0.9 settings', t[t.base_alpha.isin([.3, .7, .9])]),
        ]
        for name, subset in selections:
            groups.append(dict(experiment=name, n_features=int(p), trials=int(subset.trials.sum()),
                               serial_hours=float(subset.serial_hours.sum()),
                               planning_low_hours=float(subset.planning_low_hours.sum()),
                               planning_high_hours=float(subset.planning_high_hours.sum())))
        extra = table[(table.n_features == p) & table.optional]
        if len(extra):
            groups.append(dict(experiment='Optional n_cal=500,1000 at base alpha 0.1', n_features=int(p),
                trials=int(extra.trials.sum()), serial_hours=float(extra.serial_hours.sum()),
                planning_low_hours=float(extra.planning_low_hours.sum()), planning_high_hours=float(extra.planning_high_hours.sum())))
    return pd.DataFrame(groups)


def save_report():
    DATA.mkdir(parents=True,exist_ok=True)
    table = estimate_runtime(experiment_grid(True))
    table.to_csv(DATA/'runtime_estimates.csv', index=False)
    groups = experiment_totals(table)
    groups.to_csv(DATA/'experiment_estimates.csv', index=False)
    core = table[~table.optional]
    dump(DATA/'runtime_estimates.json', dict(
        assumptions='Single worker; six fitted pilot trials; alpha .3/.7 training times interpolated; '
                    'calibration-size costs measured separately on fixed fitted pilots for timing only. '
                    'Planning range 0.75x to 1.6x is an engineering allowance, not a confidence interval. '
                    'All methods within a trial share its fit. No full LWC. '
                    'Runtime retains full-save pilot overhead; no compact speedup measured. '
                    'Compact disk budgets saved JSON plus fingerprint/CSV overhead; scores disk uses '
                    'saved compressed score members with calibration-size scaling; full uses saved pilot files.',
        core_trials=int(core.trials.sum()), core_hours=float(core.serial_hours.sum()),
        default_storage='compact', core_estimated_gb=float(core.estimated_gb.sum()),
        core_storage_estimated_gb={tier: float(core[f'{tier}_estimated_gb'].sum())
                                  for tier in ['compact', 'scores', 'full']},
        configurations=table.to_dict('records'),
        experiments=groups.to_dict('records')))
    lines = ['# Ten-Outcome Runtime Estimates', '',
             'Measured on this Windows machine, one computation worker and one numerical-library thread.',
             'These are projections from six fresh fitted pilot trials, not completed study results.', '',
             '| Experiment | Input features | Fresh trials | Central time | Planning range |',
             '|---|---:|---:|---:|---:|']
    for row in groups.to_dict('records'):
        lines.append(f'| {row["experiment"]} | {row["n_features"]} | {row["trials"]} | '
                     f'{row["serial_hours"]:.2f} h | {row["planning_low_hours"]:.2f}-{row["planning_high_hours"]:.2f} h |')
    hours = core.serial_hours.sum()
    lines += ['', f'**Core total: {int(core.trials.sum())} trials, {hours:.2f} hours '
              f'(planning allowance {hours*.75:.2f}-{hours*1.6:.2f} hours).**', '',
              f'Default compact storage: approximately {core.compact_estimated_gb.sum():.2f} GB; '
              f'scores tier: {core.scores_estimated_gb.sum():.2f} GB; '
              f'full arrays/models: {core.full_estimated_gb.sum():.2f} GB. '
              'Reserve at least 1.5 times the chosen projection for variability.', '',
              'Disk projections use saved pilot metadata/member sizes, with allowances for fingerprints '
              'and CSV copies. Compact and scores sweeps have not been benchmarked. Runtime still '
              'includes the historical full-save overhead; no saving-time speedup is claimed. '
              'Storage controls disk retention, not training memory or the number of model fits.', '',
              'The base-alpha sweep shares its 0.1/0.5, n_cal=100 configurations with the two '
              'calibration-size sweeps. They are counted only once. Shift sensitivity reuses '
              'the same trial fit and adds seconds overall, not hundreds of new fits. '
              'That sharing is within the same experiment trial, never a reused data pool across trials.', '',
              'The CSV gives one estimate for every individual configuration. Test size is 600; '
              'training size is 12,000, with 20% internally held out for early stopping, plus '
              '3,000 separate diagnostic validation observations. All are fresh per trial.', '',
              'These ranges are engineering allowances, not statistical confidence intervals. '
              'Alpha 0.3/0.7 fitting costs are interpolated between measured 0.1/0.5/0.9 pilots. '
              'Calibration-size overhead uses measured 30/50/100/200/500/1000 timings at alpha 0.1. '
              'Rare difficult fits, thermal throttling, competing work, and different software can change timings. '
              'Do not divide the single-worker estimate by the number of logical CPUs. '
              'The notebook supports bounded process parallelism but defaults to one worker.', '',
              'Full LWC is excluded. The inexpensive TSCP_R shortcut is still included.', '']
    (HERE/'RUNTIME_ESTIMATES.md').write_text('\n'.join(lines), encoding='utf-8')
    print(groups.to_string(index=False))
    print(f'Core: {hours:.2f} h; {core.estimated_gb.sum():.2f} GB')


if __name__ == '__main__':
    save_report()

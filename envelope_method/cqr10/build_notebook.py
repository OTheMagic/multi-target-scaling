"""Rebuild the manual notebook from structured cells without stored run output."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def cell(kind, source):
    out = dict(cell_type=kind, metadata={}, source=source.strip().splitlines(keepends=True))
    if kind == 'code':
        out.update(execution_count=None, outputs=[])
    return out


cells = [
    cell('markdown', '''
# Ten-outcome quantile regression experiments

This notebook is ready for a local, checkpointed run. **The full study is OFF by default.**
It extends all 22 primary three-outcome CQR settings to ten outcomes: 1,100 fresh
trials with five input features, and 330 with ten input features. Two-dimensional
illustrations and fixed-dataset real experiments are outside this dimension extension.

Each trial independently draws 12,000 training, 3,000 diagnostic validation,
`n_cal` calibration and 600 test observations, and fits 20 quantile models.
Methods share that trial's data and fit. Nothing is reused between distinct trials
except the fixed data-generating coefficients. Resuming an already completed
checkpoint does not constitute a new trial. No full-LWC implementation is called.

Default estimator: gradient boosting with a **training-fitted linear initializer**,
quantile loss, target standardization, at most 800 trees, and early stopping on
20% of that trial's training observations. The remaining 3,000 validation points
are diagnostic only. The initializer is learned, not the known true regression.
This is an explicit model change from constant-initialized 100-tree boosting.
The DGP is linear Gaussian; the same training choice needs new validation for
nonlinear or heteroskedastic generators.

Training quality, quantile crossings, tail rates, pinball loss and noise-normalized
error from the known Gaussian conditional quantiles are saved for every trial.
Quality flags never discard a trial or trigger test/calibration-driven refitting.
Compact saving is the default: all trial metrics and diagnostics remain available,
while raw observations and fitted-model files are not retained.

Install dependencies once in the notebook's Python environment if needed:
`python -m pip install -r envelope_method/cqr10/requirements.txt`
'''),
    cell('code', '''
from pathlib import Path
import os
import sys

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'utility/envelope.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Open this notebook inside the multi-target-scaling repository.')
sys.path.insert(0, str(ROOT))
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp/mpl'))

import json
import numpy as np
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print  # Also permit a plain-Python validation of all code cells.
from envelope_method.cqr10.runner import (
    HERE, DATA, DEFAULT_OUT, experiment_grid, run_study, summarize_config, digest, versions)
from envelope_method.cqr10.planning import estimate_runtime, experiment_totals
print('Repository:', ROOT)
print('Environment:', versions())
'''),
    cell('markdown', '''
## Workload selection and estimates

The original calibration sizes are 30, 50, 100 and 200. The original base-alpha
sweep is 0.3, 0.5, 0.7 and 0.9, with the requested fair envelope/CQHR comparison
at base alpha 0.1 added. Both methods use exactly the same base intervals.
Target joint miscoverage is 0.1 throughout. Noise standard deviations are 10
through 1, extending the existing descending-scale convention.

Set `INCLUDE_LARGE_CALIBRATION=True` to add 500 and 1,000 calibration observations
at base alpha 0.1 (260 additional fresh trials). These are optional new settings.
The base-alpha and calibration sweeps overlap; the grid deduplicates their common
settings. Shift sensitivity uses the same trial fits as the appropriate base-0.5
setting, not another fitted cohort.

Timing estimates are from six saved single-trial pilots on this machine, one
worker. They are engineering projections, not guarantees or confidence intervals.
Changing sample sizes/model parameters requires a new timing pilot. Bounded
multi-process execution is available, but CPU count is not a measured speedup.
'''),
    cell('code', '''
RUN_EXPERIMENTS = False
INCLUDE_LARGE_CALIBRATION = False
INPUT_FEATURES = [5, 10]
BASE_ALPHAS = [.1, .3, .5, .7, .9]
TRIALS_OVERRIDE = None  # Keep None for 100 / 30 trials from the original grids.
WORKERS = 1
STORAGE = 'compact'  # 'scores' retains scores/base widths/bounds; 'full' also saves raw data/models.
OUTPUT = DEFAULT_OUT  # On a cluster, choose an explicit Path('/scratch/.../cqr10').

items = [item for item in experiment_grid(INCLUDE_LARGE_CALIBRATION)
         if item['config']['n_features'] in INPUT_FEATURES
         and item['config']['base_alpha'] in BASE_ALPHAS]
if TRIALS_OVERRIDE is not None:
    if not isinstance(TRIALS_OVERRIDE, int) or TRIALS_OVERRIDE < 1:
        raise ValueError('TRIALS_OVERRIDE must be a positive integer.')
    items = [dict(item, trials=TRIALS_OVERRIDE) for item in items]
estimates = estimate_runtime(items, storage=STORAGE)
display(estimates[['n_features', 'base_alpha', 'n_cal', 'trials',
                   'seconds_per_trial', 'serial_hours', 'planning_low_hours', 'planning_high_hours']])
display(experiment_totals(estimates))
print(f"Total: {sum(i['trials'] for i in items)} trials, "
      f"{estimates.serial_hours.sum():.2f} single-worker hours, "
      f"approximately {estimates.estimated_gb.sum():.2f} GB with {STORAGE} storage.")
print('Output directory:', OUTPUT.resolve())
'''),
    cell('markdown', '''
## Inspect the saved training-quality pilot

The six full pilots use independent seeds outside the formal study. The flags
are descriptive diagnostics: per-model oracle RMSE greater than 0.25 noise SD
or validation tail-frequency error greater than 0.04. Passing is evidence of good
training on these pilot draws, not a theorem about every future fit. Oracle
quantiles are used only to measure error, never passed to the estimator.
'''),
    cell('code', '''
pilot = json.loads((DATA / 'timing_pilot/study.json').read_text())
quality = pd.DataFrame([
    dict(n_features=r['n_features'], base_alpha=r['base_alpha'], **q)
    for r in pilot['rows'] for q in r['quality']])
display(quality.groupby(['n_features', 'base_alpha']).agg(
    models=('coordinate', 'size'), max_normalized_rmse=('normalized_oracle_rmse', 'max'),
    mean_normalized_rmse=('normalized_oracle_rmse', 'mean'),
    flagged_models=('quality_flag', 'sum'), tree_cap_hits=('hit_tree_cap', 'sum')))
'''),
    cell('markdown', '''
## Run or resume

Set `RUN_EXPERIMENTS=True` above and execute the cell below. Each completed trial
is immediately checkpointed. Interrupting and rerunning resumes verified completed
trials and restarts only unfinished ones. Ordinary interrupts clean their locks;
after a kernel/process crash, remove a stale `.lock` only after confirming that
its worker is no longer running. Configuration, implementation and library-version
mismatches refuse to mix cohorts; use a new output directory after such changes.

`STORAGE='compact'` saves JSON/CSV metrics, coordinates, diagnostics, seeds and
provenance for **every trial**, with a checksum for the completed JSON record.
`'scores'` additionally keeps signed calibration/validation/test scores, base
interval widths and method bounds; `'full'` also keeps raw observations,
quantile predictions and all 20 fitted models. Storage does not change trial
IDs/seeds or fitting, and does not reduce the RAM needed while a trial runs.
Use full mode selectively in a separate output folder for detailed examples.
An existing richer checkpoint can satisfy a compact request without deleting
anything. Upgrading an existing compact/scores checkpoint is rejected: choose
a new output folder to deterministically redraw and refit that cohort.
Configuration summaries and paired comparisons are rebuilt after each setting.
Keep all quality-flagged and infinite-region trials in the results. Infinite
fallbacks in capped methods are possible at small calibration sizes.

The signed envelope uses the proved rank-localized search. `TSCP_R` denotes
the cheap old shortcut, **not** full LWC. Signed GWC is produced jointly with the
envelope; its separate timing is unavailable, not zero.
'''),
    cell('code', '''
if RUN_EXPERIMENTS:
    completed = run_study(items, out=OUTPUT, workers=WORKERS, storage=STORAGE)
    print('Completed configurations:', len(completed))
else:
    print('Study not started. Set RUN_EXPERIMENTS=True to run locally.')
'''),
    cell('markdown', '''
## Summaries and paired comparisons

Uncertainty must use independent fitted trials as the units, not treat all test
points as independent of their shared calibration set. Mean paired volume
reduction and ratio of mean volumes are different statistics. The exported
paired report names each and counts nonfinite/zero-volume exclusions explicitly.
Ten-dimensional volumes can be very large; log volumes are saved as well.
Summary CSVs include mean, SD, median and empirical 5/25/75/95 percentiles;
per-trial metrics remain the source for pairing and revised plots.
'''),
    cell('code', '''
all_records = []
for item in items:
    directory = OUTPUT / digest(item['config'])[:16]
    if directory.exists():
        summarize_config(item['config'], OUTPUT)
    for checkpoint in sorted(directory.glob('trial_*.json')):
        result = json.loads(checkpoint.read_text())
        all_records.extend(result['records'])
results = pd.DataFrame(all_records)
if results.empty:
    print('No formal study results yet. Timing pilots are kept separately.')
else:
    results.to_csv(OUTPUT / 'all_trials.csv', index=False)
    display(results.groupby(['n_features', 'base_alpha', 'n_cal', 'method'])[
        ['test_coverage', 'outcome_volume', 'mean_log_volume']].agg(['mean', 'std', 'count']))
    print('Quality-flagged fitted trials:', results.groupby(['config_id', 'trial']).quality_flag.any().sum())
'''),
    cell('code', '''
if not results.empty:
    FIGURES = HERE / 'figures'
    FIGURES.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    methods = ['Envelope_signed', 'CQHR', 'TSCP_R_capped_0', 'Envelope_capped_0', 'Signed_GWC']
    for p in sorted(results.n_features.unique()):
        selected = results[(results.n_features == p) & (results.base_alpha == .1)]
        if selected.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for method in methods:
            data = selected[selected.method == method]
            grouped = data.groupby('n_cal').test_coverage
            means, se = grouped.mean(), grouped.sem()
            axes[0].errorbar(means.index, means, yerr=1.96*se.fillna(0), marker='o', label=method)
        axes[0].axhline(.9, color='black', linestyle='--', linewidth=1)
        axes[0].set(xlabel='Calibration size', ylabel='Joint coverage', title=f'10 outcomes, {p} input features')
        for n, data in selected.groupby('n_cal'):
            wide = data.pivot(index='trial', columns='method', values='outcome_volume')
            valid = np.isfinite(wide.Envelope_signed) & np.isfinite(wide.CQHR) & (wide.CQHR > 0)
            ratios = wide.loc[valid, 'Envelope_signed']/wide.loc[valid, 'CQHR']
            if len(ratios):
                se = ratios.std(ddof=1)/np.sqrt(len(ratios)) if len(ratios)>1 else 0
                axes[1].errorbar(n, ratios.mean(), yerr=1.96*se, fmt='o', color='tab:green')
        axes[1].axhline(1, color='black', linestyle='--', linewidth=1)
        axes[1].set(xlabel='Calibration size', ylabel='Mean paired envelope/CQHR volume ratio',
                    title='Finite pairs only; exclusions in paired report')
        axes[0].legend(fontsize=7)
        fig.savefig(FIGURES / f'cqr10_base01_features{p}.png', dpi=160)
        plt.show()
'''),
]


def main():
    notebook = dict(nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                  language_info=dict(name='python', version='3.11')),
                    cells=[dict(c, id=f'cqr10-{i:02d}') for i, c in enumerate(cells)])
    path = HERE/'cqr10_experiments.ipynb'
    path.write_text(json.dumps(notebook, indent=1)+'\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    main()

# Ten-Outcome CQR Experiments

Open `cqr10_experiments.ipynb` locally. The study is disabled until
`RUN_EXPERIMENTS=True`. Every empirical trial generates new training, validation,
calibration and test observations and refits all models. Full LWC is excluded.
The default saving tier is **compact**: metrics and diagnostics for every trial,
with no raw-observation NPZ or fitted-model joblib files. Set `STORAGE='scores'`
to retain scores and bounds for future conformal comparisons, or `'full'` for
selected detailed examples. This choice affects disk retention, not the science
or peak memory used to fit a trial.

The CLI is also suitable for a cluster (the first command only previews):

```bash
python envelope_method/cqr10/runner.py --storage compact --out /scratch/my-project/cqr10
python envelope_method/cqr10/runner.py --run --workers 4 --storage compact --out /scratch/my-project/cqr10
```

Choose the output path appropriate to the machine. The default remains the
local `data/envelope_method/cqr10/results/` directory, and one worker remains the default.

Timing pilots, numerical estimates, and verification JSON are under `data/envelope_method/cqr10/`. The notebook, code, documentation, and generated `figures/` remain beside this README. These local data paths are excluded from GitHub.

## Scope

- Extend all 22 primary synthetic CQR settings in `../settings.json` from three
  outcomes to ten, preserving the base-alpha/calibration-size grids, shifts and
  trial counts: 1,100 trials with five input features, 330 with ten input features.
- Training size 12,000, external diagnostic validation size 3,000, test size 600.
  Each quantile learner reserves 20% of the fresh training sample for early
  stopping (9,600 fitting observations and 2,400 stopping observations).
- Independent linear-Gaussian outcomes conditional on X, with Gaussian noise SDs
  10 through 1. Fixed DGP coefficients, fresh independent observations per trial.
- Joint target miscoverage 0.1; both envelope and CQHR use the same fitted base
  intervals and the same base miscoverage. The focused comparison uses 0.1.
- Optional calibration sizes 500 and 1,000 at base alpha 0.1 add 260 trials.
- Two-dimensional figures are intentionally still two-dimensional; this does not
  manufacture ten-outcome real datasets or rerun unrelated absolute-score studies.

## Training Choice

The old 100-tree, 2,400-training-observation models underfit the low-noise outcome
in a bounded diagnostic pilot. On that outcome's upper 0.95 quantile, normalized
oracle RMSE was 9.69 for the old settings and 1.16 for 800-tree plain boosting.
Linear-initialized boosting reduced it to 0.091. These are two-coordinate pilot
comparisons, not universal estimates of each learner's accuracy.

The default remains gradient-boosted quantile regression, but **changes its
initialization** to a linear predictor fitted only on the training data. The
Gaussian DGP has a linear conditional mean, making this a suitable starting
point. Trees then optimize quantile loss. No true coefficients, noise scales,
calibration labels or test labels enter fitting. Targets are standardized from
the training sample. Parameters: learning rate 0.05, maximum depth 3, minimum
leaf size 20, at most 800 trees, early-stopping patience 60, tolerance 1e-5.
The estimator's initialization and stopping interfaces are documented by
[scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html).

Six full pilots cover both feature counts and base alphas 0.1, 0.5 and 0.9.
All 120 fitted models pass the diagnostic thresholds (normalized oracle RMSE
at most 0.25; validation quantile-tail error at most 0.04). The observed maximum
RMSE is below 0.15 noise SD and no fit reaches the 800-tree cap. Narrow base
intervals at base alpha 0.9 have occasional raw quantile crossings, which are
recorded and ordered coordinatewise before constructing scores, matching the
existing convention. These diagnostics do not promise that every future fit is
well trained. All future quality-flagged trials remain in the saved cohort.

The oracle is used only for diagnostics on the separate validation sample, not
model selection within a formal trial. The model family was selected from an
independent preliminary pilot. New nonlinear/misspecified DGPs require new
training checks; the present quality evidence does not extend to them.

## Time and Disk

See `RUNTIME_ESTIMATES.md` for grouped costs and `runtime_estimates.csv` for
every setting. The core study is estimated at 11.8 single-worker hours, with an
engineering planning allowance of 9-19 hours. Current disk projections are
approximately **0.16 GB compact**, **1.03 GB scores**, or **5.76 GB full**;
reserve at least 1.5 times the chosen estimate. Compact/scores projections use
saved pilot metadata and compressed member sizes plus export overhead, not a
completed storage benchmark. Raw observations/models account for most full-mode
storage; saving only aggregates would sacrifice trial pairing for little further gain.

The runtime estimates include model fitting, prediction, conformal evaluation and
historical full-mode saving. No compact-mode speedup is assumed. Measured pilot
trials take 19-27 seconds with five features
and 30-47 seconds with ten features. Calibration evaluation is roughly 1.2 seconds
per 600-test-point trial at n_cal=100; it is much cheaper than training here.
Training times at base alphas 0.3 and 0.7 are interpolations. Optional larger
calibration sizes add about 2.3 hours; their tier-specific disk projections are
listed separately in `runtime_estimates.csv`. No measured parallel speedup
is claimed; one worker is the reproducible default. Multiple workers operate on
different configurations and use one numerical-library thread each.

Within-trial paired comparisons share a fit, so costs must not be charged again
for CQHR, old shortcuts or score shifts. Common configurations in overlapping
experiment sweeps are counted once. This is not reuse between distinct trials.

## Saved Evidence and Resume

`OUTPUT/<configuration-hash>/` always contains:

- `config.json`: exact configuration, software versions and implementation hash.
- `trial_NNNN.json`: per-method measurements, all seeds, model diagnostics,
  stage timings, storage tier, code/environment provenance, and array hashes
  with shapes/dtypes. A content checksum protects the metadata itself. This JSON
  is written last as the completed-trial marker. Hashes are provenance witnesses;
  they cannot reconstruct discarded arrays.
- `trials.csv`, `model_quality.csv`, `summary.csv`, `volume_states.csv`, `paired_comparisons.json`:
  per-configuration exports. The notebook additionally creates combined tables
  and figures after data are available. Summary CSVs report means, SDs, medians
  and empirical 5/25/75/95 percentiles (`higher` order-statistic convention).
  Zero, infinite and invalid volume counts are separate; pairing still uses
  every retained trial record rather than only aggregate summaries.

Additional artifacts depend on the requested tier:

| Tier | NPZ contents | Fitted models |
|---|---|---|
| `compact` (default) | No NPZ | No joblib |
| `scores` | Signed validation/calibration/test scores, corresponding base widths and method bounds | No joblib |
| `full` | All score-tier arrays plus every raw split, DGP coefficients and raw quantile predictions | All 20 estimators in `trial_NNNN.joblib` |

Artifact SHA-256 hashes and byte counts are checked on resume. Missing required
files fail verification instead of being treated as a completed compact trial.
Only load trusted local model files; joblib uses a pickle-based format.
Incomplete trials are deterministically redrawn and refitted, with no reuse of
partially fitted models. Leftover richer artifacts cannot silently become a
compact checkpoint.

Resuming verifies hashes and refuses to mix implementations or library versions.
Use a separate output directory after changing code or the environment. Do not
delete stale locks until their worker processes have stopped. The default
configuration hash does not depend on the requested number of repetitions, so
increasing the trial count extends an existing compatible cohort.
Storage is a run option outside the scientific configuration, so changing the
tier alone does not change IDs or seeds. The historical `save_models` configuration
field is retained only for ID compatibility; the tier controls serialization.
Requesting compact for an existing full/scores checkpoint returns that verified
richer checkpoint without deleting its files. Requesting more data than a saved
checkpoint contains fails explicitly: use a separate output directory and refit
with the same configuration/seeds. There is no automatic tier upgrade or downgrade.
Historical pilot checkpoints remain read-only verifiable; their files are unchanged.

Coverage uncertainty uses trials as independent units. Nonfinite or zero
comparator volumes are counted separately in finite-pair comparisons, not hidden.
Small-calibration capped scores can have zero variance and yield whole-space
fallbacks; signed scores are not capped. `TSCP_R` is the inexpensive old shortcut,
not full LWC. Signed-GWC bounds are obtained during envelope evaluation, so its
separate timing is marked unavailable. No uniform superiority claim is inferred
from a timing pilot.

## Files and Verification

- `benchmark_models.py`: bounded comparison of training strategies.
- `benchmark_study.py`: six fresh end-to-end timing/quality pilots, saved under
  `timing_pilot/`, outside formal results. Additional calibration-size microtimings
  reuse fitted pilot models solely to isolate runtime, never as empirical trials.
- `runner.py`: importable Windows/Jupyter-compatible checkpoint runner.
- `planning.py`: saved timing projections and reports.
- `build_notebook.py`: structured notebook source.
- `test_runner.py`: fresh draws, no leakage, checkpoint corruption detection,
  model reproduction, rank-search equivalence and two-process execution tests.
- `verify_delivery.py`: checks all pilot archives/models, reproduces saved
  predictions/scores/metrics, and executes the notebook with training blocked.

Install `requirements.txt` in the chosen local Jupyter environment. Tests also
need `pytest`; delivery verification also needs `nbformat` (included with Jupyter).
No formal sweep has been launched. Previous cohorts and obsolete-data quarantine
are unchanged.

Storage revision checks passed for 21 runner tests, all six pilot archives/models,
and execution of all six notebook code cells with training blocked. Notebook
structure was checked directly. Live Jupyter and nbformat schema validation
could not be performed in the delivery environment because those packages were
unavailable; the notebook lists their local installation requirements.

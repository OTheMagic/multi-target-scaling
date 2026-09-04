# Real-Data Search and Coordinate Diagnostics

This is a new diagnostic rerun, not a claim that the original fitted models or
historical wall-clock measurements were recovered. The original six datasets,
200 hash-seeded splits, 75%/5%/20% allocation, model hyperparameters, and four
displayed methods are retained. The original result files are not overwritten.

## Completed Findings

- At miscoverage 0.1, neither backward search nor fallback occurred in any of
  the 200 splits per dataset. All 10,200 coordinate searches used the binary
  branch. Candidate inspections averaged 4.00--8.49 per coordinate, and mean
  whole-call construction times ranged from 0.466 to 7.239 ms across datasets.
- Zero events in 200 splits give a pointwise one-sided 95% upper probability
  bound of approximately 1.49%, conditional on the fixed dataset. This is not
  proof that a branch is impossible or a simultaneous statement over datasets.
- The stress sweep reached backward search on energy: 3/200 splits at
  miscoverage 0.5 and 10/200 at both 0.7 and 0.9. Every backward coordinate
  inspected one candidate. Fallback occurred on all datasets at the most
  extreme level. These diagnostics exercise the branches, not worst-case scans.
- All 51 real-data coordinates are reported. TSCP reallocates width rather
  than shortening every interval: energy's mean paired ratios to Unscaled Max
  are (0.36, 1.28), while all three student ratios exceed one.
- The nominal 0.9 lies within one across-split standard deviation of every
  new TSCP joint-coverage mean. This is a variability check, not a confidence
  interval for the mean; the observed means were not adjusted to pass it.
- All 1,200 default-target TSCP rectangles match the pristine implementation
  exactly, and all 63 instrumentation regression tests passed.

## Reproduce

From the repository root in a Python environment with the packages in
`reviewer_update/requirements_real_diagnostics.txt`:

```powershell
python reviewer_update/run_real_diagnostics.py prepare
python reviewer_update/run_real_diagnostics.py fit --trials 200 --workers 6
python reviewer_update/run_real_diagnostics.py evaluate --trials 200
python reviewer_update/build_real_diagnostics.py
python reviewer_update/validate_real_diagnostics.py
python -m pytest reviewer_update/test_search_diagnostics.py -q
```

On the machine used for these runs, the Python executable is
`C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`.
Set `PYTHONPATH` to `E:\multi-target-scaling\tmp\diagnostic_packages;E:\multi-target-scaling`
to use the workspace-local dependency installation. An ordinary virtual
environment with the pinned requirements is also sufficient.

The preparation step reads the three local ARFF files and downloads the three
public UCI datasets through the original loaders. Its local cache contains the
features, targets, target names, model configurations, package versions, and
data fingerprints. Each fitted split stores its exact row indices and
calibration/test residual matrices. The fit phase resumes existing splits.
Do not reuse that cache for changed models, datasets, or split definitions;
use a separately named output directory for a different study.

Model fits are parallelized across splits. Each estimator and transformer uses
one job, without changing its statistical hyperparameters. The timing phase
runs serially after fitting has finished, with native thread pools limited to
one thread. Do not run it concurrently with expensive jobs if comparable
wall-clock measurements are required.

## Measurements

- `real_joint_trials.csv`: joint test coverage and residual-space volume for
  four methods, with sample sizes and split identifiers.
- `real_coordinate_trials.csv`: all outcome coordinates, full interval lengths
  (twice the residual thresholds), marginal coverage, and per-split length
  ratios relative to Unscaled Max. Point CHR's infinite stock intervals remain
  infinite; they are not truncated or averaged over only finite trials.
- `real_search_trials.csv`: actual TSCP fallback and backward-branch events at
  miscoverage 0.1, candidate-cell counts, and construction times.
- `real_search_coordinate_trials.csv`: the branch and candidate-cell count for
  each coordinate. Counts include inspected cells rejected by intersection
  checks, not only cells requiring a score-quantile evaluation.
- `real_timing_repeats.csv`: seven uninstrumented TSCP timings after warm-up per
  split. Each split's timing is the median; reported mean times average those
  medians. Instrumented calls are excluded from the timings.
- `real_search_alpha_trials.csv`: a computational stress sweep over miscoverage
  0.1, 0.3, 0.5, 0.7, and 0.9 using the same cached residuals. The additional
  levels use three timing repeats per split. They are not extra 90%-coverage
  benchmarks and do not estimate worst-case complexity.
- Summary CSVs retain across-split standard deviations. Marginal coverage
  intervals are pointwise Monte Carlo approximations conditional on the fixed
  dataset, not population-level or simultaneous uncertainty statements.
- Search summaries distinguish an observed zero event count from an
  unobserved conditional runtime. The latter is missing, not zero. Coordinate
  denominators exclude fallback runs; split-event denominators include all
  runs. Binomial upper limits apply to split events, not dependent coordinates.

## Verification and Outputs

`test_search_diagnostics.py` checks optional instrumentation against a pristine
copy of the pre-instrumentation code, including a genuine backward-search
fixture and fallback cases. `validate_real_diagnostics.py` then checks all
1,200 real-data splits against that pristine implementation and verifies the
stored indices, coordinate metrics, timing denominators, and event counts.

Four vector PDF figures are written to `reviewer_update/figures` and copied to
the manuscript's `figures` folder. CSV summaries, trial records, generated
tables, and the run manifest are copied to
`multi_target_scaling_latex/experiment_data/real_diagnostics`. The raw fit cache
is not needed to compile the manuscript. The final verification report is
`verification.json` beside this README.
